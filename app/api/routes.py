"""HTTP routes.

The chat route is intentionally provider-agnostic: it normalizes the request,
asks the router for an adapter, validates content support, then delegates to the
adapter via the canonical interface. It contains no OpenAI/Gemini specifics.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.config import MODEL_REGISTRY
from app.models.canonical import CanonicalLLMResponse
from app.models.openai_contract import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)
from app.models.usage import UsageStatsResponse, UsageSummaryResponse
from app.services.normalizer import normalize_request, validate_content_support
from app.services.router import ProviderRouter
from app.services.streaming import UsageCollector, sse_stream
from app.services.usage import aggregate, ensure_utc, now_utc, record_usage, summarize
from app.services.usage_store import UsageStore
from app.utils.ids import new_completion_id, new_request_id
from app.utils.logging import get_logger

logger = get_logger()
router = APIRouter()

# Default usage window when the caller does not specify a range.
_DEFAULT_USAGE_WINDOW = timedelta(days=30)


def _provider_router(request: Request) -> ProviderRouter:
    return request.app.state.provider_router


def _usage_store(request: Request) -> UsageStore:
    return request.app.state.usage_store


def _record_usage_safely(request: Request, canonical, usage, *, stream: bool) -> None:
    """Record usage without ever breaking the request path."""
    try:
        record_usage(_usage_store(request), canonical, usage, stream=stream)
    except Exception:  # pragma: no cover - defensive
        logger.exception("usage.record_failed")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ai-gateway"}


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    data = [
        ModelCard(id=alias, provider=cfg["provider"], provider_model=cfg["provider_model"])
        for alias, cfg in MODEL_REGISTRY.items()
    ]
    return ModelList(data=data)


def _build_usage(usage) -> Usage | None:
    if usage is None:
        return None
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


def _build_response(
    canonical_response: CanonicalLLMResponse, model_alias: str, completion_id: str, created: int
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_alias,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(content=canonical_response.content),
                finish_reason=canonical_response.finish_reason or "stop",
            )
        ],
        usage=_build_usage(canonical_response.usage),
    )


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    request_id = request.headers.get("x-request-id") or new_request_id()
    started = time.perf_counter()

    # Normalize -> route -> validate. Each step raises a GatewayError that the
    # registered exception handler turns into a consistent JSON error.
    canonical = normalize_request(body)
    provider = _provider_router(request).get(canonical.provider)
    validate_content_support(
        canonical, provider.supported_content_types(canonical.provider_model)
    )

    completion_id = new_completion_id()
    created = int(time.time())
    log_ctx = {
        "request_id": request_id,
        "model_alias": canonical.model_alias,
        "provider": canonical.provider,
        "provider_model": canonical.provider_model,
        "stream": canonical.stream,
    }
    logger.info("chat.completions.start", extra={"context": log_ctx})

    if canonical.stream:
        # Surface configuration errors (e.g. missing key) before the SSE stream
        # starts, so they become a normal HTTP error rather than a stream event.
        provider.ensure_ready()

        collector = UsageCollector()

        async def event_source():
            async for chunk in sse_stream(provider, canonical, completion_id, created, collector):
                yield chunk
            _record_usage_safely(request, canonical, collector.usage, stream=True)
            logger.info(
                "chat.completions.stream_done",
                extra={"context": {**log_ctx, "latency_ms": _elapsed_ms(started)}},
            )

        return StreamingResponse(event_source(), media_type="text/event-stream")

    try:
        result = await provider.complete(canonical)
    except Exception:
        logger.exception("chat.completions.error", extra={"context": log_ctx})
        raise
    _record_usage_safely(request, canonical, result.usage, stream=False)
    logger.info(
        "chat.completions.done",
        extra={"context": {**log_ctx, "latency_ms": _elapsed_ms(started)}},
    )
    return _build_response(result, canonical.model_alias, completion_id, created)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _resolve_window(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    end = ensure_utc(end) if end else now_utc()
    start = ensure_utc(start) if start else end - _DEFAULT_USAGE_WINDOW
    return start, end


@router.get("/v1/usage", response_model=UsageStatsResponse)
async def usage_stats(
    request: Request,
    start: datetime | None = Query(None, description="Start of the window (ISO 8601); default end-30d"),
    end: datetime | None = Query(None, description="End of the window (ISO 8601); default now"),
    provider: str | None = Query(None, description="Filter to a single provider"),
    interval: str | None = Query(
        None, pattern="^(day|week|month)$", description="Optional time bucketing"
    ),
) -> UsageStatsResponse:
    """Token usage by provider and modality over a time window, with estimated cost."""
    start, end = _resolve_window(start, end)
    pricing = request.app.state.pricing
    await pricing.refresh_if_stale()
    records = _usage_store(request).query(start, end, provider)
    return aggregate(records, start=start, end=end, interval=interval, price_of=pricing.get)


@router.get("/v1/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    request: Request,
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    provider: str | None = Query(None),
) -> UsageSummaryResponse:
    """Overall usage totals plus estimated cost for a time window."""
    start, end = _resolve_window(start, end)
    pricing = request.app.state.pricing
    await pricing.refresh_if_stale()
    records = _usage_store(request).query(start, end, provider)
    return summarize(records, start=start, end=end, price_of=pricing.get)
