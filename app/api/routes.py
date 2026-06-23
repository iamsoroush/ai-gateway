"""HTTP routes.

The chat route is intentionally provider-agnostic: it normalizes the request,
asks the router for an adapter, validates content support, then delegates to the
adapter via the canonical interface. It contains no OpenAI/Gemini specifics.

Every routed chat request — success or failure — is persisted to the request store
as PHI-safe operational metadata (status, tokens, realized cost, latency, content
flags, caller IP/UA; never prompts or content). Usage stats are computed from those
records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.config import MODEL_CATALOG, MODEL_REGISTRY
from app.models.canonical import CanonicalLLMRequest, CanonicalLLMResponse, CanonicalUsage
from app.models.errors import GatewayError
from app.models.openai_contract import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)
from app.models.usage import (
    RequestListResponse,
    RequestRecord,
    UsageStatsResponse,
    UsageSummaryResponse,
)
from app.services.normalizer import _AUDIO_TYPES, normalize_request, validate_content_support
from app.services.pricing import PricingService
from app.services.request_store import RequestStore
from app.services.router import ProviderRouter
from app.services.streaming import UsageCollector, sse_stream
from app.services.usage import (
    aggregate,
    ensure_utc,
    estimate_cost,
    now_utc,
    summarize,
    tokens_from_usage,
)
from app.utils.ids import new_completion_id, new_request_id
from app.utils.logging import get_logger

logger = get_logger()
router = APIRouter()

# Default usage/requests window when the caller does not specify a range.
_DEFAULT_USAGE_WINDOW = timedelta(days=30)
_COST_DP = 6  # round the per-request cost snapshot to micro-dollars


def _provider_router(request: Request) -> ProviderRouter:
    return request.app.state.provider_router


def _request_store(request: Request) -> RequestStore:
    return request.app.state.request_store


def _record_request_safely(request: Request, record: RequestRecord) -> None:
    """Persist a request record without ever breaking the request path."""
    try:
        _request_store(request).record(record)
    except Exception:  # pragma: no cover - defensive
        logger.exception("request.record_failed")


def _client_info(request: Request) -> tuple[str | None, str | None]:
    """Best-effort caller IP (honoring ``X-Forwarded-For``) and user-agent.

    These are caller *infrastructure* metadata, not patient content (D6/D18). The
    ``X-Forwarded-For`` first hop is client-controlled — fine for observability,
    never for auth.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip: str | None = forwarded.split(",")[0].strip()
    elif request.client is not None:
        ip = request.client.host
    else:
        ip = None
    return ip, request.headers.get("user-agent")


def _content_flags(body: ChatCompletionRequest) -> tuple[bool, bool]:
    """Whether the request carries image / audio content (modality presence only)."""
    has_image = has_audio = False
    for message in body.messages:
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "image_url":
                    has_image = True
                elif part.type in _AUDIO_TYPES:
                    has_audio = True
    return has_image, has_audio


@dataclass
class _ReqCtx:
    """Per-request context captured up front so any path can record a record."""

    request_id: str
    started: float
    client_ip: str | None
    user_agent: str | None
    has_image: bool
    has_audio: bool
    requested_model: str | None
    stream: bool
    pricing: PricingService


def _success_record(
    ctx: _ReqCtx, canonical: CanonicalLLMRequest, usage: CanonicalUsage | None, latency_ms: float
) -> RequestRecord:
    inp, out, total, in_mod, out_mod = tokens_from_usage(usage)
    rates = ctx.pricing.get(canonical.provider_model)
    # None (not 0.0) when the model is unpriced, so the snapshot reads as "unknown".
    cost = round(estimate_cost(rates, in_mod, out_mod, inp, out), _COST_DP) if rates else None
    return RequestRecord(
        timestamp=now_utc(),
        request_id=ctx.request_id,
        status="success",
        provider=canonical.provider,
        provider_model=canonical.provider_model,
        model_alias=canonical.model_alias,
        stream=ctx.stream,
        latency_ms=latency_ms,
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        input_modality_tokens=in_mod,
        output_modality_tokens=out_mod,
        has_image=ctx.has_image,
        has_audio=ctx.has_audio,
        cost_usd=cost,
        client_ip=ctx.client_ip,
        user_agent=ctx.user_agent,
    )


def _error_record(
    ctx: _ReqCtx, canonical: CanonicalLLMRequest | None, error: Exception, latency_ms: float
) -> RequestRecord:
    if isinstance(error, GatewayError):
        error_type, error_code, http_status = error.error_type, error.code, error.status_code
    else:
        error_type, error_code, http_status = "internal_error", "internal_error", 500
    return RequestRecord(
        timestamp=now_utc(),
        request_id=ctx.request_id,
        status="error",
        provider=canonical.provider if canonical else None,
        provider_model=canonical.provider_model if canonical else None,
        model_alias=canonical.model_alias if canonical else ctx.requested_model,
        stream=ctx.stream,
        error_type=error_type,
        error_code=error_code,
        http_status=http_status,
        latency_ms=latency_ms,
        has_image=ctx.has_image,
        has_audio=ctx.has_audio,
        client_ip=ctx.client_ip,
        user_agent=ctx.user_agent,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ai-gateway"}


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    aliases = [
        ModelCard(id=alias, provider=cfg["provider"], provider_model=cfg["provider_model"])
        for alias, cfg in MODEL_REGISTRY.items()
    ]
    provider_models = [
        ModelCard(id=model_id, provider=provider, provider_model=model_id)
        for model_id, provider in MODEL_CATALOG.items()
    ]
    return ModelList(data=aliases + provider_models)


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
    client_ip, user_agent = _client_info(request)
    has_image, has_audio = _content_flags(body)
    ctx = _ReqCtx(
        request_id=request_id,
        started=time.perf_counter(),
        client_ip=client_ip,
        user_agent=user_agent,
        has_image=has_image,
        has_audio=has_audio,
        requested_model=body.model,
        stream=body.stream,
        pricing=request.app.state.pricing,
    )

    # canonical stays None until normalization succeeds, so the error path can still
    # record what it knows (the requested model name) when normalization fails.
    canonical: CanonicalLLMRequest | None = None
    try:
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
            # starts, so they become a normal HTTP error (recorded by the outer
            # except) rather than a stream event.
            provider.ensure_ready()

            collector = UsageCollector()
            canonical_stream = canonical  # bind non-None for the closure

            async def event_source():
                async for chunk in sse_stream(
                    provider, canonical_stream, completion_id, created, collector
                ):
                    yield chunk
                # Recorded exactly once here: errors after the response has started
                # cannot reach the outer except, so the collector carries them.
                latency = _elapsed_ms(ctx.started)
                if collector.error is not None:
                    record = _error_record(ctx, canonical_stream, collector.error, latency)
                else:
                    record = _success_record(ctx, canonical_stream, collector.usage, latency)
                _record_request_safely(request, record)
                logger.info(
                    "chat.completions.stream_done",
                    extra={"context": {**log_ctx, "latency_ms": latency, "status": record.status}},
                )

            return StreamingResponse(event_source(), media_type="text/event-stream")

        result = await provider.complete(canonical)
        latency = _elapsed_ms(ctx.started)
        _record_request_safely(request, _success_record(ctx, canonical, result.usage, latency))
        logger.info(
            "chat.completions.done",
            extra={"context": {**log_ctx, "latency_ms": latency}},
        )
        return _build_response(result, canonical.model_alias, completion_id, created)

    except Exception as exc:
        # Every routed request is accounted for, failures included. (Streaming
        # failures after the response has started are handled inside event_source.)
        latency = _elapsed_ms(ctx.started)
        _record_request_safely(request, _error_record(ctx, canonical, exc, latency))
        logger.exception(
            "chat.completions.error",
            extra={"context": {"request_id": request_id, "status": "error"}},
        )
        raise


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
    records = _request_store(request).query(start, end, provider)
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
    records = _request_store(request).query(start, end, provider)
    return summarize(records, start=start, end=end, price_of=pricing.get)


@router.get("/v1/requests", response_model=RequestListResponse)
async def list_requests(
    request: Request,
    start: datetime | None = Query(None, description="Start of the window (ISO 8601); default end-30d"),
    end: datetime | None = Query(None, description="End of the window (ISO 8601); default now"),
    provider: str | None = Query(None, description="Filter to a single provider"),
    model: str | None = Query(None, description="Filter by model alias or provider model"),
    status: str | None = Query(
        None, pattern="^(success|error)$", description="Filter by outcome"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Max rows (newest first)"),
) -> RequestListResponse:
    """Recent request records (newest first), as PHI-safe operational metadata."""
    start, end = _resolve_window(start, end)
    records = _request_store(request).query(
        start, end, provider, model=model, status=status, limit=limit, newest_first=True
    )
    return RequestListResponse(start=start, end=end, count=len(records), data=records)
