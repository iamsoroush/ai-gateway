"""Server-Sent Events formatting for streaming chat completions.

Wraps a provider's text-delta stream into OpenAI-style ``chat.completion.chunk``
SSE events, terminated by ``data: [DONE]``. Errors raised mid-stream (after the
HTTP response has started) are emitted as a final error event so the client is
not left hanging.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from app.models.canonical import CanonicalLLMRequest
from app.models.errors import ErrorBody, ErrorResponse, GatewayError
from app.providers.base import BaseLLMProvider


class UsageCollector:
    """Captures the outcome of a stream for accounting.

    ``usage`` holds the token usage from the stream's terminal event (if any);
    ``error`` holds an exception raised mid-stream (after the HTTP response has
    already started), so the request can still be recorded as a failure.
    """

    def __init__(self) -> None:
        self.usage = None
        self.error: Exception | None = None
        self.finish_reason: str | None = None


def _chunk(completion_id: str, model: str, created: int, delta: dict, finish_reason=None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def sse_stream(
    provider: BaseLLMProvider,
    request: CanonicalLLMRequest,
    completion_id: str,
    created: int,
    collector: "UsageCollector | None" = None,
) -> AsyncIterator[str]:
    model = request.model_alias
    # First chunk announces the assistant role, matching OpenAI's behaviour.
    yield _chunk(completion_id, model, created, {"role": "assistant", "content": ""})
    try:
        async for event in provider.stream_complete(request):
            if event.delta:
                yield _chunk(completion_id, model, created, {"content": event.delta})
            if event.delta_payload:
                yield _chunk(completion_id, model, created, event.delta_payload)
            if event.finish_reason is not None and collector is not None:
                collector.finish_reason = event.finish_reason
            if event.usage is not None and collector is not None:
                collector.usage = event.usage
        finish_reason = collector.finish_reason if collector is not None else None
        yield _chunk(completion_id, model, created, {}, finish_reason=finish_reason or "stop")
    except GatewayError as exc:
        if collector is not None:
            collector.error = exc
        yield f"data: {json.dumps(exc.to_response().model_dump())}\n\n"
    except Exception as exc:  # noqa: BLE001 - never leak a mid-stream error to the client
        # The response is already 200; surface a content-free error event and record
        # the failure (so "record every request" holds even for unexpected errors).
        if collector is not None:
            collector.error = exc
        body = ErrorResponse(
            error=ErrorBody(
                message="The upstream provider stream failed.",
                type="provider_error",
                code="provider_request_failed",
            )
        )
        yield f"data: {json.dumps(body.model_dump())}\n\n"
    yield "data: [DONE]\n\n"
