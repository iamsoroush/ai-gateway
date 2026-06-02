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
from app.models.errors import GatewayError
from app.providers.base import BaseLLMProvider


class UsageCollector:
    """Captures the usage reported on a stream's terminal event (if any)."""

    def __init__(self) -> None:
        self.usage = None


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
            if event.usage is not None and collector is not None:
                collector.usage = event.usage
        yield _chunk(completion_id, model, created, {}, finish_reason="stop")
    except GatewayError as exc:
        yield f"data: {json.dumps(exc.to_response().model_dump())}\n\n"
    yield "data: [DONE]\n\n"
