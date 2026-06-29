"""OpenAI adapter.

Maps the canonical request to the OpenAI Chat Completions format using the
official async SDK, and normalizes the response back to canonical. The SDK is
imported lazily so the module (and the tests) load without the package present
or an API key configured.
"""

from __future__ import annotations

import base64
from typing import Any, AsyncIterator

from app.models.canonical import (
    CanonicalLLMRequest,
    CanonicalLLMResponse,
    CanonicalMessage,
    CanonicalUsage,
    StreamEvent,
)
from app.models.errors import MissingAPIKeyError, ProviderRequestError
from app.models.openai_contract import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.providers.base import BaseLLMProvider
from app.utils.media import fetch_bytes


def _to_canonical_usage(usage) -> CanonicalUsage | None:
    """Map an OpenAI usage object to CanonicalUsage with a modality breakdown.

    OpenAI reports audio tokens under ``*_tokens_details.audio_tokens``; image
    tokens are folded into the prompt count, so everything else is treated as text.
    """
    if usage is None:
        return None
    prompt = usage.prompt_tokens or 0
    completion = usage.completion_tokens or 0
    audio_in = getattr(getattr(usage, "prompt_tokens_details", None), "audio_tokens", 0) or 0
    audio_out = getattr(getattr(usage, "completion_tokens_details", None), "audio_tokens", 0) or 0

    input_modality = {}
    if prompt - audio_in:
        input_modality["text"] = prompt - audio_in
    if audio_in:
        input_modality["audio"] = audio_in
    output_modality = {}
    if completion - audio_out:
        output_modality["text"] = completion - audio_out
    if audio_out:
        output_modality["audio"] = audio_out

    return CanonicalUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=usage.total_tokens,
        input_modality_tokens=input_modality or None,
        output_modality_tokens=output_modality or None,
    )


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any = None

    def supported_content_types(self, provider_model: str) -> set[str]:
        types = {"text", "image_url"}
        # Only the audio-capable models (e.g. gpt-4o-audio-preview) accept audio
        # via Chat Completions. Plain gpt-4o does not.
        if "audio" in provider_model:
            types |= {"audio_url", "input_audio"}
        return types

    def ensure_ready(self) -> None:
        self._get_client()

    def _get_client(self) -> Any:
        if not self._api_key:
            raise MissingAPIKeyError("OPENAI_API_KEY is not configured")
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def _to_openai_messages(self, messages: list[CanonicalMessage]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            # Collapse pure-text messages to the simple string form.
            if all(p.type == "text" for p in msg.content):
                text = "".join(p.text or "" for p in msg.content)
                out.append({"role": msg.role, "content": text})
                continue

            parts: list[dict] = []
            for p in msg.content:
                if p.type == "text":
                    parts.append({"type": "text", "text": p.text or ""})
                elif p.type == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": p.url}})
                elif p.type == "input_audio":
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": p.data, "format": p.format or "wav"},
                        }
                    )
                elif p.type == "audio_url":
                    # OpenAI only accepts inline base64 audio, so fetch + encode.
                    data, mime = await self._download(p.url)
                    fmt = (p.mime_type or mime or "audio/wav").rsplit("/", 1)[-1]
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(data).decode(),
                                "format": fmt,
                            },
                        }
                    )
            out.append({"role": msg.role, "content": parts})
        return out

    @staticmethod
    async def _download(url: str | None):
        try:
            return await fetch_bytes(url or "")
        except Exception as exc:  # network / HTTP errors
            raise ProviderRequestError(f"Failed to fetch media: {exc}") from exc

    def _build_kwargs(self, request: CanonicalLLMRequest, messages: list[dict]) -> dict:
        kwargs: dict[str, Any] = {"model": request.provider_model, "messages": messages}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        # OpenAI accepts reasoning_effort natively on reasoning-capable models.
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        return kwargs

    def _build_embedding_kwargs(self, request: EmbeddingRequest) -> dict:
        kwargs: dict[str, Any] = {"model": request.model, "input": request.input}
        if request.encoding_format is not None:
            kwargs["encoding_format"] = request.encoding_format
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        if request.user is not None:
            kwargs["user"] = request.user
        return kwargs

    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
        client = self._get_client()
        messages = await self._to_openai_messages(request.messages)
        try:
            resp = await client.chat.completions.create(**self._build_kwargs(request, messages))
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

        choice = resp.choices[0]
        return CanonicalLLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            provider_model=request.provider_model,
            usage=_to_canonical_usage(getattr(resp, "usage", None)),
        )

    async def stream_complete(self, request: CanonicalLLMRequest) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        messages = await self._to_openai_messages(request.messages)
        kwargs = self._build_kwargs(request, messages)
        kwargs["stream"] = True
        # Ask OpenAI to emit a final usage-only chunk so streaming is accountable.
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI stream failed: {exc}") from exc

        try:
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield StreamEvent(delta=delta.content)
                if getattr(chunk, "usage", None):
                    yield StreamEvent(usage=_to_canonical_usage(chunk.usage))
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI stream interrupted: {exc}") from exc

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        client = self._get_client()
        try:
            resp = await client.embeddings.create(**self._build_embedding_kwargs(request))
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI embeddings request failed: {exc}") from exc

        usage = getattr(resp, "usage", None)
        return EmbeddingResponse(
            data=[
                EmbeddingData(
                    object=getattr(item, "object", "embedding"),
                    embedding=item.embedding,
                    index=item.index,
                )
                for item in resp.data
            ],
            model=getattr(resp, "model", request.model),
            usage=(
                EmbeddingUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )
                if usage is not None
                else None
            ),
        )
