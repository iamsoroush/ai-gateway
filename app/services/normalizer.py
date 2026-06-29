"""Normalize the public OpenAI-shaped request into the canonical format.

This is the single place that understands the inbound contract. It resolves the
requested model (a registered alias or a raw provider model name) to a provider,
picks a content-aware default when no model is given, flattens string-or-list
message content into typed canonical parts, and offers a capability check
against a provider's supported content types.
"""

from __future__ import annotations

from app.config import get_settings, resolve_model
from app.models.canonical import (
    CanonicalContentPart,
    CanonicalLLMRequest,
    CanonicalMessage,
)
from app.models.errors import UnknownModelError, UnsupportedContentError, UnsupportedFeatureError
from app.models.openai_contract import ChatCompletionRequest, ChatMessage

# Canonical content types that count as "audio" for default-model selection.
_AUDIO_TYPES = {"audio_url", "input_audio"}


def _request_has_audio(req: ChatCompletionRequest) -> bool:
    for message in req.messages:
        if isinstance(message.content, list):
            if any(part.type in _AUDIO_TYPES for part in message.content):
                return True
    return False


def _select_model(req: ChatCompletionRequest) -> str:
    """Return the requested model, or a content-aware default if none was given.

    Audio-bearing requests default to the audio-capable model; everything else
    defaults to the general model. Both defaults are configurable via Settings.
    """
    if req.model:
        return req.model
    settings = get_settings()
    return settings.default_audio_model if _request_has_audio(req) else settings.default_model


def _uses_tooling(req: ChatCompletionRequest) -> bool:
    if req.tools:
        return True
    if req.tool_choice not in (None, "none"):
        return True
    for message in req.messages:
        if message.role == "tool" or message.tool_calls or message.tool_call_id or message.function_call:
            return True
    return False


def _uses_openai_only_request_metadata(req: ChatCompletionRequest) -> bool:
    return bool(req.prompt_cache_key or req.prompt_cache_retention or req.user)


def _normalize_content(message: ChatMessage) -> list[CanonicalContentPart]:
    if message.content is None:
        return []
    if isinstance(message.content, str):
        return [CanonicalContentPart(type="text", text=message.content)]

    parts: list[CanonicalContentPart] = []
    for p in message.content:
        if p.type == "text":
            parts.append(CanonicalContentPart(type="text", text=p.text))
        elif p.type == "image_url":
            parts.append(CanonicalContentPart(type="image_url", url=p.image_url.url))
        elif p.type == "audio_url":
            parts.append(
                CanonicalContentPart(
                    type="audio_url", url=p.audio_url.url, mime_type=p.audio_url.mime_type
                )
            )
        elif p.type == "input_audio":
            parts.append(
                CanonicalContentPart(
                    type="input_audio", data=p.input_audio.data, format=p.input_audio.format
                )
            )
    return parts


def normalize_request(req: ChatCompletionRequest) -> CanonicalLLMRequest:
    """Resolve the model and convert to a :class:`CanonicalLLMRequest`.

    The model may be a registered alias or a raw provider model name; if omitted
    a content-aware default is chosen. Raises :class:`UnknownModelError` if the
    model is neither a known alias nor a recognizable provider model name.
    """
    model = _select_model(req)
    model_cfg = resolve_model(model)
    if model_cfg is None:
        raise UnknownModelError(
            f"Unknown model: '{model}'. Use a registered alias, or a model name "
            f"whose provider can be inferred (e.g. 'gpt-...', 'gemini-...')."
        )

    if model_cfg["provider"] != "openai" and _uses_tooling(req):
        raise UnsupportedFeatureError(
            "Tool calling is currently supported only for OpenAI models. "
            "Use an OpenAI model such as 'gpt-5.4-nano', or omit tools/tool messages."
        )
    if model_cfg["provider"] != "openai" and _uses_openai_only_request_metadata(req):
        raise UnsupportedFeatureError(
            "prompt_cache_key, prompt_cache_retention, and user are currently "
            "supported only for OpenAI chat models."
        )

    messages = [
        CanonicalMessage(
            role=m.role,
            content=_normalize_content(m),
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
            function_call=m.function_call,
            name=m.name,
        )
        for m in req.messages
    ]
    return CanonicalLLMRequest(
        model_alias=model,
        provider=model_cfg["provider"],
        provider_model=model_cfg["provider_model"],
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
        response_format=req.response_format,
        tools=req.tools,
        tool_choice=req.tool_choice,
        parallel_tool_calls=req.parallel_tool_calls,
        prompt_cache_key=req.prompt_cache_key,
        prompt_cache_retention=req.prompt_cache_retention,
        user=req.user,
        reasoning_effort=req.reasoning_effort,
        metadata=req.metadata,
    )


def validate_content_support(request: CanonicalLLMRequest, supported: set[str]) -> None:
    """Reject requests using content types the provider/model cannot handle.

    We never silently drop unsupported content — this raises a 400 listing the
    offending types.
    """
    used = {part.type for message in request.messages for part in message.content}
    unsupported = used - supported
    if unsupported:
        raise UnsupportedContentError(
            f"Model '{request.model_alias}' (provider '{request.provider}', "
            f"provider_model '{request.provider_model}') does not support content "
            f"type(s): {', '.join(sorted(unsupported))}."
        )
