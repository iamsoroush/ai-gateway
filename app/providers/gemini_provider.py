"""Google Gemini adapter.

Maps the canonical request to the Google GenAI SDK's content format and
normalizes the response back. Gemini multimodal models accept text, images and
audio natively. The SDK needs media as inline bytes, so URL-referenced media is
downloaded first. SDK imports are lazy so the module loads without the package
or an API key.
"""

from __future__ import annotations

import base64
from typing import Any, AsyncIterator

from app.models.canonical import (
    CanonicalLLMRequest,
    CanonicalLLMResponse,
    CanonicalUsage,
    StreamEvent,
)
from app.models.errors import MissingAPIKeyError, ProviderRequestError
from app.providers.base import BaseLLMProvider
from app.utils.media import fetch_bytes

# Map Gemini finish reasons onto OpenAI-style values where they differ.
_FINISH_MAP = {
    "stop": "stop",
    "max_tokens": "length",
    "safety": "content_filter",
    "recitation": "content_filter",
}


def _modality_breakdown(details) -> dict[str, int]:
    """Turn Gemini's [ModalityTokenCount(modality, token_count), ...] into a dict."""
    out: dict[str, int] = {}
    for detail in details or []:
        name = str(getattr(detail, "modality", "") or "").rsplit(".", 1)[-1].lower() or "text"
        out[name] = out.get(name, 0) + (getattr(detail, "token_count", 0) or 0)
    return out


def _structured_output(response_format: dict | None) -> tuple[str | None, dict | None]:
    """Translate an OpenAI ``response_format`` into Gemini structured-output settings.

    Returns ``(response_mime_type, json_schema)`` so callers using the OpenAI SDK's
    structured-output feature get the same schema-constrained JSON from Gemini:
      * ``{"type": "json_object"}`` -> JSON mode, no schema.
      * ``{"type": "json_schema", "json_schema": {"schema": {...}}}`` -> JSON mode
        constrained to that JSON Schema (the shape the OpenAI SDK sends for a
        Pydantic ``response_format``).
    ``{"type": "text"}``, ``None``, or anything unrecognized adds no constraint.
    """
    if not response_format:
        return None, None
    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return "application/json", None
    if fmt_type == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        return "application/json", schema or None
    return None, None


# OpenAI reasoning_effort -> Gemini thinking controls. Gemini 3+ models take a
# `thinking_level` (its enum members line up 1:1 with the effort levels); Gemini
# 2.5 models take an integer `thinking_budget` (0 = off where allowed, -1 = let the
# model decide). Budgets are model-dependent and clamped by the API to its range.
_EFFORT_TO_LEVEL = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_EFFORT_TO_BUDGET = {"minimal": 0, "low": 1024, "medium": 8192, "high": -1}


def _thinking_spec(provider_model: str, effort: str | None) -> tuple[str, object] | None:
    """Resolve a ``(ThinkingConfig field, value)`` for an effort, or ``None``.

    Returns the *field name* and value to set on Gemini's ``ThinkingConfig`` so the
    caller (which holds the lazily-imported SDK ``types``) can build it. Gemini 3+
    uses ``thinking_level``; older (2.x) models use ``thinking_budget``.
    """
    if not effort:
        return None
    name = provider_model.lower().removeprefix("models/")
    if name.startswith("gemini-3"):
        level = _EFFORT_TO_LEVEL.get(effort)
        return ("thinking_level", level) if level is not None else None
    budget = _EFFORT_TO_BUDGET.get(effort)
    return ("thinking_budget", budget) if budget is not None else None


def _to_canonical_usage(meta) -> CanonicalUsage | None:
    """Map Gemini usage_metadata to CanonicalUsage with a modality breakdown."""
    if meta is None:
        return None
    prompt = getattr(meta, "prompt_token_count", None)
    candidates = getattr(meta, "candidates_token_count", None)
    input_modality = _modality_breakdown(getattr(meta, "prompt_tokens_details", None))
    output_modality = _modality_breakdown(getattr(meta, "candidates_tokens_details", None))
    if not input_modality and prompt:
        input_modality = {"text": prompt}
    if not output_modality and candidates:
        output_modality = {"text": candidates}
    return CanonicalUsage(
        prompt_tokens=prompt,
        completion_tokens=candidates,
        total_tokens=getattr(meta, "total_token_count", None),
        input_modality_tokens=input_modality or None,
        output_modality_tokens=output_modality or None,
    )


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any = None

    def supported_content_types(self, provider_model: str) -> set[str]:
        # Gemini 1.5/2.x multimodal models handle all of these natively.
        return {"text", "image_url", "audio_url", "input_audio"}

    def ensure_ready(self) -> None:
        self._get_client()

    def _get_client(self) -> Any:
        if not self._api_key:
            raise MissingAPIKeyError("GEMINI_API_KEY is not configured")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def _build(self, request: CanonicalLLMRequest):
        """Return ``(contents, config)`` for the GenAI SDK."""
        from google.genai import types

        system_instructions: list[str] = []
        contents = []
        for msg in request.messages:
            if msg.role == "system":
                system_instructions.extend(p.text or "" for p in msg.content if p.type == "text")
                continue

            role = "model" if msg.role == "assistant" else "user"
            parts = []
            for p in msg.content:
                if p.type == "text":
                    parts.append(types.Part(text=p.text or ""))
                elif p.type == "image_url":
                    data, mime = await self._download(p.url)
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"))
                elif p.type == "audio_url":
                    data, mime = await self._download(p.url)
                    parts.append(
                        types.Part.from_bytes(data=data, mime_type=p.mime_type or mime or "audio/wav")
                    )
                elif p.type == "input_audio":
                    raw = base64.b64decode(p.data or "")
                    mime = p.mime_type or f"audio/{p.format or 'wav'}"
                    parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
            contents.append(types.Content(role=role, parts=parts))

        mime_type, json_schema = _structured_output(request.response_format)
        config_kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "system_instruction": "\n".join(system_instructions) or None,
        }
        if mime_type:
            config_kwargs["response_mime_type"] = mime_type
        if json_schema is not None:
            # Prefer response_json_schema: it accepts a full JSON Schema (incl. the
            # $ref/$defs/additionalProperties that Pydantic and the OpenAI SDK emit).
            # Fall back to response_schema on older SDKs that predate it.
            field = (
                "response_json_schema"
                if "response_json_schema" in types.GenerateContentConfig.model_fields
                else "response_schema"
            )
            config_kwargs[field] = json_schema

        spec = _thinking_spec(request.provider_model, request.reasoning_effort)
        if spec is not None and "thinking_config" in types.GenerateContentConfig.model_fields:
            tf_field, tf_value = spec
            # thinking_level is an enum; resolve the member (fall back to a budget
            # on SDKs too old to know thinking_level).
            if tf_field == "thinking_level":
                if hasattr(types, "ThinkingLevel"):
                    tf_value = getattr(types.ThinkingLevel, tf_value)
                else:
                    tf_field = "thinking_budget"
                    tf_value = _EFFORT_TO_BUDGET.get(request.reasoning_effort, -1)
            config_kwargs["thinking_config"] = types.ThinkingConfig(**{tf_field: tf_value})

        config = types.GenerateContentConfig(**config_kwargs)
        return contents, config

    @staticmethod
    async def _download(url: str | None):
        try:
            return await fetch_bytes(url or "")
        except Exception as exc:
            raise ProviderRequestError(f"Failed to fetch media: {exc}") from exc

    @staticmethod
    def _finish_reason(resp: Any) -> str:
        try:
            raw = resp.candidates[0].finish_reason
            if raw is None:
                return "stop"
            name = str(raw).rsplit(".", 1)[-1].lower()
            return _FINISH_MAP.get(name, name)
        except Exception:
            return "stop"

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # resp.text can raise when a response has no valid text parts
        # (e.g. safety-blocked); treat that as empty content.
        try:
            return resp.text or ""
        except Exception:
            return ""

    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
        client = self._get_client()
        contents, config = await self._build(request)
        try:
            resp = await client.aio.models.generate_content(
                model=request.provider_model, contents=contents, config=config
            )
        except Exception as exc:
            raise ProviderRequestError(f"Gemini request failed: {exc}") from exc

        return CanonicalLLMResponse(
            content=self._extract_text(resp),
            finish_reason=self._finish_reason(resp),
            provider_model=request.provider_model,
            usage=_to_canonical_usage(getattr(resp, "usage_metadata", None)),
        )

    async def stream_complete(self, request: CanonicalLLMRequest) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        contents, config = await self._build(request)
        try:
            stream = await client.aio.models.generate_content_stream(
                model=request.provider_model, contents=contents, config=config
            )
        except Exception as exc:
            raise ProviderRequestError(f"Gemini stream failed: {exc}") from exc

        last_usage = None
        try:
            async for chunk in stream:
                if chunk.text:
                    yield StreamEvent(delta=chunk.text)
                # Gemini reports cumulative usage on chunks; keep the latest.
                meta = getattr(chunk, "usage_metadata", None)
                if meta is not None:
                    last_usage = meta
        except Exception as exc:
            raise ProviderRequestError(f"Gemini stream interrupted: {exc}") from exc

        if last_usage is not None:
            yield StreamEvent(usage=_to_canonical_usage(last_usage))
