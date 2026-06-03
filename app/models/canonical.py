"""Internal canonical representation of an LLM request/response.

This is the provider-agnostic format that the rest of the system speaks. The
API layer normalizes inbound OpenAI-shaped requests into these models, provider
adapters translate them to/from their wire formats, and responses are mapped
back out to the OpenAI contract. Nothing here is provider specific.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CanonicalContentPart(BaseModel):
    """A single piece of message content (text or a media reference)."""

    type: Literal["text", "image_url", "audio_url", "input_audio"]
    text: str | None = None
    url: str | None = None
    data: str | None = None  # base64-encoded inline data
    mime_type: str | None = None
    format: str | None = None  # e.g. "wav", "mp3" for input_audio


class CanonicalMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[CanonicalContentPart]


class CanonicalLLMRequest(BaseModel):
    # ``model_`` is a Pydantic-protected prefix; opt out so field names are free.
    model_config = ConfigDict(protected_namespaces=())

    model_alias: str
    provider: str
    provider_model: str
    messages: list[CanonicalMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    response_format: dict | None = None
    # Reasoning effort: "minimal" | "low" | "medium" | "high". Providers translate
    # this to their own thinking controls (OpenAI reasoning_effort, Gemini thinking).
    reasoning_effort: str | None = None
    metadata: dict | None = None


class CanonicalUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Best-effort per-modality token breakdown (e.g. {"text": 10, "audio": 4}).
    # Providers populate what they report; anything unbroken counts as text.
    input_modality_tokens: dict[str, int] | None = None
    output_modality_tokens: dict[str, int] | None = None


class CanonicalLLMResponse(BaseModel):
    content: str
    finish_reason: str | None = "stop"
    provider_model: str
    usage: CanonicalUsage | None = None


class StreamEvent(BaseModel):
    """One item from a provider's streaming response.

    ``delta`` carries an incremental text chunk; ``usage`` is normally ``None``
    and set only on the terminal event that reports token usage (so streaming
    requests can be accounted for too).
    """

    delta: str = ""
    usage: CanonicalUsage | None = None
