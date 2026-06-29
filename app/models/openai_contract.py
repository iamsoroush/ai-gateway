"""OpenAI-compatible request/response schemas (the public contract).

These mirror the shape of the OpenAI Chat Completions API so existing OpenAI
clients can talk to the gateway unchanged. Unknown extra request fields are
ignored rather than rejected, to stay forward-compatible with OpenAI clients.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Request: multimodal content parts                                           #
# --------------------------------------------------------------------------- #


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


class AudioURL(BaseModel):
    url: str
    mime_type: str | None = None


class AudioURLPart(BaseModel):
    type: Literal["audio_url"]
    audio_url: AudioURL


class InputAudio(BaseModel):
    data: str  # base64
    format: str | None = None  # e.g. "wav", "mp3"


class InputAudioPart(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPart = Annotated[
    Union[TextPart, ImagePart, AudioURLPart, InputAudioPart],
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Either a plain string (the common case) or a list of typed content parts.
    content: Union[str, list[ContentPart], None] = None
    # OpenAI tool-calling loop fields. Kept as dicts so the gateway can pass
    # through the current OpenAI schema without re-modeling every nested variant.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    # Optional: if omitted, the gateway picks a default based on the content
    # (see services.normalizer). May be a registered alias or a raw model name.
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Reasoning/"thinking" effort (OpenAI-style). Forwarded natively to OpenAI and
    # translated to Gemini's thinking controls — see services.normalizer / providers.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    metadata: dict | None = None


# --------------------------------------------------------------------------- #
# Response                                                                    #
# --------------------------------------------------------------------------- #


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


# --------------------------------------------------------------------------- #
# Model listing                                                               #
# --------------------------------------------------------------------------- #


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    provider: str
    provider_model: str | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #


EmbeddingInput = Union[str, list[str], list[int], list[list[int]]]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model: str
    input: EmbeddingInput
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage | None = None
