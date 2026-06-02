import pytest

from app.config import Settings
from app.models.errors import (
    MissingAPIKeyError,
    UnsupportedContentError,
    UnsupportedProviderError,
)
from app.models.canonical import CanonicalContentPart, CanonicalLLMRequest, CanonicalMessage
from app.providers.openai_provider import OpenAIProvider
from app.providers.gemini_provider import GeminiProvider
from app.services.normalizer import validate_content_support
from app.services.router import ProviderRouter
from app.utils.media import _parse_data_uri


def test_router_unsupported_provider_raises():
    router = ProviderRouter(Settings(openai_api_key=None, gemini_api_key=None))
    with pytest.raises(UnsupportedProviderError):
        router.get("anthropic")  # no adapter registered


def test_router_returns_known_providers():
    router = ProviderRouter(Settings())
    assert router.get("openai").name == "openai"
    assert router.get("gemini").name == "gemini"


def test_missing_api_key_raises():
    provider = OpenAIProvider(api_key=None)
    with pytest.raises(MissingAPIKeyError):
        provider.ensure_ready()


def test_provider_capabilities():
    # gpt-4o does not accept audio; the audio-preview variant does.
    openai = OpenAIProvider()
    assert "audio_url" not in openai.supported_content_types("gpt-4o")
    assert "input_audio" in openai.supported_content_types("gpt-4o-audio-preview")

    # Gemini multimodal handles everything.
    gemini = GeminiProvider()
    assert gemini.supported_content_types("gemini-2.5-flash") == {
        "text",
        "image_url",
        "audio_url",
        "input_audio",
    }


def test_validate_content_support_rejects_unsupported():
    request = CanonicalLLMRequest(
        model_alias="report-large",
        provider="openai",
        provider_model="gpt-4o",
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalContentPart(type="audio_url", url="https://x/a.wav")],
            )
        ],
    )
    supported = OpenAIProvider().supported_content_types("gpt-4o")
    with pytest.raises(UnsupportedContentError):
        validate_content_support(request, supported)


def test_parse_base64_data_uri():
    # "QUJD" is base64 for "ABC".
    content, mime = _parse_data_uri("data:image/png;base64,QUJD")
    assert content == b"ABC"
    assert mime == "image/png"


def _text_request(provider: str, provider_model: str, **overrides) -> CanonicalLLMRequest:
    return CanonicalLLMRequest(
        model_alias="alias",
        provider=provider,
        provider_model=provider_model,
        messages=[
            CanonicalMessage(
                role="user", content=[CanonicalContentPart(type="text", text="hi")]
            )
        ],
        **overrides,
    )


def test_structured_output_translation():
    from app.providers.gemini_provider import _structured_output

    # No constraint for absent / plain-text response formats.
    assert _structured_output(None) == (None, None)
    assert _structured_output({"type": "text"}) == (None, None)
    # JSON mode without a schema.
    assert _structured_output({"type": "json_object"}) == ("application/json", None)
    # JSON schema -> JSON mode constrained to the embedded schema (the shape the
    # OpenAI SDK sends for a Pydantic response_format).
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    assert _structured_output(
        {"type": "json_schema", "json_schema": {"name": "S", "schema": schema, "strict": True}}
    ) == ("application/json", schema)
    # json_schema with no embedded schema -> JSON mode only.
    assert _structured_output({"type": "json_schema", "json_schema": {}}) == (
        "application/json",
        None,
    )


def test_gemini_build_applies_response_schema():
    pytest.importorskip("google.genai")
    import asyncio

    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    request = _text_request(
        "gemini",
        "gemini-2.5-flash",
        response_format={"type": "json_schema", "json_schema": {"name": "S", "schema": schema}},
    )
    _, config = asyncio.run(GeminiProvider()._build(request))
    assert config.response_mime_type == "application/json"
    applied = getattr(config, "response_json_schema", None) or getattr(config, "response_schema", None)
    assert applied == schema

    # No response_format -> no JSON constraint imposed.
    _, plain = asyncio.run(GeminiProvider()._build(_text_request("gemini", "gemini-2.5-flash")))
    assert plain.response_mime_type is None


def test_openai_build_kwargs_forwards_response_format():
    # OpenAI understands response_format natively, so the adapter passes it through.
    rf = {"type": "json_schema", "json_schema": {"name": "S", "schema": {"type": "object"}}}
    request = _text_request("openai", "gpt-4o", response_format=rf)
    kwargs = OpenAIProvider()._build_kwargs(request, messages=[])
    assert kwargs["response_format"] == rf
