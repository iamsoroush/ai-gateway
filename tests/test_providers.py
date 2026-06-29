from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models.errors import (
    MissingAPIKeyError,
    UnsupportedContentError,
    UnsupportedProviderError,
)
from app.models.canonical import CanonicalContentPart, CanonicalLLMRequest, CanonicalMessage
from app.providers.openai_provider import OpenAIProvider
from app.providers.openai_provider import _to_canonical_usage as openai_usage
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


def test_openai_build_kwargs_forwards_reasoning_effort():
    # OpenAI accepts reasoning_effort natively; forward it when set, omit otherwise.
    req = _text_request("openai", "gpt-5.4-nano", reasoning_effort="high")
    assert OpenAIProvider()._build_kwargs(req, messages=[])["reasoning_effort"] == "high"
    plain = _text_request("openai", "gpt-5.4-nano")
    assert "reasoning_effort" not in OpenAIProvider()._build_kwargs(plain, messages=[])


def test_openai_build_kwargs_prefers_max_completion_tokens():
    req = _text_request(
        "openai",
        "gpt-5.4-nano",
        max_tokens=100,
        max_completion_tokens=20,
    )
    kwargs = OpenAIProvider()._build_kwargs(req, messages=[])
    assert kwargs["max_completion_tokens"] == 20
    assert "max_tokens" not in kwargs

    legacy = _text_request("openai", "gpt-5.4-nano", max_tokens=100)
    assert OpenAIProvider()._build_kwargs(legacy, messages=[])["max_tokens"] == 100


def test_openai_build_kwargs_forwards_tools():
    tool = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
    req = _text_request(
        "openai",
        "gpt-5.4-nano",
        tools=[tool],
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    kwargs = OpenAIProvider()._build_kwargs(req, messages=[])
    assert kwargs["tools"] == [tool]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False


def test_openai_build_kwargs_forwards_prompt_cache_and_user():
    req = _text_request(
        "openai",
        "gpt-5.4-nano",
        prompt_cache_key="tenant-a-report-template",
        prompt_cache_retention="24h",
        user="user_123",
    )

    kwargs = OpenAIProvider()._build_kwargs(req, messages=[])
    assert kwargs["prompt_cache_key"] == "tenant-a-report-template"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["user"] == "user_123"


def test_openai_embedding_kwargs_forward_user():
    from app.models.openai_contract import EmbeddingRequest

    req = EmbeddingRequest(
        model="text-embedding-3-small",
        input="hello",
        user="user_123",
    )
    kwargs = OpenAIProvider()._build_embedding_kwargs(req)
    assert kwargs["user"] == "user_123"


def test_openai_usage_maps_cached_prompt_tokens():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=SimpleNamespace(audio_tokens=10, cached_tokens=40),
        completion_tokens_details=SimpleNamespace(audio_tokens=0),
    )

    canonical = openai_usage(usage)

    assert canonical.cached_input_tokens == 40
    assert canonical.input_modality_tokens == {"text": 90, "audio": 10}
    assert canonical.raw_usage["prompt_tokens_details"]["cached_tokens"] == 40


def test_openai_messages_preserve_tool_call_loop():
    import asyncio

    request = CanonicalLLMRequest(
        model_alias="gpt-5.4-nano",
        provider="openai",
        provider_model="gpt-5.4-nano",
        messages=[
            CanonicalMessage(
                role="assistant",
                content=[],
                tool_calls=[
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tehran"}',
                        },
                    }
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[CanonicalContentPart(type="text", text='{"temperature_c":21}')],
                tool_call_id="call_weather",
            ),
        ],
    )

    messages = asyncio.run(OpenAIProvider()._to_openai_messages(request.messages))
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] is None
    assert messages[0]["tool_calls"][0]["id"] == "call_weather"
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_weather",
        "content": '{"temperature_c":21}',
    }


def test_gemini_thinking_spec_maps_effort_by_model_family():
    from app.providers.gemini_provider import _thinking_spec

    # No effort -> no thinking control.
    assert _thinking_spec("gemini-2.5-flash", None) is None
    # Gemini 2.5 -> integer thinking_budget (minimal=off, high=dynamic).
    assert _thinking_spec("gemini-2.5-flash", "minimal") == ("thinking_budget", 0)
    assert _thinking_spec("gemini-2.5-flash", "medium") == ("thinking_budget", 8192)
    assert _thinking_spec("gemini-2.5-flash", "high") == ("thinking_budget", -1)
    # Gemini 3+ -> thinking_level (1:1 with the effort names).
    assert _thinking_spec("gemini-3-pro", "low") == ("thinking_level", "LOW")
    assert _thinking_spec("models/gemini-3.5-flash", "high") == ("thinking_level", "HIGH")


def test_gemini_build_applies_thinking_config():
    pytest.importorskip("google.genai")
    import asyncio

    # 2.5 model -> thinking_budget on the config.
    _, cfg25 = asyncio.run(
        GeminiProvider()._build(_text_request("gemini", "gemini-2.5-flash", reasoning_effort="medium"))
    )
    assert cfg25.thinking_config is not None
    assert cfg25.thinking_config.thinking_budget == 8192

    # 3.x model -> thinking_level enum on the config.
    from google.genai.types import ThinkingLevel

    _, cfg3 = asyncio.run(
        GeminiProvider()._build(_text_request("gemini", "gemini-3-pro", reasoning_effort="high"))
    )
    assert cfg3.thinking_config.thinking_level == ThinkingLevel.HIGH

    # No reasoning_effort -> no thinking_config imposed.
    _, plain = asyncio.run(GeminiProvider()._build(_text_request("gemini", "gemini-2.5-flash")))
    assert plain.thinking_config is None
