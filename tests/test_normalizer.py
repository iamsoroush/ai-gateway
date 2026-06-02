import pytest

from app.config import get_settings, resolve_model
from app.models.errors import UnknownModelError
from app.models.openai_contract import ChatCompletionRequest
from app.services.normalizer import normalize_request


def _request(**overrides) -> ChatCompletionRequest:
    payload = {"model": "report-fast", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def test_normalize_simple_string_content():
    canonical = normalize_request(_request())

    assert canonical.model_alias == "report-fast"
    assert canonical.provider == "gemini"
    assert canonical.provider_model == "gemini-2.5-flash"
    assert len(canonical.messages) == 1

    parts = canonical.messages[0].content
    assert len(parts) == 1
    assert parts[0].type == "text"
    assert parts[0].text == "hi"


def test_normalize_multimodal_content_list():
    req = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these inputs."},
                    {"type": "image_url", "image_url": {"url": "https://x/img.jpg"}},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://x/a.wav", "mime_type": "audio/wav"},
                    },
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                ],
            }
        ]
    )

    canonical = normalize_request(req)
    parts = canonical.messages[0].content
    assert [p.type for p in parts] == ["text", "image_url", "audio_url", "input_audio"]

    assert parts[0].text == "Describe these inputs."
    assert parts[1].url == "https://x/img.jpg"
    assert parts[2].url == "https://x/a.wav"
    assert parts[2].mime_type == "audio/wav"
    assert parts[3].data == "QUJD"
    assert parts[3].format == "wav"


def test_normalize_passes_response_format_through():
    # response_format is carried verbatim into the canonical request, where the
    # provider adapter decides how to honor it.
    rf = {"type": "json_schema", "json_schema": {"name": "S", "schema": {"type": "object"}}}
    canonical = normalize_request(_request(response_format=rf))
    assert canonical.response_format == rf


def test_normalize_unknown_model_raises():
    # Not a registered alias and not a recognizable provider model name.
    with pytest.raises(UnknownModelError):
        normalize_request(_request(model="does-not-exist"))


def test_normalize_bare_model_name_infers_provider():
    # A raw model name (not an alias) routes by inferring the provider.
    openai = normalize_request(_request(model="gpt-4o-mini"))
    assert openai.provider == "openai"
    assert openai.provider_model == "gpt-4o-mini"
    assert openai.model_alias == "gpt-4o-mini"  # echoed back as requested

    gemini = normalize_request(_request(model="gemini-1.5-pro"))
    assert gemini.provider == "gemini"
    assert gemini.provider_model == "gemini-1.5-pro"


def test_default_model_no_audio_uses_general_default():
    # No "model" + no audio -> the configured general default (DEFAULT_MODEL).
    canonical = normalize_request(_request(model=None))
    expected = resolve_model(get_settings().default_model)
    assert canonical.model_alias == get_settings().default_model
    assert canonical.provider == expected["provider"]
    assert canonical.provider_model == expected["provider_model"]


def test_default_model_with_audio_uses_audio_default():
    # No "model" + audio present -> the configured audio default (DEFAULT_AUDIO_MODEL).
    req = _request(
        model=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize."},
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                ],
            }
        ],
    )
    canonical = normalize_request(req)
    expected = resolve_model(get_settings().default_audio_model)
    assert canonical.model_alias == get_settings().default_audio_model
    assert canonical.provider == expected["provider"]
    assert canonical.provider_model == expected["provider_model"]
