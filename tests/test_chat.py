"""Route-level tests for /v1/chat/completions using a fake provider."""

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from tests.conftest import FakeProvider, FakeRouter


def test_non_streaming_completion(fake_client):
    resp = fake_client.post(
        "/v1/chat/completions",
        json={
            "model": "report-fast",
            "stream": False,
            "messages": [{"role": "user", "content": "Write a short test report."}],
        },
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "report-fast"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hello report"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 5


def test_streaming_completion(fake_client):
    with fake_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "report-fast",
            "stream": True,
            "messages": [{"role": "user", "content": "Write a short test report."}],
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    assert "chat.completion.chunk" in text
    assert '"content": "hello"' in text
    assert '"content": " report"' in text
    assert text.strip().endswith("data: [DONE]")


def test_unknown_model_returns_404(fake_client):
    resp = fake_client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "model_not_found"


def test_unsupported_content_returns_400(client):
    # report-large -> gpt-4o, which does not accept audio. Uses the real router;
    # the 400 is raised before any provider/network call.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "report-large",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "transcribe"},
                        {"type": "audio_url", "audio_url": {"url": "https://x/a.wav"}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unsupported_content_type"
    assert "audio_url" in body["error"]["message"]


def test_invalid_body_returns_422(client):
    # "model" is optional, but at least one message is required.
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_bare_model_name_passthrough(fake_client):
    # A raw model name (not an alias) is accepted and echoed back.
    resp = fake_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4o-mini"


def test_default_model_without_model_field(fake_client):
    # No "model" + no audio -> the configured general default (DEFAULT_MODEL).
    resp = fake_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == get_settings().default_model


def test_response_format_forwarded_to_provider():
    # The provider-agnostic route carries response_format through to the adapter
    # (where OpenAI passes it through and Gemini translates it to native JSON mode).
    captured = {}

    class CapturingProvider(FakeProvider):
        async def complete(self, request):
            captured["response_format"] = request.response_format
            return await super().complete(request)

    original = app.state.provider_router
    app.state.provider_router = FakeRouter(CapturingProvider())
    try:
        with TestClient(app) as test_client:
            rf = {"type": "json_schema", "json_schema": {"name": "S", "schema": {"type": "object"}}}
            resp = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "report-fast",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": rf,
                },
            )
    finally:
        app.state.provider_router = original

    assert resp.status_code == 200
    assert captured["response_format"] == rf


def test_reasoning_effort_forwarded_to_provider(fake_client):
    # A valid reasoning_effort is accepted and carried into the canonical request.
    captured = {}

    class CapturingProvider(FakeProvider):
        async def complete(self, request):
            captured["reasoning_effort"] = request.reasoning_effort
            return await super().complete(request)

    original = app.state.provider_router
    app.state.provider_router = FakeRouter(CapturingProvider())
    try:
        with TestClient(app) as test_client:
            resp = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "report-fast",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "high",
                },
            )
    finally:
        app.state.provider_router = original

    assert resp.status_code == 200
    assert captured["reasoning_effort"] == "high"


def test_openai_tools_forwarded_and_tool_call_response(fake_client):
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    resp = fake_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4-nano",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [tool],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Tehran"}'},
        }
    ]


def test_openai_tool_result_messages_are_accepted():
    captured = {}

    class CapturingProvider(FakeProvider):
        async def complete(self, request):
            captured["roles"] = [m.role for m in request.messages]
            captured["tool_call_id"] = request.messages[-1].tool_call_id
            captured["assistant_tool_calls"] = request.messages[1].tool_calls
            return await super().complete(request)

    original = app.state.provider_router
    app.state.provider_router = FakeRouter(CapturingProvider())
    try:
        with TestClient(app) as test_client:
            resp = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4-nano",
                    "messages": [
                        {"role": "user", "content": "weather"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Tehran"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_weather",
                            "content": '{"temperature_c": 21}',
                        },
                    ],
                },
            )
    finally:
        app.state.provider_router = original

    assert resp.status_code == 200
    assert captured["roles"] == ["user", "assistant", "tool"]
    assert captured["tool_call_id"] == "call_weather"
    assert captured["assistant_tool_calls"][0]["id"] == "call_weather"


def test_tools_are_rejected_for_gemini(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "report-fast",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_feature"


def test_invalid_reasoning_effort_returns_422(fake_client):
    resp = fake_client.post(
        "/v1/chat/completions",
        json={
            "model": "report-fast",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "ultra",  # not one of minimal/low/medium/high
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_default_model_with_audio(fake_client):
    # No "model" + audio -> the configured audio default (DEFAULT_AUDIO_MODEL).
    resp = fake_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize"},
                        {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                    ],
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == get_settings().default_audio_model
