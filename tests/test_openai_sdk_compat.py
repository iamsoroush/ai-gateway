"""Verify the gateway's responses parse into the official OpenAI SDK types.

This is what lets callers use the standard `openai` Python SDK against the
gateway and get clean, typed `ChatCompletion` / `ChatCompletionChunk` objects.
We validate the gateway's *actual* emitted payloads (via the fake provider, so
no network/keys) against the SDK's Pydantic models.
"""

import json

import pytest

# openai is a runtime dependency, but skip cleanly if it isn't installed.
pytest.importorskip("openai")

from openai.types.chat import ChatCompletion, ChatCompletionChunk  # noqa: E402


def test_non_streaming_response_matches_openai_type(fake_client):
    resp = fake_client.post(
        "/v1/chat/completions",
        json={"model": "report-fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200

    # Will raise pydantic.ValidationError if the shape is not OpenAI-compatible.
    parsed = ChatCompletion.model_validate(resp.json())
    assert parsed.object == "chat.completion"
    assert parsed.choices[0].message.content == "hello report"
    assert parsed.choices[0].message.role == "assistant"


def test_streaming_chunks_match_openai_type(fake_client):
    with fake_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "report-fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    chunks = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        # Each SSE chunk must validate as a ChatCompletionChunk.
        chunks.append(ChatCompletionChunk.model_validate(json.loads(payload)))

    assert chunks, "expected at least one streamed chunk"
    assert chunks[0].object == "chat.completion.chunk"
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert streamed == "hello report"
    assert chunks[-1].choices[0].finish_reason == "stop"
