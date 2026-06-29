"""Shared test fixtures.

Provides a FastAPI TestClient and a fake provider so route tests never touch a
real OpenAI/Gemini API.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.canonical import (
    CanonicalLLMRequest,
    CanonicalLLMResponse,
    CanonicalUsage,
    StreamEvent,
)
from app.models.errors import ProviderRequestError
from app.models.openai_contract import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.providers.base import BaseLLMProvider
from app.services.request_store import InMemoryRequestStore


class FakeProvider(BaseLLMProvider):
    """In-memory provider that echoes deterministic content."""

    name = "fake"

    def supported_content_types(self, provider_model: str) -> set[str]:
        return {"text", "image_url", "audio_url", "input_audio"}

    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
        if request.tools:
            return CanonicalLLMResponse(
                content=None,
                finish_reason="tool_calls",
                provider_model=request.provider_model,
                usage=CanonicalUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
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
            )
        return CanonicalLLMResponse(
            content="hello report",
            finish_reason="stop",
            provider_model=request.provider_model,
            usage=CanonicalUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    async def stream_complete(self, request: CanonicalLLMRequest) -> AsyncIterator[StreamEvent]:
        for token in ["hello", " report"]:
            yield StreamEvent(delta=token)
        # Terminal usage event so streaming requests are accounted for.
        yield StreamEvent(
            usage=CanonicalUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        )

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)],
            model=request.model,
            usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
        )


class FailingProvider(BaseLLMProvider):
    """Provider whose upstream call fails — for failure-recording tests."""

    name = "failing"

    def supported_content_types(self, provider_model: str) -> set[str]:
        return {"text", "image_url", "audio_url", "input_audio"}

    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
        raise ProviderRequestError("upstream boom")

    async def stream_complete(self, request: CanonicalLLMRequest) -> AsyncIterator[StreamEvent]:
        raise ProviderRequestError("upstream boom")
        yield StreamEvent(delta="")  # pragma: no cover - unreachable, makes this a generator


class FakeRouter:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    def get(self, provider_name: str) -> BaseLLMProvider:
        return self._provider


@pytest.fixture
def client() -> TestClient:
    """TestClient backed by the real ProviderRouter (no network calls made)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_client() -> TestClient:
    """TestClient whose provider router always returns the FakeProvider."""
    original = app.state.provider_router
    app.state.provider_router = FakeRouter(FakeProvider())
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.provider_router = original


def _store_client(provider: BaseLLMProvider):
    """Context-managed TestClient with a fresh in-memory store and the given provider."""
    orig_router = app.state.provider_router
    orig_store = app.state.request_store
    store = InMemoryRequestStore()
    app.state.provider_router = FakeRouter(provider)
    app.state.request_store = store
    try:
        with TestClient(app) as test_client:
            yield test_client, store
    finally:
        app.state.provider_router = orig_router
        app.state.request_store = orig_store


@pytest.fixture
def usage_client():
    """TestClient with a fresh in-memory request store and the FakeProvider.

    Yields ``(client, store)`` so tests can pre-populate records or assert what
    the chat route recorded, in isolation from other tests.
    """
    yield from _store_client(FakeProvider())


@pytest.fixture
def failing_client():
    """Like ``usage_client`` but the provider's upstream call fails.

    Yields ``(client, store)`` for exercising failure recording.
    """
    yield from _store_client(FailingProvider())
