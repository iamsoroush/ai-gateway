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
from app.providers.base import BaseLLMProvider
from app.services.usage_store import InMemoryUsageStore


class FakeProvider(BaseLLMProvider):
    """In-memory provider that echoes deterministic content."""

    name = "fake"

    def supported_content_types(self, provider_model: str) -> set[str]:
        return {"text", "image_url", "audio_url", "input_audio"}

    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
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


@pytest.fixture
def usage_client():
    """TestClient with a fresh in-memory usage store and the FakeProvider.

    Yields ``(client, store)`` so tests can pre-populate records or assert what
    the chat route recorded, in isolation from other tests.
    """
    orig_router = app.state.provider_router
    orig_store = app.state.usage_store
    store = InMemoryUsageStore()
    app.state.provider_router = FakeRouter(FakeProvider())
    app.state.usage_store = store
    try:
        with TestClient(app) as test_client:
            yield test_client, store
    finally:
        app.state.provider_router = orig_router
        app.state.usage_store = orig_store
