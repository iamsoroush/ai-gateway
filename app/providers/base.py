"""Provider interface that every adapter implements.

The API route and services depend only on this interface — never on a concrete
provider. Adding a new provider means writing a new subclass and registering it
in the :class:`~app.services.router.ProviderRouter`; nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.models.canonical import CanonicalLLMRequest, CanonicalLLMResponse, StreamEvent


class BaseLLMProvider(ABC):
    """Translate between the canonical format and a specific provider's API."""

    name: str = "base"

    def supported_content_types(self, provider_model: str) -> set[str]:
        """Canonical content types this provider/model can accept.

        Defaults to text only. Subclasses widen this (and may key it off the
        concrete model, e.g. only audio-capable models accept audio).
        """
        return {"text"}

    def ensure_ready(self) -> None:
        """Validate the adapter can serve a request (e.g. API key present).

        Called before a streaming response begins so configuration errors are
        surfaced as a normal HTTP error rather than mid-stream. Default no-op.
        """

    @abstractmethod
    async def complete(self, request: CanonicalLLMRequest) -> CanonicalLLMResponse:
        """Perform a non-streaming completion."""

    @abstractmethod
    def stream_complete(self, request: CanonicalLLMRequest) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamEvent`\\ s for a streaming completion.

        Most events carry a text ``delta``; the terminal event may carry ``usage``
        so streaming requests can be accounted for. Implemented as an ``async def``
        generator in subclasses.
        """
