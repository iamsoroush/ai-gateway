"""Provider router: maps a provider name to its adapter instance.

Holds one adapter per provider, constructed with the relevant API key. This is
the only component that knows the full set of concrete providers; the API route
asks it for an adapter by name and depends only on the base interface.
"""

from __future__ import annotations

from app.config import Settings
from app.models.errors import UnsupportedProviderError
from app.providers.base import BaseLLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider


class ProviderRouter:
    def __init__(self, settings: Settings) -> None:
        self._providers: dict[str, BaseLLMProvider] = {
            "openai": OpenAIProvider(settings.openai_api_key),
            "gemini": GeminiProvider(settings.gemini_api_key),
        }

    def get(self, provider_name: str) -> BaseLLMProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise UnsupportedProviderError(f"Unsupported provider: '{provider_name}'")
        return provider
