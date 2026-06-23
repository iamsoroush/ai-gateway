"""Configuration and the internal model alias registry.

Everything that a deployment can tune lives here so that the rest of the code
never reads ``os.environ`` directly. The :data:`MODEL_REGISTRY` is intentionally
a plain dict for the MVP; it can later be replaced by a database or a config
service without changing any callers (they only use :func:`get_model_config`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / ``.env``."""

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    log_level: str = "INFO"
    # Used when a request omits "model". If the request carries audio we pick the
    # audio-capable default; otherwise the general default.
    default_model: str = "gpt-5.4-nano"
    default_audio_model: str = "gemini-2.5-flash"
    # Postgres DSN for the durable `requests` store (e.g.
    # "postgresql://user:pass@host:5432/db"). Set → records persist to Postgres;
    # unset → process-local in-memory store (resets on restart).
    database_url: str | None = None
    # Optional hosted JSON of model prices (see app/services/pricing.py). When set,
    # it overrides the static PRICING table below and is refreshed on this interval.
    # When unset, only the static table is used.
    pricing_source_url: str | None = None
    pricing_refresh_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Maps a caller-facing alias to a concrete provider + provider model.
# Callers can use these aliases (e.g. "report-fast"), but they may also pass a
# raw provider model name directly (see PROVIDER_NAME_PREFIXES / resolve_model).
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "report-fast": {
        "provider": "gemini",
        "provider_model": "gemini-2.5-flash",
    },
    "report-large": {
        "provider": "openai",
        "provider_model": "gpt-5.4-nano",
    },
}

# Callable, general-purpose model IDs advertised by ``GET /v1/models``. Keep
# this separate from MODEL_REGISTRY (caller-friendly aliases) and PRICING (cost
# metadata): neither is an authoritative catalog of selectable models.
# Date-pinned and specialist image/TTS/search models are intentionally omitted.
MODEL_CATALOG: dict[str, str] = {
    # OpenAI
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
    "gpt-5.4": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4-nano": "openai",
    "gpt-5.4-pro": "openai",
    "gpt-5.5": "openai",
    "gpt-5.5-pro": "openai",
    # Google Gemini
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-flash-lite": "gemini",
    "gemini-3-pro-preview": "gemini",
    "gemini-3-flash-preview": "gemini",
    "gemini-3.1-pro-preview": "gemini",
    "gemini-3.1-flash-lite": "gemini",
    "gemini-3.5-flash": "gemini",
}

# When a caller passes a bare model name that is not a registered alias, the
# provider is inferred from the model name prefix. Order matters: the first
# matching provider wins. Extend this list as new providers/models appear.
PROVIDER_NAME_PREFIXES: list[tuple[tuple[str, ...], str]] = [
    (("gpt", "o1", "o3", "o4", "chatgpt", "davinci", "babbage"), "openai"),
    (("gemini", "gemma", "models/gemini"), "gemini"),
]


def infer_provider(model_name: str) -> str | None:
    """Infer the provider for a bare model name, or ``None`` if unrecognized."""
    name = model_name.lower()
    for prefixes, provider in PROVIDER_NAME_PREFIXES:
        if name.startswith(prefixes):
            return provider
    return None


def resolve_model(model: str) -> dict[str, str] | None:
    """Resolve a request ``model`` to ``{"provider", "provider_model"}``.

    Resolution order:
      1. A registered alias in :data:`MODEL_REGISTRY`.
      2. A bare provider model name, with the provider inferred from its prefix.
    Returns ``None`` if neither matches.
    """
    cfg = MODEL_REGISTRY.get(model)
    if cfg is not None:
        return cfg
    provider = infer_provider(model)
    if provider is not None:
        return {"provider": provider, "provider_model": model}
    return None


# Static / fallback pricing per provider model, in USD per 1,000,000 tokens.
# This is the seed table and the fallback when no hosted source is configured (or
# a fetch fails) — see app/services/pricing.py. Rates are provider *list prices*,
# verified against the OpenAI and Google pricing pages (June 2026). Token-tiered
# models (Gemini `*-pro`: <=200k vs >200k tokens) use the smaller-context tier —
# the schema has no token-threshold dimension.
#
# Each side ("input"/"output") is either a flat number (same rate for every
# modality) OR a per-modality map with an optional "default" (used for any modality
# not listed — e.g. image/video fall back to the text rate), e.g.:
#   "gemini-2.5-flash": {
#       "input":  {"text": 0.30, "audio": 1.00, "default": 0.30},
#       "output": 2.50,
#   }
# Cost is computed at query time, so edits here apply to historical usage too.
PRICING: dict[str, dict] = {
    # OpenAI — GPT-5 family (text/image; flat per-side rates). Current lineup is the
    # 5.4 and 5.5 series; gpt-5 base is the original Aug-2025 release.
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},  # latest flagship (Apr 2026)
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    # Google — Gemini 2.5+ family (audio priced above text on the input side;
    # *-pro rates are the <=200k-token tier — the larger-context tier isn't modeled here).
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": {"text": 0.30, "audio": 1.00, "default": 0.30}, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": {"text": 0.10, "audio": 0.30, "default": 0.10}, "output": 0.40},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": {"text": 0.50, "audio": 1.00, "default": 0.50}, "output": 3.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-flash-lite": {"input": {"text": 0.25, "audio": 0.50, "default": 0.25}, "output": 1.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},  # latest GA (May 2026)
}


def get_pricing(provider_model: str) -> dict | None:
    """Return static per-1M-token rates for a model, or ``None`` if unpriced.

    This is the fallback lookup; the live, possibly remote-backed lookup is
    ``app.services.pricing.PricingService.get`` (held on ``app.state.pricing``).
    """
    return PRICING.get(provider_model)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def get_model_config(alias: str) -> dict[str, str] | None:
    """Resolve a model alias to its provider config, or ``None`` if unknown."""
    return MODEL_REGISTRY.get(alias)
