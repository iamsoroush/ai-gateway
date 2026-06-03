"""Model-price resolution, optionally from a hosted JSON source.

Model prices change over time, so the gateway can load them from a hosted JSON
URL you control (``PRICING_SOURCE_URL``) instead of only the static table in
``config.PRICING``. Remote rates **override** the static table per model; on any
fetch/parse failure the last-known-good (or static) rates are kept, so cost
estimation never breaks. Refresh is lazy and TTL-throttled.

Note: OpenAI/Google do not publish first-party pricing APIs, so the source is a
JSON document you host (your config service, object store, or a mirror), shaped
like ``config.PRICING`` (optionally wrapped in a top-level ``"models"`` key):

    {
      "models": {
        "gemini-2.5-flash": {
          "input":  {"text": 0.30, "audio": 1.00, "default": 0.30},
          "output": 2.50
        },
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25}
      }
    }
"""

from __future__ import annotations

import threading
import time
from typing import Any, Awaitable, Callable

import httpx

from app.config import PRICING as STATIC_PRICING
from app.utils.logging import get_logger

logger = get_logger()

Fetcher = Callable[[str], Awaitable[Any]]
_FETCH_TIMEOUT = httpx.Timeout(10.0)


async def _http_fetch(url: str) -> Any:
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def parse_pricing(data: Any) -> dict[str, dict]:
    """Normalize a pricing document into ``{provider_model: rates}``.

    Accepts either a bare ``{model: rates}`` mapping or one wrapped in a
    top-level ``"models"`` key. Non-dict entries are ignored.
    """
    if isinstance(data, dict) and isinstance(data.get("models"), dict):
        data = data["models"]
    if not isinstance(data, dict):
        raise ValueError("pricing document must be a JSON object (optionally under 'models')")
    return {model: rates for model, rates in data.items() if isinstance(rates, dict)}


class PricingService:
    """Resolves per-model pricing, with an optional hosted JSON override."""

    def __init__(
        self,
        static_rates: dict[str, dict] | None = None,
        *,
        source_url: str | None = None,
        refresh_seconds: int = 3600,
        fetcher: Fetcher | None = None,
    ) -> None:
        self._static = dict(static_rates if static_rates is not None else STATIC_PRICING)
        self._source_url = source_url or None
        self._refresh_seconds = max(0, int(refresh_seconds))
        self._fetcher = fetcher or _http_fetch
        self._remote: dict[str, dict] = {}
        self._last_attempt: float | None = None
        self._lock = threading.Lock()

    def get(self, provider_model: str) -> dict | None:
        """Current rates for a model (remote overrides static), or ``None``."""
        with self._lock:
            if provider_model in self._remote:
                return self._remote[provider_model]
            return self._static.get(provider_model)

    def _is_stale(self) -> bool:
        if self._last_attempt is None:
            return True
        return (time.monotonic() - self._last_attempt) >= self._refresh_seconds

    async def refresh_if_stale(self) -> None:
        """Fetch + cache remote prices if configured and the TTL has elapsed.

        Never raises: a failed fetch logs a warning and keeps the last-known
        rates. The attempt is timestamped even on failure so we don't hammer a
        broken source within one refresh interval.
        """
        if not self._source_url:
            return
        with self._lock:
            if not self._is_stale():
                return
            self._last_attempt = time.monotonic()

        try:
            parsed = parse_pricing(await self._fetcher(self._source_url))
        except Exception as exc:
            logger.warning("pricing.refresh_failed", extra={"context": {"error": str(exc)}})
            return

        with self._lock:
            self._remote = parsed
        logger.info("pricing.refreshed", extra={"context": {"models": len(parsed)}})
