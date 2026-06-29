"""Tests for modality-aware cost and the hosted-JSON PricingService."""

import asyncio
from datetime import datetime, timezone

import pytest

from app.main import app
from app.models.usage import RequestRecord
from app.services.pricing import PricingService, parse_pricing
from app.services.usage import estimate_cost

STATIC = {
    "flat-model": {"input": 0.10, "output": 0.20},
    "modal-model": {
        "input": {"text": 0.10, "audio": 0.50, "default": 0.10},
        "output": {"text": 0.20, "default": 0.20},
    },
}


# --------------------------------------------------------------------------- #
# Modality-aware cost                                                         #
# --------------------------------------------------------------------------- #


def test_estimate_cost_flat_rates():
    # Flat rate applies to every modality (1M text + 1M image, both @0.10).
    cost = estimate_cost(
        {"input": 0.10, "output": 0.20},
        {"text": 800_000, "image": 200_000},
        {"text": 1_000_000},
    )
    assert cost == 1_000_000 / 1e6 * 0.10 + 1_000_000 / 1e6 * 0.20  # 0.30


def test_estimate_cost_per_modality_rates():
    # Audio priced higher than text on the input side.
    cost = estimate_cost(
        STATIC["modal-model"],
        {"text": 1_000_000, "audio": 1_000_000},
        {"text": 1_000_000},
    )
    # input: 1M*0.10 + 1M*0.50 = 0.60 ; output: 1M*0.20 = 0.20
    assert cost == 0.80


def test_estimate_cost_discounts_cached_input_tokens():
    cost = estimate_cost(
        {"input": 0.20, "cached_input": 0.02, "output": 1.00},
        {"text": 1_000_000},
        {},
        input_tokens=1_000_000,
        cached_input_tokens=250_000,
    )
    assert cost == pytest.approx(0.155)


def test_estimate_cost_cached_input_falls_back_to_input_rate():
    cost = estimate_cost(
        {"input": 0.20, "output": 1.00},
        {"text": 1_000_000},
        {},
        input_tokens=1_000_000,
        cached_input_tokens=250_000,
    )
    assert cost == 0.20


def test_estimate_cost_unknown_modality_uses_default():
    cost = estimate_cost({"input": {"text": 0.10, "default": 0.99}}, {"video": 1_000_000}, {})
    assert cost == 0.99


def test_estimate_cost_no_rates_is_zero():
    assert estimate_cost(None, {"text": 1_000_000}, {"text": 1_000_000}) == 0.0


# --------------------------------------------------------------------------- #
# PricingService                                                              #
# --------------------------------------------------------------------------- #


def test_parse_pricing_accepts_models_wrapper_and_bare():
    assert parse_pricing({"models": {"m": {"input": 1}}}) == {"m": {"input": 1}}
    assert parse_pricing({"m": {"input": 1}, "bad": 5}) == {"m": {"input": 1}}


def test_static_lookup_without_source():
    svc = PricingService(STATIC)
    assert svc.get("flat-model") == {"input": 0.10, "output": 0.20}
    assert svc.get("unknown") is None
    # No source configured -> refresh is a no-op (and still no network).
    asyncio.run(svc.refresh_if_stale())
    assert svc.get("flat-model") == {"input": 0.10, "output": 0.20}


def test_remote_overrides_static():
    async def fake_fetch(url):
        return {"models": {"flat-model": {"input": 9.0, "output": 9.0}}}

    svc = PricingService(STATIC, source_url="http://x", refresh_seconds=0, fetcher=fake_fetch)
    asyncio.run(svc.refresh_if_stale())
    assert svc.get("flat-model") == {"input": 9.0, "output": 9.0}  # remote wins
    assert svc.get("modal-model") == STATIC["modal-model"]  # static still used for others


def test_fetch_failure_keeps_last_known():
    async def boom(url):
        raise RuntimeError("network down")

    svc = PricingService(STATIC, source_url="http://x", refresh_seconds=0, fetcher=boom)
    asyncio.run(svc.refresh_if_stale())  # must not raise
    assert svc.get("flat-model") == {"input": 0.10, "output": 0.20}  # fell back to static


# --------------------------------------------------------------------------- #
# End-to-end: endpoint uses remote pricing                                    #
# --------------------------------------------------------------------------- #


def test_usage_endpoint_uses_remote_pricing(usage_client):
    client, store = usage_client

    async def fake_fetch(url):
        return {
            "models": {
                "gemini-2.5-flash": {
                    "input": {"text": 1.0, "audio": 10.0},
                    "output": {"text": 2.0},
                }
            }
        }

    original = app.state.pricing
    app.state.pricing = PricingService(
        source_url="http://prices", refresh_seconds=0, fetcher=fake_fetch
    )
    try:
        store.record(
            RequestRecord(
                timestamp=datetime.now(timezone.utc),
                provider="gemini",
                provider_model="gemini-2.5-flash",
                model_alias="report-fast",
                input_tokens=2_000_000,
                output_tokens=1_000_000,
                total_tokens=3_000_000,
                input_modality_tokens={"text": 1_000_000, "audio": 1_000_000},
                output_modality_tokens={"text": 1_000_000},
            )
        )
        body = client.get("/v1/usage").json()
        # input: 1M*1.0 + 1M*10.0 = 11.0 ; output: 1M*2.0 = 2.0 ; total 13.0
        assert body["totals"]["estimated_cost_usd"] == 13.0
    finally:
        app.state.pricing = original
