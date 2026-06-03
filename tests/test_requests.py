"""Tests for the requests table: per-request recording (success + failure),
the /v1/requests listing endpoint, and failure metrics on usage stats."""

from datetime import datetime, timedelta, timezone

from app.models.usage import RequestRecord

BASE = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _img_message():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# Recording: success                                                          #
# --------------------------------------------------------------------------- #


def test_success_request_recorded_with_metadata(usage_client):
    client, _store = usage_client
    r = client.post(
        "/v1/chat/completions",
        json={"model": "report-fast", "messages": _img_message()},
        headers={"x-request-id": "req-123", "user-agent": "pytest-agent"},
    )
    assert r.status_code == 200

    listing = client.get("/v1/requests").json()
    assert listing["count"] == 1
    row = listing["data"][0]
    assert row["status"] == "success"
    assert row["request_id"] == "req-123"
    assert row["provider"] == "gemini"
    assert row["provider_model"] == "gemini-2.5-flash"
    assert row["model_alias"] == "report-fast"
    assert row["has_image"] is True
    assert row["has_audio"] is False
    assert row["total_tokens"] == 5
    assert row["latency_ms"] is not None
    assert row["cost_usd"] is not None and row["cost_usd"] > 0
    assert row["user_agent"] == "pytest-agent"
    assert row["client_ip"]  # TestClient sets a client host
    # PHI-safety: no prompt/content fields leak into the record.
    assert "messages" not in row and "content" not in row


# --------------------------------------------------------------------------- #
# Recording: failures                                                         #
# --------------------------------------------------------------------------- #


def test_unknown_model_failure_recorded(usage_client):
    client, _store = usage_client
    r = client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model-xyz", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404

    errors = client.get("/v1/requests", params={"status": "error"}).json()
    assert errors["count"] == 1
    row = errors["data"][0]
    assert row["status"] == "error"
    assert row["error_code"] == "model_not_found"
    assert row["http_status"] == 404
    # Failed before model resolution: no provider, but the requested name is kept.
    assert row["provider"] is None
    assert row["model_alias"] == "no-such-model-xyz"
    assert row["total_tokens"] == 0

    summary = client.get("/v1/usage/summary").json()
    assert summary["requests"] == 1
    assert summary["failed_requests"] == 1


def test_provider_failure_recorded(failing_client):
    client, _store = failing_client
    r = client.post(
        "/v1/chat/completions",
        json={"model": "report-fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502

    errors = client.get("/v1/requests", params={"status": "error"}).json()
    assert errors["count"] == 1
    row = errors["data"][0]
    assert row["error_code"] == "provider_request_failed"
    assert row["http_status"] == 502
    # Resolution succeeded, so the provider is known on this failure.
    assert row["provider"] == "gemini"
    assert row["provider_model"] == "gemini-2.5-flash"


def test_streaming_failure_recorded(failing_client):
    client, _store = failing_client
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "report-fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        # The response has already started (200); the failure surfaces as an SSE event.
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "provider_request_failed" in body
    assert "[DONE]" in body

    errors = client.get("/v1/requests", params={"status": "error"}).json()
    assert errors["count"] == 1
    row = errors["data"][0]
    assert row["status"] == "error"
    assert row["stream"] is True


def test_mixed_success_and_failure_counts(usage_client):
    client, _store = usage_client
    client.post(
        "/v1/chat/completions",
        json={"model": "report-fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    client.post(
        "/v1/chat/completions",
        json={"model": "bad-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    summary = client.get("/v1/usage/summary").json()
    assert summary["requests"] == 2
    assert summary["failed_requests"] == 1
    # Tokens/cost come from the one success only (failures carry zero tokens).
    assert summary["total_tokens"] == 5
    assert summary["estimated_cost_usd"] > 0


# --------------------------------------------------------------------------- #
# /v1/requests listing: filters, ordering, limit                             #
# --------------------------------------------------------------------------- #


def _seed(store):
    store.record(
        RequestRecord(
            timestamp=BASE,
            status="success",
            provider="gemini",
            provider_model="gemini-2.5-flash",
            model_alias="report-fast",
            latency_ms=10.0,
            total_tokens=100,
        )
    )
    store.record(
        RequestRecord(
            timestamp=BASE + timedelta(minutes=1),
            status="error",
            provider="openai",
            provider_model="gpt-5.4-nano",
            model_alias="report-large",
            error_code="provider_request_failed",
            http_status=502,
            latency_ms=20.0,
        )
    )
    store.record(
        RequestRecord(
            timestamp=BASE + timedelta(minutes=2),
            status="success",
            provider="openai",
            provider_model="gpt-5.4-nano",
            model_alias="report-large",
            latency_ms=30.0,
            total_tokens=50,
        )
    )


def test_requests_listing_newest_first_and_window(usage_client):
    client, store = usage_client
    _seed(store)
    body = client.get(
        "/v1/requests",
        params={"start": (BASE - timedelta(days=1)).isoformat(), "end": (BASE + timedelta(days=1)).isoformat()},
    ).json()
    assert body["count"] == 3
    ts = [row["timestamp"] for row in body["data"]]
    assert ts == sorted(ts, reverse=True)  # newest first


def test_requests_listing_filters(usage_client):
    client, store = usage_client
    _seed(store)
    window = {"start": (BASE - timedelta(days=1)).isoformat(), "end": (BASE + timedelta(days=1)).isoformat()}

    only_err = client.get("/v1/requests", params={**window, "status": "error"}).json()
    assert only_err["count"] == 1
    assert only_err["data"][0]["provider"] == "openai"

    only_gemini = client.get("/v1/requests", params={**window, "provider": "gemini"}).json()
    assert only_gemini["count"] == 1

    by_alias = client.get("/v1/requests", params={**window, "model": "report-large"}).json()
    assert by_alias["count"] == 2

    by_provider_model = client.get(
        "/v1/requests", params={**window, "model": "gemini-2.5-flash"}
    ).json()
    assert by_provider_model["count"] == 1

    limited = client.get("/v1/requests", params={**window, "limit": 1}).json()
    assert limited["count"] == 1


def test_requests_listing_rejects_bad_params(usage_client):
    client, _store = usage_client
    assert client.get("/v1/requests", params={"status": "weird"}).status_code == 422
    assert client.get("/v1/requests", params={"limit": 0}).status_code == 422
    assert client.get("/v1/requests", params={"limit": 5000}).status_code == 422
