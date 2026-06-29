"""Tests for usage recording, aggregation, cost, and the /v1/usage endpoints."""

from datetime import datetime, timedelta, timezone

from app.models.usage import RequestRecord
from app.services.usage import aggregate, summarize

BASE = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _records() -> list[RequestRecord]:
    return [
        # Gemini: 1M input (mixed text/image) + 1M output text.
        RequestRecord(
            timestamp=BASE,
            provider="gemini",
            provider_model="gemini-2.5-flash",
            model_alias="report-fast",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
            input_modality_tokens={"text": 800_000, "image": 200_000},
            output_modality_tokens={"text": 1_000_000},
        ),
        # OpenAI: 1M input text + 1M output text, on the next day.
        RequestRecord(
            timestamp=BASE + timedelta(days=1),
            provider="openai",
            provider_model="gpt-5.4-nano",
            model_alias="report-large",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
            input_modality_tokens={"text": 1_000_000},
            output_modality_tokens={"text": 1_000_000},
        ),
    ]


def test_aggregate_totals_modality_and_cost():
    start, end = BASE - timedelta(days=1), BASE + timedelta(days=2)
    stats = aggregate(_records(), start=start, end=end)

    assert stats.totals.requests == 2
    assert stats.totals.input_tokens == 2_000_000
    assert stats.totals.output_tokens == 2_000_000
    # modality sums across providers
    assert stats.totals.input_by_modality == {"text": 1_800_000, "image": 200_000}
    assert stats.totals.output_by_modality == {"text": 2_000_000}

    # cost: gemini 1M input @0.30 + 1M output @2.50 = 2.80 ; openai 1M*0.20 + 1M*1.25 = 1.45
    assert stats.by_provider["gemini"].estimated_cost_usd == 2.80
    assert stats.by_provider["openai"].estimated_cost_usd == 1.45
    assert stats.totals.estimated_cost_usd == 4.25


def test_aggregate_time_range_excludes_outside():
    # The store filters by range; aggregate sees only in-range records. With a
    # window after both records, nothing is in range.
    start, end = BASE + timedelta(days=5), BASE + timedelta(days=10)
    in_range = [r for r in _records() if start <= r.timestamp <= end]
    stats = aggregate(in_range, start=start, end=end)
    assert stats.totals.requests == 0
    assert stats.totals.estimated_cost_usd == 0.0


def test_aggregate_daily_buckets():
    start, end = BASE - timedelta(days=1), BASE + timedelta(days=2)
    stats = aggregate(_records(), start=start, end=end, interval="day")
    assert stats.interval == "day"
    assert stats.buckets is not None
    assert len(stats.buckets) == 2  # two distinct days
    # first bucket is the gemini record's day
    assert "gemini" in stats.buckets[0].by_provider
    assert stats.buckets[0].totals.requests == 1


def test_summarize_overall_and_cost_by_provider():
    start, end = BASE - timedelta(days=1), BASE + timedelta(days=2)
    summary = summarize(_records(), start=start, end=end)
    assert summary.requests == 2
    assert summary.failed_requests == 0
    # These records carry no latency, so the latency stats are absent (not zero).
    assert summary.latency_ms_avg is None
    assert summary.latency_ms_p50 is None
    assert summary.total_tokens == 4_000_000
    assert summary.estimated_cost_usd == 4.25
    assert summary.cost_by_provider == {"gemini": 2.80, "openai": 1.45}

    # Input/output split, overall and per provider.
    # gemini: input 1M*0.30=0.30, output 1M*2.50=2.50 ; openai: 1M*0.20=0.20, 1M*1.25=1.25
    assert summary.input_cost_usd == 0.50
    assert summary.output_cost_usd == 3.75
    assert summary.input_cost_usd + summary.output_cost_usd == summary.estimated_cost_usd
    assert summary.input_cost_by_provider == {"gemini": 0.30, "openai": 0.20}
    assert summary.output_cost_by_provider == {"gemini": 2.50, "openai": 1.25}
    assert summary.embedding_cost_usd == 0.0
    assert summary.embedding_cost_by_provider == {}


def test_embedding_costs_are_included_and_broken_out():
    start, end = BASE - timedelta(days=1), BASE + timedelta(days=3)
    records = _records() + [
        RequestRecord(
            timestamp=BASE + timedelta(days=2),
            provider="openai",
            provider_model="text-embedding-3-small",
            model_alias="text-embedding-3-small",
            input_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            input_modality_tokens={"text": 1_000_000},
            output_modality_tokens={},
        )
    ]

    stats = aggregate(records, start=start, end=end)
    assert stats.totals.estimated_cost_usd == 4.27
    assert stats.totals.embedding_cost_usd == 0.02
    assert stats.by_provider["openai"].estimated_cost_usd == 1.47
    assert stats.by_provider["openai"].embedding_cost_usd == 0.02

    summary = summarize(records, start=start, end=end)
    assert summary.estimated_cost_usd == 4.27
    assert summary.input_cost_usd == 0.52
    assert summary.output_cost_usd == 3.75
    assert summary.cost_by_provider == {"gemini": 2.80, "openai": 1.47}
    assert summary.input_cost_by_provider == {"gemini": 0.30, "openai": 0.22}
    assert summary.output_cost_by_provider == {"gemini": 2.50, "openai": 1.25}
    assert summary.embedding_cost_usd == 0.02
    assert summary.embedding_cost_by_provider == {"openai": 0.02}


# --------------------------------------------------------------------------- #
# Endpoint tests                                                              #
# --------------------------------------------------------------------------- #


def test_usage_endpoint_empty(usage_client):
    client, _store = usage_client
    resp = client.get("/v1/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"]["requests"] == 0
    assert body["by_provider"] == {}
    assert body["interval"] is None


def test_chat_completion_is_recorded(usage_client):
    client, _store = usage_client
    r = client.post(
        "/v1/chat/completions",
        json={"model": "report-fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200

    summary = client.get("/v1/usage/summary").json()
    assert summary["requests"] == 1
    assert summary["failed_requests"] == 0
    # The route records latency, so the summary surfaces it.
    assert summary["latency_ms_avg"] is not None
    assert summary["latency_ms_p50"] is not None
    assert summary["input_tokens"] == 3
    assert summary["output_tokens"] == 2
    assert summary["total_tokens"] == 5
    # report-fast -> gemini-2.5-flash, which is priced, so cost > 0.
    assert summary["estimated_cost_usd"] > 0
    assert "gemini" in summary["cost_by_provider"]
    # Input/output split is present, per-provider, and reconciles with the total.
    assert summary["input_cost_usd"] >= 0
    assert summary["output_cost_usd"] >= 0
    assert round(summary["input_cost_usd"] + summary["output_cost_usd"], 8) == summary["estimated_cost_usd"]
    assert "gemini" in summary["input_cost_by_provider"]
    assert "gemini" in summary["output_cost_by_provider"]


def test_streaming_completion_is_recorded(usage_client):
    client, _store = usage_client
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "report-fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        assert r.status_code == 200
        "".join(r.iter_text())  # drain the stream so recording happens

    summary = client.get("/v1/usage/summary").json()
    assert summary["requests"] == 1
    assert summary["total_tokens"] == 5  # captured from the terminal usage event


def test_usage_endpoint_modality_and_provider_filter(usage_client):
    client, store = usage_client
    now = datetime.now(timezone.utc)
    store.record(
        RequestRecord(
            timestamp=now,
            provider="gemini",
            provider_model="gemini-2.5-flash",
            model_alias="report-fast",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_modality_tokens={"text": 60, "audio": 40},
            output_modality_tokens={"text": 50},
        )
    )
    store.record(
        RequestRecord(
            timestamp=now,
            provider="openai",
            provider_model="gpt-5.4-nano",
            model_alias="report-large",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_modality_tokens={"text": 10},
            output_modality_tokens={"text": 5},
        )
    )

    full = client.get("/v1/usage").json()
    assert full["totals"]["requests"] == 2
    assert full["totals"]["input_by_modality"] == {"text": 70, "audio": 40}
    assert set(full["by_provider"]) == {"gemini", "openai"}

    only_gemini = client.get("/v1/usage", params={"provider": "gemini"}).json()
    assert set(only_gemini["by_provider"]) == {"gemini"}
    assert only_gemini["totals"]["input_by_modality"] == {"text": 60, "audio": 40}


def test_usage_endpoint_rejects_bad_interval(usage_client):
    client, _store = usage_client
    resp = client.get("/v1/usage", params={"interval": "hour"})
    assert resp.status_code == 422
