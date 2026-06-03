"""Integration tests for `PostgresRequestStore`.

These hit a real Postgres, so they only run when ``TEST_DATABASE_URL`` points at a
throwaway database (the ``requests`` table is TRUNCATEd between tests). They are
skipped otherwise — and skipped entirely if the ``psycopg`` driver isn't installed —
so the default unit-test run (in-memory store) stays fast and dependency-free.

Run them with, e.g.:

    docker run --rm -e POSTGRES_PASSWORD=pg -p 5432:5432 -d postgres:16
    TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres pytest tests/test_postgres_store.py
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from app.models.usage import RequestRecord
from app.services.usage import summarize

pytest.importorskip("psycopg", reason="psycopg not installed")

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TEST_DATABASE_URL (a throwaway DB) to run Postgres store tests"
)

BASE = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def pg_store():
    from app.services.request_store import PostgresRequestStore

    store = PostgresRequestStore(DSN)
    with store._pool.connection() as conn:  # fresh state per test
        conn.execute("TRUNCATE requests")
    try:
        yield store
    finally:
        store.close()


def _record(provider="gemini", at=BASE, **kw) -> RequestRecord:
    base = dict(
        timestamp=at,
        status="success",
        provider=provider,
        provider_model="gemini-2.5-flash",
        model_alias="report-fast",
        stream=True,
        latency_ms=12.5,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_modality_tokens={"text": 60, "audio": 40},
        output_modality_tokens={"text": 50},
        has_audio=True,
        cost_usd=0.000123,
        client_ip="10.0.0.7",
        user_agent="pytest",
    )
    base.update(kw)
    return RequestRecord(**base)


def test_record_and_query_roundtrip(pg_store):
    pg_store.record(_record())
    rows = pg_store.query(BASE - timedelta(days=1), BASE + timedelta(days=1))
    assert len(rows) == 1
    r = rows[0]
    # Timestamps come back UTC-aware; JSONB comes back as dicts; bools as bools.
    assert r.timestamp == BASE
    assert r.provider == "gemini"
    assert r.stream is True
    assert r.has_audio is True
    assert r.input_modality_tokens == {"text": 60, "audio": 40}
    assert r.latency_ms == 12.5
    assert r.cost_usd == pytest.approx(0.000123)
    assert r.client_ip == "10.0.0.7"


def test_persists_across_reopen(pg_store):
    from app.services.request_store import PostgresRequestStore

    pg_store.record(_record("gemini"))
    pg_store.record(_record("openai", at=BASE + timedelta(hours=1)))
    pg_store.close()

    reopened = PostgresRequestStore(DSN)
    try:
        rows = reopened.query(BASE - timedelta(days=1), BASE + timedelta(days=1))
        assert len(rows) == 2
        summary = summarize(rows, start=BASE - timedelta(days=1), end=BASE + timedelta(days=1))
        assert summary.requests == 2
        assert summary.estimated_cost_usd > 0
    finally:
        reopened.close()


def test_filters_window_provider_model_status_limit(pg_store):
    pg_store.record(_record("gemini", at=BASE))
    pg_store.record(_record("openai", at=BASE + timedelta(days=2), provider_model="gpt-5.4-nano", model_alias="report-large"))
    pg_store.record(_record("openai", at=BASE + timedelta(days=2, hours=1), status="error", provider_model="gpt-5.4-nano", model_alias="report-large"))

    # Window excludes the day-2 records.
    in_window = pg_store.query(BASE - timedelta(hours=1), BASE + timedelta(hours=1))
    assert [r.provider for r in in_window] == ["gemini"]

    wide = (BASE - timedelta(days=1), BASE + timedelta(days=5))
    assert [r.provider for r in pg_store.query(*wide, provider="openai")] == ["openai", "openai"]
    assert len(pg_store.query(*wide, model="report-large")) == 2
    assert len(pg_store.query(*wide, model="gemini-2.5-flash")) == 1
    assert len(pg_store.query(*wide, status="error")) == 1
    assert len(pg_store.query(*wide, limit=1)) == 1

    # newest_first orders by ts descending.
    newest = pg_store.query(*wide, newest_first=True)
    ts = [r.timestamp for r in newest]
    assert ts == sorted(ts, reverse=True)


def test_null_provider_failure_row(pg_store):
    pg_store.record(
        _record(
            provider=None,
            provider_model=None,
            model_alias="no-such-model",
            status="error",
            error_type="invalid_request_error",
            error_code="model_not_found",
            http_status=404,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_modality_tokens={},
            output_modality_tokens={},
            has_audio=False,
            cost_usd=None,
        )
    )
    rows = pg_store.query(BASE - timedelta(days=1), BASE + timedelta(days=1), status="error")
    assert len(rows) == 1
    r = rows[0]
    assert r.provider is None
    assert r.provider_model is None
    assert r.model_alias == "no-such-model"
    assert r.error_code == "model_not_found"
    assert r.cost_usd is None
    assert r.input_modality_tokens == {}
