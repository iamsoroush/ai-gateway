"""Storage for request records (the ``requests`` table).

`RequestStore` is the interface the rest of the app depends on. Two implementations:
`InMemoryRequestStore` (process-local, resets on restart — the default and what tests
use) and `PostgresRequestStore` (durable, scalable: records survive restarts and can be
shared across workers/instances). Both store only operational metadata (token counts,
status, latency, client info, a cost snapshot); usage cost is *recomputed* at query time
from token counts, so a price change re-prices history with no migration (D10).
``app.main`` picks one based on ``DATABASE_URL`` (nothing else changes).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Protocol

from app.models.usage import RequestRecord


class RequestStore(Protocol):
    def record(self, record: RequestRecord) -> None: ...

    def query(
        self,
        start: datetime,
        end: datetime,
        provider: str | None = None,
        *,
        model: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RequestRecord]: ...


def _as_utc(dt: datetime) -> datetime:
    """Make a datetime timezone-aware (UTC) so it maps cleanly to ``timestamptz``."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class InMemoryRequestStore:
    """Thread-safe in-memory store with a bounded ring of records."""

    def __init__(self, max_records: int = 100_000) -> None:
        self._records: list[RequestRecord] = []
        self._lock = threading.Lock()
        self._max = max_records

    def record(self, record: RequestRecord) -> None:
        with self._lock:
            self._records.append(record)
            overflow = len(self._records) - self._max
            if overflow > 0:
                del self._records[:overflow]

    def query(
        self,
        start: datetime,
        end: datetime,
        provider: str | None = None,
        *,
        model: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RequestRecord]:
        with self._lock:
            snapshot = list(self._records)
        rows = [
            r
            for r in snapshot
            if start <= r.timestamp <= end
            and (provider is None or r.provider == provider)
            and (model is None or r.provider_model == model or r.model_alias == model)
            and (status is None or r.status == status)
        ]
        rows.sort(key=lambda r: r.timestamp, reverse=newest_first)
        if limit is not None:
            rows = rows[:limit]
        return rows


_COLUMNS = (
    "ts, request_id, status, provider, provider_model, model_alias, stream, "
    "error_type, error_code, http_status, latency_ms, "
    "input_tokens, cached_input_tokens, output_tokens, total_tokens, "
    "input_modality_tokens, output_modality_tokens, "
    "has_image, has_audio, cost_usd, client_ip, user_agent"
)
_PLACEHOLDERS = ",".join(["%s"] * len(_COLUMNS.split(",")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    request_id TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    provider_model TEXT,
    model_alias TEXT,
    stream BOOLEAN NOT NULL,
    error_type TEXT,
    error_code TEXT,
    http_status INTEGER,
    latency_ms DOUBLE PRECISION,
    input_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    input_modality_tokens JSONB NOT NULL,
    output_modality_tokens JSONB NOT NULL,
    has_image BOOLEAN NOT NULL,
    has_audio BOOLEAN NOT NULL,
    cost_usd DOUBLE PRECISION,
    client_ip TEXT,
    user_agent TEXT
)
"""


class PostgresRequestStore:
    """Durable, scalable request store backed by Postgres.

    Same interface as :class:`InMemoryRequestStore`, but records persist to a
    Postgres database so request history survives restarts and can be shared across
    workers/instances. Timestamps are stored as ``timestamptz`` (range queries are
    exact regardless of client timezone) and modality breakdowns as ``JSONB``.

    A thread-safe :class:`psycopg_pool.ConnectionPool` backs concurrent access — the
    route may call ``record()``/``query()`` from FastAPI worker threads. Connections
    are autocommit, so each append-only INSERT commits immediately. ``psycopg`` and
    the pool are imported lazily so the module (and the tests) load without the driver
    installed when only the in-memory store is used (mirrors the lazy-SDK rule, D3).
    """

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        from psycopg.types.json import Json
        from psycopg_pool import ConnectionPool

        self._Json = Json
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            open=True,
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(_SCHEMA)
            conn.execute(
                "ALTER TABLE requests "
                "ADD COLUMN IF NOT EXISTS cached_input_tokens INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests (status)")

    def record(self, record: RequestRecord) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"INSERT INTO requests ({_COLUMNS}) VALUES ({_PLACEHOLDERS})",
                (
                    _as_utc(record.timestamp),
                    record.request_id,
                    record.status,
                    record.provider,
                    record.provider_model,
                    record.model_alias,
                    record.stream,
                    record.error_type,
                    record.error_code,
                    record.http_status,
                    record.latency_ms,
                    record.input_tokens,
                    record.cached_input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    self._Json(record.input_modality_tokens),
                    self._Json(record.output_modality_tokens),
                    record.has_image,
                    record.has_audio,
                    record.cost_usd,
                    record.client_ip,
                    record.user_agent,
                ),
            )

    def query(
        self,
        start: datetime,
        end: datetime,
        provider: str | None = None,
        *,
        model: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RequestRecord]:
        sql = f"SELECT {_COLUMNS} FROM requests WHERE ts BETWEEN %s AND %s"
        params: list = [_as_utc(start), _as_utc(end)]
        if provider is not None:
            sql += " AND provider = %s"
            params.append(provider)
        if model is not None:
            sql += " AND (provider_model = %s OR model_alias = %s)"
            params.extend([model, model])
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY ts DESC" if newest_first else " ORDER BY ts"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> RequestRecord:
        (
            ts,
            request_id,
            status,
            provider,
            provider_model,
            model_alias,
            stream,
            error_type,
            error_code,
            http_status,
            latency_ms,
            inp,
            cached_inp,
            out,
            total,
            in_mod,
            out_mod,
            has_image,
            has_audio,
            cost_usd,
            client_ip,
            user_agent,
        ) = row
        # psycopg returns timestamptz as an aware datetime and JSONB as a dict.
        return RequestRecord(
            timestamp=_as_utc(ts),
            request_id=request_id,
            status=status,
            provider=provider,
            provider_model=provider_model,
            model_alias=model_alias,
            stream=stream,
            error_type=error_type,
            error_code=error_code,
            http_status=http_status,
            latency_ms=latency_ms,
            input_tokens=inp,
            cached_input_tokens=cached_inp,
            output_tokens=out,
            total_tokens=total,
            input_modality_tokens=in_mod,
            output_modality_tokens=out_mod,
            has_image=has_image,
            has_audio=has_audio,
            cost_usd=cost_usd,
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def close(self) -> None:
        self._pool.close()
