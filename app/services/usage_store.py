"""Storage for usage records.

`UsageStore` is the interface the rest of the app depends on. Two implementations:
`InMemoryUsageStore` (process-local, resets on restart — simplest for a single run)
and `SQLiteUsageStore` (durable: records survive restarts). Both store only token
counts; cost is computed at query time, so a price change re-prices history with no
migration. ``app.main`` picks one based on ``USAGE_DB_PATH`` (nothing else changes).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.models.usage import UsageRecord


class UsageStore(Protocol):
    def record(self, record: UsageRecord) -> None: ...

    def query(
        self, start: datetime, end: datetime, provider: str | None = None
    ) -> list[UsageRecord]: ...


def _epoch(dt: datetime) -> float:
    """Seconds since the epoch, treating naive datetimes as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class InMemoryUsageStore:
    """Thread-safe in-memory store with a bounded ring of records."""

    def __init__(self, max_records: int = 100_000) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._max = max_records

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            overflow = len(self._records) - self._max
            if overflow > 0:
                del self._records[:overflow]

    def query(
        self, start: datetime, end: datetime, provider: str | None = None
    ) -> list[UsageRecord]:
        with self._lock:
            snapshot = list(self._records)
        return [
            r
            for r in snapshot
            if start <= r.timestamp <= end and (provider is None or r.provider == provider)
        ]


_COLUMNS = (
    "ts, provider, provider_model, model_alias, stream, "
    "input_tokens, output_tokens, total_tokens, "
    "input_modality_tokens, output_modality_tokens"
)


class SQLiteUsageStore:
    """Durable usage store backed by a SQLite file.

    Same interface as :class:`InMemoryUsageStore`, but records persist to disk so
    usage survives restarts. Timestamps are stored as epoch seconds (so range
    queries are exact regardless of timezone formatting) and modality breakdowns
    as JSON. A single connection is serialized with a lock — ample for the MVP's
    single-process, low-write-rate usage; swap for Postgres to share across workers.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        # Ensure the parent directory exists so SQLite can create the file.
        parent = Path(path).expanduser().parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may call record()/query() from worker
        # threads. We serialize every access with _lock to stay thread-safe.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    provider TEXT NOT NULL,
                    provider_model TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    stream INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    input_modality_tokens TEXT NOT NULL,
                    output_modality_tokens TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_records (ts)")
            self._conn.commit()

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO usage_records ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    _epoch(record.timestamp),
                    record.provider,
                    record.provider_model,
                    record.model_alias,
                    int(record.stream),
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    json.dumps(record.input_modality_tokens),
                    json.dumps(record.output_modality_tokens),
                ),
            )
            self._conn.commit()

    def query(
        self, start: datetime, end: datetime, provider: str | None = None
    ) -> list[UsageRecord]:
        sql = f"SELECT {_COLUMNS} FROM usage_records WHERE ts BETWEEN ? AND ?"
        params: list = [_epoch(start), _epoch(end)]
        if provider is not None:
            sql += " AND provider = ?"
            params.append(provider)
        sql += " ORDER BY ts"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> UsageRecord:
        (ts, provider, provider_model, model_alias, stream, inp, out, total, in_mod, out_mod) = row
        return UsageRecord(
            timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
            provider=provider,
            provider_model=provider_model,
            model_alias=model_alias,
            stream=bool(stream),
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            input_modality_tokens=json.loads(in_mod),
            output_modality_tokens=json.loads(out_mod),
        )
