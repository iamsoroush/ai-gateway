"""Storage for usage records.

`UsageStore` is the interface the rest of the app depends on; `InMemoryUsageStore`
is the MVP implementation. It is process-local and resets on restart — fine for a
single-instance MVP. To make usage durable / shared across workers, implement this
same interface over SQLite/Postgres and swap it in ``app.main`` (nothing else changes).
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol

from app.models.usage import UsageRecord


class UsageStore(Protocol):
    def record(self, record: UsageRecord) -> None: ...

    def query(
        self, start: datetime, end: datetime, provider: str | None = None
    ) -> list[UsageRecord]: ...


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
