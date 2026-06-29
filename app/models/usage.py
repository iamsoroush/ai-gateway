"""Models for the requests table and usage-stats responses.

A :class:`RequestRecord` is stored per chat-completion request — **successes and
failures alike** — as PHI-safe operational metadata only (never prompts, media, or
generated content; see decisions D6/D18). The aggregate/response models are what the
``/v1/usage`` and ``/v1/requests`` endpoints return. Estimated cost in the aggregates
is recomputed at query time from ``config.PRICING`` (not from the stored snapshot), so
rate changes apply retroactively (D10); the per-record ``cost_usd`` is the realized
cost snapshotted at request time for audit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestRecord(BaseModel):
    """A single accounted request (one row of the ``requests`` table).

    Stores operational metadata only. ``provider``/``provider_model``/``model_alias``
    are nullable because a request can fail before the model is resolved (e.g. an
    unknown model). Token counts are zero on failures; ``cost_usd`` is the realized
    cost at request time (``None`` when unpriced / on failure).
    """

    model_config = ConfigDict(protected_namespaces=())

    timestamp: datetime
    request_id: str | None = None
    status: Literal["success", "error"] = "success"
    provider: str | None = None
    provider_model: str | None = None
    model_alias: str | None = None
    stream: bool = False
    # Failure detail (mirrors the error envelope; None on success).
    error_type: str | None = None
    error_code: str | None = None
    http_status: int | None = None
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Per-modality token counts (e.g. {"text": 10, "image": 258, "audio": 4}).
    input_modality_tokens: dict[str, int] = Field(default_factory=dict)
    output_modality_tokens: dict[str, int] = Field(default_factory=dict)
    # Whether the request carried image/audio content (modality presence, not content).
    has_image: bool = False
    has_audio: bool = False
    # Realized cost (USD) snapshotted at request time; None when unpriced/failed.
    cost_usd: float | None = None
    # Caller infrastructure metadata (not patient content — see D6/D18).
    client_ip: str | None = None
    user_agent: str | None = None


class UsageAggregate(BaseModel):
    """Summed usage for a scope (overall, a provider, or a time bucket)."""

    requests: int = 0
    # Subset of ``requests`` that errored (success count = requests - failed_requests).
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_by_modality: dict[str, int] = Field(default_factory=dict)
    output_by_modality: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    # Subset of estimated_cost_usd attributable to embedding models.
    embedding_cost_usd: float = 0.0
    # Latency over records that report one (None when no records carry latency).
    latency_ms_avg: float | None = None
    latency_ms_p50: float | None = None


class UsageBucket(BaseModel):
    """Usage for one time interval (when ``interval`` is requested)."""

    start: datetime
    totals: UsageAggregate
    by_provider: dict[str, UsageAggregate]
    by_model: dict[str, UsageAggregate] = Field(default_factory=dict)


class UsageStatsResponse(BaseModel):
    """Detailed usage report for ``GET /v1/usage``."""

    start: datetime
    end: datetime
    interval: str | None = None
    totals: UsageAggregate
    by_provider: dict[str, UsageAggregate]
    by_model: dict[str, UsageAggregate] = Field(default_factory=dict)
    buckets: list[UsageBucket] | None = None


class UsageSummaryResponse(BaseModel):
    """Overall usage + estimated cost for ``GET /v1/usage/summary``."""

    start: datetime
    end: datetime
    requests: int
    failed_requests: int = 0
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Estimated cost (USD), split by token direction. The ``*_cost_usd`` fields are
    # the overall figures; the ``*_by_provider`` maps break the same numbers down
    # per provider. By construction ``estimated_cost_usd == input_cost_usd + output_cost_usd``.
    estimated_cost_usd: float
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    # Standalone embedding spend, also included in estimated/input/provider totals.
    embedding_cost_usd: float = 0.0
    cost_by_provider: dict[str, float]
    input_cost_by_provider: dict[str, float] = Field(default_factory=dict)
    output_cost_by_provider: dict[str, float] = Field(default_factory=dict)
    embedding_cost_by_provider: dict[str, float] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    input_cost_by_model: dict[str, float] = Field(default_factory=dict)
    output_cost_by_model: dict[str, float] = Field(default_factory=dict)
    embedding_cost_by_model: dict[str, float] = Field(default_factory=dict)
    # Latency over records that report one (None when no records carry latency).
    latency_ms_avg: float | None = None
    latency_ms_p50: float | None = None


class RequestListResponse(BaseModel):
    """Per-request listing for ``GET /v1/requests`` (newest first)."""

    start: datetime
    end: datetime
    count: int
    data: list[RequestRecord]
