"""Models for usage recording and usage-stats responses.

A :class:`UsageRecord` is stored per successful completion. The aggregate/response
models are what the ``/v1/usage`` endpoints return. Estimated cost is computed at
query time from ``config.PRICING`` (not stored), so rate changes apply retroactively.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UsageRecord(BaseModel):
    """A single accounted request."""

    model_config = ConfigDict(protected_namespaces=())

    timestamp: datetime
    provider: str
    provider_model: str
    model_alias: str
    stream: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Per-modality token counts (e.g. {"text": 10, "image": 258, "audio": 4}).
    input_modality_tokens: dict[str, int] = Field(default_factory=dict)
    output_modality_tokens: dict[str, int] = Field(default_factory=dict)


class UsageAggregate(BaseModel):
    """Summed usage for a scope (overall, a provider, or a time bucket)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_by_modality: dict[str, int] = Field(default_factory=dict)
    output_by_modality: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0


class UsageBucket(BaseModel):
    """Usage for one time interval (when ``interval`` is requested)."""

    start: datetime
    totals: UsageAggregate
    by_provider: dict[str, UsageAggregate]


class UsageStatsResponse(BaseModel):
    """Detailed usage report for ``GET /v1/usage``."""

    start: datetime
    end: datetime
    interval: str | None = None
    totals: UsageAggregate
    by_provider: dict[str, UsageAggregate]
    buckets: list[UsageBucket] | None = None


class UsageSummaryResponse(BaseModel):
    """Overall usage + estimated cost for ``GET /v1/usage/summary``."""

    start: datetime
    end: datetime
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Estimated cost (USD), split by token direction. The ``*_cost_usd`` fields are
    # the overall figures; the ``*_by_provider`` maps break the same numbers down
    # per provider. By construction ``estimated_cost_usd == input_cost_usd + output_cost_usd``.
    estimated_cost_usd: float
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cost_by_provider: dict[str, float]
    input_cost_by_provider: dict[str, float] = Field(default_factory=dict)
    output_cost_by_provider: dict[str, float] = Field(default_factory=dict)
