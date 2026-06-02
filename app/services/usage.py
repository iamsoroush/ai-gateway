"""Recording and aggregation of usage, including estimated cost.

Recording: :func:`record_usage` turns a canonical request + provider usage into a
:class:`UsageRecord` and stores it (never raises into the request path — callers
wrap it defensively). Aggregation: :func:`aggregate` / :func:`summarize` roll
records up by provider, modality and (optionally) time bucket, pricing each via a
``price_of`` lookup (defaults to the static ``config.get_pricing``; the route
passes the live, possibly remote-backed ``PricingService.get``). Cost is
**modality-aware**: a model's rates may differ by modality (e.g. audio ≠ text).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from app.config import get_pricing
from app.models.canonical import CanonicalLLMRequest, CanonicalUsage
from app.models.usage import (
    UsageAggregate,
    UsageBucket,
    UsageRecord,
    UsageStatsResponse,
    UsageSummaryResponse,
)
from app.services.usage_store import UsageStore

_COST_DP = 6  # round money to micro-dollars


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons with stored records work."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Recording                                                                   #
# --------------------------------------------------------------------------- #


def build_usage_record(
    request: CanonicalLLMRequest, usage: CanonicalUsage | None, *, stream: bool, at: datetime
) -> UsageRecord:
    inp = out = total = 0
    in_mod: dict[str, int] = {}
    out_mod: dict[str, int] = {}
    if usage is not None:
        inp = usage.prompt_tokens or 0
        out = usage.completion_tokens or 0
        total = usage.total_tokens or (inp + out)
        in_mod = dict(usage.input_modality_tokens or {})
        out_mod = dict(usage.output_modality_tokens or {})
    # Fall back to attributing everything to text so the modality view is always populated.
    if not in_mod and inp:
        in_mod = {"text": inp}
    if not out_mod and out:
        out_mod = {"text": out}
    return UsageRecord(
        timestamp=at,
        provider=request.provider,
        provider_model=request.provider_model,
        model_alias=request.model_alias,
        stream=stream,
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        input_modality_tokens=in_mod,
        output_modality_tokens=out_mod,
    )


def record_usage(
    store: UsageStore, request: CanonicalLLMRequest, usage: CanonicalUsage | None, *, stream: bool
) -> None:
    store.record(build_usage_record(request, usage, stream=stream, at=now_utc()))


# --------------------------------------------------------------------------- #
# Cost                                                                        #
# --------------------------------------------------------------------------- #


# A price lookup maps a provider model name to its rate entry (or None).
PriceOf = Callable[[str], dict | None]


def _rate_for(side: object, modality: str) -> float:
    """Resolve the per-1M rate for a modality on one side (input/output).

    ``side`` is either a flat number (same rate for all modalities) or a
    ``{modality: rate}`` map with an optional ``"default"``.
    """
    if isinstance(side, (int, float)):
        return float(side)
    if isinstance(side, dict):
        if modality in side:
            return float(side[modality])
        if "default" in side:
            return float(side["default"])
    return 0.0


def estimate_cost(
    rates: dict | None,
    input_modality_tokens: dict[str, int],
    output_modality_tokens: dict[str, int],
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Estimate USD cost from a model's ``rates`` and its token breakdown.

    Prices each modality bucket at its own rate (falling back to a flat rate /
    ``default`` / text). If no modality breakdown is present, falls back to the
    plain input/output totals.
    """
    if not rates:
        return 0.0
    inp, out = rates.get("input"), rates.get("output")
    cost = 0.0
    if input_modality_tokens:
        for modality, tokens in input_modality_tokens.items():
            cost += tokens / 1_000_000 * _rate_for(inp, modality)
    else:
        cost += input_tokens / 1_000_000 * _rate_for(inp, "text")
    if output_modality_tokens:
        for modality, tokens in output_modality_tokens.items():
            cost += tokens / 1_000_000 * _rate_for(out, modality)
    else:
        cost += output_tokens / 1_000_000 * _rate_for(out, "text")
    return cost


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


class _Acc:
    """Mutable accumulator; converted to a UsageAggregate at the end."""

    def __init__(self, price_of: PriceOf) -> None:
        self._price_of = price_of
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.input_by_modality: dict[str, int] = {}
        self.output_by_modality: dict[str, int] = {}
        self.cost = 0.0

    def add(self, rec: UsageRecord) -> None:
        self.requests += 1
        self.input_tokens += rec.input_tokens
        self.output_tokens += rec.output_tokens
        self.total_tokens += rec.total_tokens
        for modality, tokens in rec.input_modality_tokens.items():
            self.input_by_modality[modality] = self.input_by_modality.get(modality, 0) + tokens
        for modality, tokens in rec.output_modality_tokens.items():
            self.output_by_modality[modality] = self.output_by_modality.get(modality, 0) + tokens
        self.cost += estimate_cost(
            self._price_of(rec.provider_model),
            rec.input_modality_tokens,
            rec.output_modality_tokens,
            rec.input_tokens,
            rec.output_tokens,
        )

    def to_aggregate(self) -> UsageAggregate:
        return UsageAggregate(
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            input_by_modality=dict(self.input_by_modality),
            output_by_modality=dict(self.output_by_modality),
            estimated_cost_usd=round(self.cost, _COST_DP),
        )


def _accumulate(
    records: list[UsageRecord], price_of: PriceOf
) -> tuple[_Acc, dict[str, _Acc]]:
    totals = _Acc(price_of)
    by_provider: dict[str, _Acc] = {}
    for rec in records:
        totals.add(rec)
        by_provider.setdefault(rec.provider, _Acc(price_of)).add(rec)
    return totals, by_provider


def _bucket_start(ts: datetime, interval: str) -> datetime:
    ts = ts.astimezone(timezone.utc)
    day = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval == "day":
        return day
    if interval == "week":  # ISO week starting Monday
        return day - timedelta(days=ts.weekday())
    if interval == "month":
        return day.replace(day=1)
    raise ValueError(f"unsupported interval: {interval}")


def aggregate(
    records: list[UsageRecord],
    *,
    start: datetime,
    end: datetime,
    interval: str | None = None,
    price_of: PriceOf | None = None,
) -> UsageStatsResponse:
    price_of = price_of or get_pricing
    totals, by_provider = _accumulate(records, price_of)

    buckets: list[UsageBucket] | None = None
    if interval:
        grouped: dict[datetime, list[UsageRecord]] = {}
        for rec in records:
            grouped.setdefault(_bucket_start(rec.timestamp, interval), []).append(rec)
        buckets = []
        for bucket_start in sorted(grouped):
            b_totals, b_by_provider = _accumulate(grouped[bucket_start], price_of)
            buckets.append(
                UsageBucket(
                    start=bucket_start,
                    totals=b_totals.to_aggregate(),
                    by_provider={p: acc.to_aggregate() for p, acc in b_by_provider.items()},
                )
            )

    return UsageStatsResponse(
        start=start,
        end=end,
        interval=interval,
        totals=totals.to_aggregate(),
        by_provider={p: acc.to_aggregate() for p, acc in by_provider.items()},
        buckets=buckets,
    )


def summarize(
    records: list[UsageRecord],
    *,
    start: datetime,
    end: datetime,
    price_of: PriceOf | None = None,
) -> UsageSummaryResponse:
    price_of = price_of or get_pricing
    totals, by_provider = _accumulate(records, price_of)
    return UsageSummaryResponse(
        start=start,
        end=end,
        requests=totals.requests,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        estimated_cost_usd=round(totals.cost, _COST_DP),
        cost_by_provider={p: round(acc.cost, _COST_DP) for p, acc in by_provider.items()},
    )
