"""Building and aggregation of request records, including estimated cost.

Building: :func:`build_request_record` turns a request's provider usage (plus route
context — status, latency, client, content flags) into a :class:`RequestRecord` for
the store. Both successes and failures are recorded; failures carry zero tokens.
Aggregation: :func:`aggregate` / :func:`summarize` roll records up by provider,
modality and (optionally) time bucket, pricing each via a ``price_of`` lookup
(defaults to the static ``config.get_pricing``; the route passes the live, possibly
remote-backed ``PricingService.get``). Cost is recomputed at query time and is
**modality-aware**: a model's rates may differ by modality (e.g. audio ≠ text).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from app.config import get_pricing
from app.models.canonical import CanonicalUsage
from app.models.usage import (
    RequestRecord,
    TokenCostBreakdown,
    UsageAggregate,
    UsageBucket,
    UsageModelAggregate,
    UsageStatsResponse,
    UsageSummaryResponse,
)

# Keep enough precision for cheap embedding calls: 1 token on text-embedding-3-small
# costs $0.00000002, which disappears at 6 decimal places.
_COST_DP = 8


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons with stored records work."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Record building                                                             #
# --------------------------------------------------------------------------- #


def tokens_from_usage(usage: CanonicalUsage | None) -> tuple[int, int, int, int, dict, dict]:
    """Flatten a :class:`CanonicalUsage` into token totals and modality maps.

    Anything not broken down by modality is attributed to ``text`` so the modality
    view is always populated.
    """
    inp = out = total = cached_inp = 0
    in_mod: dict[str, int] = {}
    out_mod: dict[str, int] = {}
    if usage is not None:
        inp = usage.prompt_tokens or 0
        out = usage.completion_tokens or 0
        total = usage.total_tokens or (inp + out)
        cached_inp = min(usage.cached_input_tokens or 0, inp)
        in_mod = dict(usage.input_modality_tokens or {})
        out_mod = dict(usage.output_modality_tokens or {})
    if not in_mod and inp:
        in_mod = {"text": inp}
    if not out_mod and out:
        out_mod = {"text": out}
    return inp, out, total, cached_inp, in_mod, out_mod


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


def _modality_tokens(tokens_by_modality: dict[str, int], total_tokens: int) -> dict[str, int]:
    """Return modality token buckets, falling back to text when only a total exists."""
    if tokens_by_modality:
        return dict(tokens_by_modality)
    return {"text": total_tokens} if total_tokens else {}


def _cached_tokens_by_modality(
    tokens_by_modality: dict[str, int], cached_input_tokens: int
) -> dict[str, int]:
    """Allocate provider-reported cached input tokens across known input modalities.

    OpenAI reports cached prompt tokens as one total, not per modality. Most prompt
    cache use is text, so text receives cached tokens first; any remainder is assigned
    to the other modalities in stable alphabetical order.
    """
    remaining = max(cached_input_tokens, 0)
    if not tokens_by_modality or not remaining:
        return {}
    out: dict[str, int] = {}
    modalities = (["text"] if "text" in tokens_by_modality else []) + sorted(
        m for m in tokens_by_modality if m != "text"
    )
    for modality in modalities:
        if remaining <= 0:
            break
        cached = min(tokens_by_modality.get(modality, 0), remaining)
        if cached:
            out[modality] = cached
            remaining -= cached
    return out


def estimate_cost_by_modality(
    rates: dict | None,
    input_modality_tokens: dict[str, int],
    output_modality_tokens: dict[str, int],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate per-modality USD costs for input and output tokens."""
    in_tokens = _modality_tokens(input_modality_tokens, input_tokens)
    out_tokens = _modality_tokens(output_modality_tokens, output_tokens)
    if not rates:
        return ({m: 0.0 for m in in_tokens}, {m: 0.0 for m in out_tokens})

    inp = rates.get("input")
    cached_inp = rates.get("cached_input", inp)
    out = rates.get("output")
    cached_by_modality = _cached_tokens_by_modality(in_tokens, min(cached_input_tokens, input_tokens))
    in_costs = {
        modality: (
            max(tokens - cached_by_modality.get(modality, 0), 0)
            / 1_000_000
            * _rate_for(inp, modality)
            + cached_by_modality.get(modality, 0)
            / 1_000_000
            * _rate_for(cached_inp, modality)
        )
        for modality, tokens in in_tokens.items()
    }
    out_costs = {
        modality: tokens / 1_000_000 * _rate_for(out, modality)
        for modality, tokens in out_tokens.items()
    }
    return in_costs, out_costs


def estimate_cost_breakdown(
    rates: dict | None,
    input_modality_tokens: dict[str, int],
    output_modality_tokens: dict[str, int],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> tuple[float, float]:
    """Estimate ``(input_cost, output_cost)`` in USD from a model's ``rates``.

    Prices each modality bucket at its own rate (falling back to a flat rate /
    ``default`` / text). If no modality breakdown is present, falls back to the
    plain input/output totals.
    """
    in_costs, out_costs = estimate_cost_by_modality(
        rates,
        input_modality_tokens,
        output_modality_tokens,
        input_tokens,
        output_tokens,
        cached_input_tokens,
    )
    return sum(in_costs.values()), sum(out_costs.values())


def estimate_cost(
    rates: dict | None,
    input_modality_tokens: dict[str, int],
    output_modality_tokens: dict[str, int],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> float:
    """Total USD cost — the sum of the input- and output-token spend."""
    in_cost, out_cost = estimate_cost_breakdown(
        rates,
        input_modality_tokens,
        output_modality_tokens,
        input_tokens,
        output_tokens,
        cached_input_tokens,
    )
    return in_cost + out_cost


def _is_embedding_record(rec: RequestRecord) -> bool:
    model = (rec.provider_model or rec.model_alias or "").lower()
    return model.startswith("text-embedding-")


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def _latency_stats(values: list[float]) -> tuple[float | None, float | None]:
    """Return ``(avg, p50)`` over latency samples, or ``(None, None)`` if empty.

    p50 uses the lower-median convention (no interpolation) for simplicity.
    """
    if not values:
        return None, None
    avg = round(sum(values) / len(values), 2)
    ordered = sorted(values)
    p50 = round(ordered[(len(ordered) - 1) // 2], 2)
    return avg, p50


class _Acc:
    """Mutable accumulator; converted to a UsageAggregate at the end."""

    def __init__(self, price_of: PriceOf) -> None:
        self._price_of = price_of
        self.requests = 0
        self.failed_requests = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.input_by_modality: dict[str, int] = {}
        self.cached_input_by_modality: dict[str, int] = {}
        self.output_by_modality: dict[str, int] = {}
        self.input_cost_by_modality: dict[str, float] = {}
        self.output_cost_by_modality: dict[str, float] = {}
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.embedding_cost = 0.0
        self.latencies: list[float] = []

    @property
    def cost(self) -> float:
        return self.input_cost + self.output_cost

    def add(self, rec: RequestRecord) -> None:
        self.requests += 1
        if rec.status == "error":
            self.failed_requests += 1
        if rec.latency_ms is not None:
            self.latencies.append(rec.latency_ms)
        self.input_tokens += rec.input_tokens
        self.cached_input_tokens += min(rec.cached_input_tokens, rec.input_tokens)
        self.output_tokens += rec.output_tokens
        self.total_tokens += rec.total_tokens
        input_tokens_by_modality = _modality_tokens(rec.input_modality_tokens, rec.input_tokens)
        output_tokens_by_modality = _modality_tokens(rec.output_modality_tokens, rec.output_tokens)
        for modality, tokens in input_tokens_by_modality.items():
            self.input_by_modality[modality] = self.input_by_modality.get(modality, 0) + tokens
        cached_input_by_modality = _cached_tokens_by_modality(
            input_tokens_by_modality, min(rec.cached_input_tokens, rec.input_tokens)
        )
        for modality, tokens in cached_input_by_modality.items():
            self.cached_input_by_modality[modality] = (
                self.cached_input_by_modality.get(modality, 0) + tokens
            )
        for modality, tokens in output_tokens_by_modality.items():
            self.output_by_modality[modality] = self.output_by_modality.get(modality, 0) + tokens
        in_costs, out_costs = estimate_cost_by_modality(
            self._price_of(rec.provider_model),
            input_tokens_by_modality,
            output_tokens_by_modality,
            rec.input_tokens,
            rec.output_tokens,
            rec.cached_input_tokens,
        )
        in_cost = sum(in_costs.values())
        out_cost = sum(out_costs.values())
        self.input_cost += in_cost
        self.output_cost += out_cost
        for modality, cost in in_costs.items():
            self.input_cost_by_modality[modality] = (
                self.input_cost_by_modality.get(modality, 0.0) + cost
            )
        for modality, cost in out_costs.items():
            self.output_cost_by_modality[modality] = (
                self.output_cost_by_modality.get(modality, 0.0) + cost
            )
        if _is_embedding_record(rec):
            self.embedding_cost += in_cost + out_cost

    @staticmethod
    def _token_cost_breakdown(
        total_tokens: int,
        cached_tokens: int,
        total_cost: float,
        tokens_by_modality: dict[str, int],
        cached_by_modality: dict[str, int],
        cost_by_modality: dict[str, float],
    ) -> dict[str, TokenCostBreakdown]:
        out = {
            "total": TokenCostBreakdown(
                tokens=total_tokens,
                cached_tokens=cached_tokens,
                total_cost=round(total_cost, _COST_DP),
            )
        }
        for modality in sorted(tokens_by_modality):
            out[modality] = TokenCostBreakdown(
                tokens=tokens_by_modality[modality],
                cached_tokens=cached_by_modality.get(modality, 0),
                total_cost=round(cost_by_modality.get(modality, 0.0), _COST_DP),
            )
        return out

    def _base_aggregate_kwargs(self) -> dict:
        avg, p50 = _latency_stats(self.latencies)
        return dict(
            requests=self.requests,
            failed_requests=self.failed_requests,
            input_tokens=self._token_cost_breakdown(
                self.input_tokens,
                min(self.cached_input_tokens, self.input_tokens),
                self.input_cost,
                self.input_by_modality,
                self.cached_input_by_modality,
                self.input_cost_by_modality,
            ),
            output_tokens=self._token_cost_breakdown(
                self.output_tokens,
                0,
                self.output_cost,
                self.output_by_modality,
                {},
                self.output_cost_by_modality,
            ),
            total_tokens=self.total_tokens,
            input_by_modality=dict(self.input_by_modality),
            output_by_modality=dict(self.output_by_modality),
            estimated_cost_usd=round(self.cost, _COST_DP),
            latency_ms_avg=avg,
            latency_ms_p50=p50,
        )

    def to_aggregate(self) -> UsageAggregate:
        return UsageAggregate(
            **self._base_aggregate_kwargs(),
            embedding_cost_usd=round(self.embedding_cost, _COST_DP),
        )

    def to_model_aggregate(self) -> UsageModelAggregate:
        return UsageModelAggregate(**self._base_aggregate_kwargs())


def _accumulate(
    records: list[RequestRecord], price_of: PriceOf
) -> tuple[_Acc, dict[str, _Acc], dict[str, _Acc]]:
    totals = _Acc(price_of)
    by_provider: dict[str, _Acc] = {}
    by_model: dict[str, _Acc] = {}
    for rec in records:
        totals.add(rec)
        # Records that failed before model resolution have no provider; they count
        # toward totals/failed_requests but are not bucketed under a null provider.
        if rec.provider is not None:
            by_provider.setdefault(rec.provider, _Acc(price_of)).add(rec)
        if rec.provider_model is not None:
            by_model.setdefault(rec.provider_model, _Acc(price_of)).add(rec)
    return totals, by_provider, by_model


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
    records: list[RequestRecord],
    *,
    start: datetime,
    end: datetime,
    interval: str | None = None,
    price_of: PriceOf | None = None,
) -> UsageStatsResponse:
    price_of = price_of or get_pricing
    totals, by_provider, by_model = _accumulate(records, price_of)

    buckets: list[UsageBucket] | None = None
    if interval:
        grouped: dict[datetime, list[RequestRecord]] = {}
        for rec in records:
            grouped.setdefault(_bucket_start(rec.timestamp, interval), []).append(rec)
        buckets = []
        for bucket_start in sorted(grouped):
            b_totals, b_by_provider, b_by_model = _accumulate(grouped[bucket_start], price_of)
            buckets.append(
                UsageBucket(
                    start=bucket_start,
                    totals=b_totals.to_aggregate(),
                    by_provider={p: acc.to_aggregate() for p, acc in b_by_provider.items()},
                    by_model={m: acc.to_model_aggregate() for m, acc in b_by_model.items()},
                )
            )

    return UsageStatsResponse(
        start=start,
        end=end,
        interval=interval,
        totals=totals.to_aggregate(),
        by_provider={p: acc.to_aggregate() for p, acc in by_provider.items()},
        by_model={m: acc.to_model_aggregate() for m, acc in by_model.items()},
        buckets=buckets,
    )


def summarize(
    records: list[RequestRecord],
    *,
    start: datetime,
    end: datetime,
    price_of: PriceOf | None = None,
) -> UsageSummaryResponse:
    price_of = price_of or get_pricing
    totals, by_provider, by_model = _accumulate(records, price_of)
    latency_avg, latency_p50 = _latency_stats(totals.latencies)
    return UsageSummaryResponse(
        start=start,
        end=end,
        requests=totals.requests,
        failed_requests=totals.failed_requests,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        estimated_cost_usd=round(totals.cost, _COST_DP),
        input_cost_usd=round(totals.input_cost, _COST_DP),
        output_cost_usd=round(totals.output_cost, _COST_DP),
        embedding_cost_usd=round(totals.embedding_cost, _COST_DP),
        cost_by_provider={p: round(acc.cost, _COST_DP) for p, acc in by_provider.items()},
        input_cost_by_provider={p: round(acc.input_cost, _COST_DP) for p, acc in by_provider.items()},
        output_cost_by_provider={p: round(acc.output_cost, _COST_DP) for p, acc in by_provider.items()},
        embedding_cost_by_provider={
            p: round(acc.embedding_cost, _COST_DP)
            for p, acc in by_provider.items()
            if acc.embedding_cost
        },
        cost_by_model={m: round(acc.cost, _COST_DP) for m, acc in by_model.items()},
        input_cost_by_model={m: round(acc.input_cost, _COST_DP) for m, acc in by_model.items()},
        output_cost_by_model={m: round(acc.output_cost, _COST_DP) for m, acc in by_model.items()},
        embedding_cost_by_model={
            m: round(acc.embedding_cost, _COST_DP)
            for m, acc in by_model.items()
            if acc.embedding_cost
        },
        latency_ms_avg=latency_avg,
        latency_ms_p50=latency_p50,
    )
