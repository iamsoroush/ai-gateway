# Technical decisions (decision log)

> Layer: **rationale**. Start at [../CLAUDE.md](../CLAUDE.md). Siblings:
> [product.md](product.md) · [architecture.md](architecture.md) ·
> [api-contract.md](api-contract.md). **Read this before reversing a design choice** —
> each entry records *why*, so you don't undo something load-bearing.

Format per entry: **Decision → Why → Consequences / how to change**.

---

### D1. Public contract is OpenAI-compatible
**Decision.** The gateway exposes the OpenAI Chat Completions shape.
**Why.** Callers can reuse the battle-tested `openai` SDK (typed `ChatCompletion` /
`ChatCompletionChunk` objects) with zero custom client code, and existing OpenAI
clients work unchanged — regardless of which provider serves the request.
**Consequences.** Response shape is constrained by OpenAI's models; a compatibility
test ([../tests/test_openai_sdk_compat.py](../tests/test_openai_sdk_compat.py)) guards it.
The gateway's `audio_url` is a non-standard extension (use `input_audio` via the SDK).

### D2. Route is provider-agnostic; everything flows through a canonical layer
**Decision.** `app/api/routes.py` depends only on `BaseLLMProvider`; OpenAI requests are
normalized to `CanonicalLLMRequest`, and providers return `CanonicalLLMResponse` /
`StreamEvent`, which the route maps back to the OpenAI shape.
**Why.** Decouples the API from providers so adding/replacing a provider never touches the
route, and the contract can evolve independently of provider wire formats.
**Consequences.** New providers = new adapter + router registration. Don't leak provider
specifics into the route or normalizer.

### D3. Provider SDK imports are lazy
**Decision.** `openai` / `google-genai` are imported inside provider methods, not at module top.
**Why.** The app and the whole test suite import and run without those SDKs installed or any
API key set; failures surface as clean `MissingAPIKeyError`s, not import errors.
**Consequences.** Keep imports lazy. Tests rely on this + injected fakes to avoid real calls.

### D4. Model selection: alias → raw name → content-based default
**Decision.** `model` resolves as: registered alias, else raw provider model name (provider
inferred from prefix), else (omitted) a content-based default (audio → `DEFAULT_AUDIO_MODEL`,
else `DEFAULT_MODEL`).
**Why.** Aliases give central control and indirection; raw names give flexibility without a
registry edit; the content-based default routes audio to an audio-capable model automatically.
**Consequences.** Inference is prefix-based (`config.PROVIDER_NAME_PREFIXES`); unknown prefixes
→ `404`. Evolved from "aliases only" per explicit product requests. See
[architecture.md](architecture.md#request-lifecycle-chat-completions).

### D5. Stores are behind interfaces on `app.state`; usage store is selectable
**Decision.** `ProviderRouter` and `UsageStore` are constructed in `main.py` and held on
`app.state`. The usage store is chosen by config: `SQLiteUsageStore` when `USAGE_DB_PATH`
is set (durable, survives restarts), else `InMemoryUsageStore` (process-local).
**Why.** Keeps a clean seam — tests swap fakes via `app.state`, and the durable store was
added behind the same `UsageStore` interface with no change to routes/aggregation. SQLite
needs no external service, which fits the single-instance MVP.
**Consequences.** Both stores hold only token counts (cost stays query-time, so price edits
re-price history with no migration — see D10). SQLite is single-file and serialized with a
lock: fine for one process, not shared across workers/instances — swap for Postgres for that.
In-memory remains the default when `USAGE_DB_PATH` is unset and still resets on restart.

### D6. PHI-safe structured logging
**Decision.** Logs are JSON metadata only (request id, model, provider, latency, errors);
prompts, media URLs, audio, and generated content are never logged. The 422 handler also
strips Pydantic's echoed input values.
**Why.** The target use cases involve medical content; leaking it into logs is unacceptable.
**Consequences.** This is a hard rule. Don't add content to logs or error bodies.

### D7. Media: URL fetch + `data:` URI decoding in a shared helper
**Decision.** [../app/utils/media.py](../app/utils/media.py) resolves a media reference to
bytes, handling both HTTP(S) URLs and inline `data:` URIs.
**Why.** OpenAI accepts image URLs directly, but Gemini needs inline bytes; and callers want
to send local files. Data-URI support lets a local file ride in as base64 without a public URL.
**Consequences.** Gemini image/audio URLs (and data URIs) are downloaded/decoded; OpenAI image
URLs pass through untouched.

### D8. Streaming usage via `StreamEvent` (interface deviation from `-> str`)
**Decision.** `stream_complete` yields `StreamEvent` (text `delta` + optional terminal `usage`)
instead of plain `str`.
**Why.** Streaming requests must be accountable in usage stats; the only clean way to surface
end-of-stream token counts through the canonical layer was a small richer event.
**Consequences.** OpenAI streaming sets `stream_options={"include_usage": True}`; Gemini's
last chunk carries `usage_metadata`. `sse_stream` formats deltas and collects usage via
`UsageCollector`. This intentionally supersedes the original `AsyncIterator[str]` spec.

### D9. Per-modality token breakdown is best-effort
**Decision.** Usage records carry `input/output_modality_tokens`; providers fill what they
report (Gemini: full text/image/audio breakdown; OpenAI: audio tokens), and anything
unbroken is attributed to `text`.
**Why.** Providers expose modality detail inconsistently; attributing the remainder to text
keeps the modality view always populated and useful without overstating precision.
**Consequences.** Image tokens on OpenAI fold into `text` (OpenAI doesn't report them separately).

### D10. Cost computed at query time, modality-aware
**Decision.** Cost is computed when usage is queried (not stored on the record), per
modality: a model's `input`/`output` rate is either a flat USD-per-1M number or a
`{modality: rate}` map (with optional `default`). `estimate_cost` prices each modality
bucket of a record separately.
**Why.** Pricing changes should apply to historical usage; and some models charge
different rates for audio/image vs text input, so a single blended rate misestimates
multimodal traffic.
**Consequences.** The static table seeds **provider list prices** for the GPT-5+ and Gemini-2.5+
families, verified against the OpenAI and Google pricing pages (June 2026). Token-tiered models
(Gemini `*-pro`: ≤200k vs >200k tokens) use the smaller-context tier — the schema has no
token-threshold dimension. Because cost is computed at query time, editing a rate re-prices all
historical usage on the next query (no migration). Unpriced models contribute `0`. Where rates are
configured as flat numbers, modality-aware costing reduces to the old totals × rate (backward
compatible). The *source* of rates is pluggable — see D16.

### D11. Consistent error envelope via `GatewayError`
**Decision.** Domain errors subclass `GatewayError` (carry `status_code`/`type`/`code`); one
handler renders `{"error": {...}}`. Validation errors are reshaped to match.
**Why.** Predictable, machine-parseable errors for callers; one place to evolve error policy.
**Consequences.** Add new error types in [../app/models/errors.py](../app/models/errors.py);
don't raise bare `HTTPException`. Table in [api-contract.md](api-contract.md#errors).

### D12. Config & registries are plain dicts in `config.py`
**Decision.** `MODEL_REGISTRY`, `PRICING`, provider prefixes live as dicts in
[../app/config.py](../app/config.py); secrets/tunables come from `.env` via `Settings`.
**Why.** Simplest thing that works for the MVP, isolated so it can be swapped for a DB/config
service without changing any caller (they use `resolve_model` / `get_pricing`).
**Consequences.** Editing a model mapping or price is a one-line change; no migration.

### D13. Docker host port is `8081`
**Decision.** `docker-compose.yml` maps host `8081` → container `8080`.
**Why.** Host `8080` is commonly occupied (e.g. other local dev tools), which blocked startup.
**Consequences.** Service is reached at `http://localhost:8081`. Container still listens on 8080.

### D14. `gpt-5.4-nano` model name (now a real model)
**Decision.** `report-large` and `DEFAULT_MODEL` map to `gpt-5.4-nano` per an explicit request.
**Why.** Requested by the product owner; the gateway forwards the name as-is.
**Consequences.** Originally a placeholder that may not have existed, `gpt-5.4-nano` shipped as a
real OpenAI model (GPT-5.4 nano tier, $0.20/$1.25 per 1M per the June-2026 pricing page), so live
calls resolve and `PRICING` carries its real rate. Revisit the alias if the product picks a
different size (`report-large` currently targets the *nano* tier).

### D15. Structured output (`response_format`) is honored by every provider
**Decision.** The OpenAI `response_format` field flows through the canonical request
(`CanonicalLLMRequest.response_format`) and each adapter honors it: OpenAI passes it
through natively; Gemini translates it (`_structured_output` in
[../app/providers/gemini_provider.py](../app/providers/gemini_provider.py)) into a
`application/json` response mime type plus a native response schema. Gemini prefers
`response_json_schema` (accepts a full JSON Schema, incl. `$ref`/`$defs`/
`additionalProperties` as Pydantic/the OpenAI SDK emit) and falls back to
`response_schema` on SDKs that predate it.
**Why.** Callers using the OpenAI SDK's structured-output feature
(`response_format=` / `.parse(response_format=PydanticModel)`) should get the same
schema-constrained JSON no matter which provider serves the request — the whole point
of an OpenAI-compatible gateway (D1, D2). The translation lives in the Gemini adapter
so the route and canonical layer stay provider-agnostic.
**Consequences.** Translation is in the adapter, not the route. `{"type":"json_object"}`
→ JSON mode; `{"type":"json_schema",…}` → schema-constrained; `{"type":"text"}`/unknown
→ no constraint (lenient, like other forward-compatible fields). To support a new
provider's structured output, translate `response_format` inside its adapter.

### D16. Pricing is a pluggable source (hosted JSON), with static fallback
**Decision.** Prices are resolved through `PricingService`
([../app/services/pricing.py](../app/services/pricing.py)) on `app.state.pricing`. When
`PRICING_SOURCE_URL` is set, it fetches a hosted JSON you control, caches it on a TTL
(`PRICING_REFRESH_SECONDS`), and **remote rates override** the static `config.PRICING`
table per model. On any fetch/parse failure it keeps the last-known-good (or static)
rates. The usage endpoints `await pricing.refresh_if_stale()` then pass `pricing.get` as
the `price_of` lookup into aggregation.
**Why.** Model prices change over time and shouldn't require a code edit + redeploy. There
is **no first-party pricing API** from OpenAI/Google (their `/models` endpoints carry no
prices), so the realistic source is a JSON document you host (config service / object
store / mirror). The hosted-JSON option was chosen over third-party catalogs (LiteLLM,
OpenRouter) to avoid an external dependency and provider-name mapping, and to keep control
of the data in a PHI-adjacent context.
**Consequences.** Default (no URL) behaves exactly as before — static table only, no
network. Refresh is lazy (on the first usage request after the TTL elapses) and throttled
even on failure, so cost estimation never breaks and a broken source isn't hammered. The
JSON is shaped like `config.PRICING` (optionally under a `"models"` key). Aggregation takes
`price_of` as a parameter (default `config.get_pricing`) so unit tests price against the
static table while the route prices against the live service.

### D17. Reasoning effort via OpenAI's `reasoning_effort`, translated per provider
**Decision.** The public knob for "thinking" is OpenAI's `reasoning_effort`
(`minimal`/`low`/`medium`/`high`), carried on the canonical request. OpenAI forwards it
natively; Gemini translates it in the adapter — `thinking_level` for Gemini 3+ (its enum
matches the effort names 1:1), an integer `thinking_budget` for Gemini 2.5
(`_thinking_spec` in [../app/providers/gemini_provider.py](../app/providers/gemini_provider.py)).
**Why.** Callers already use the OpenAI SDK, so one OpenAI-shaped field keeps the contract
provider-agnostic (invariant: translate at the edges, providers speak canonical). Gemini's
two thinking mechanisms are a provider detail that belongs in its adapter, not the route.
**Consequences.** Omitting the field keeps each provider's default behavior. Gemini 2.5
budgets are approximate and clamped by the API to the model's range (`minimal`→`0` can't
disable thinking on models that don't allow it). Sending an effort to a non-reasoning model
surfaces the provider's error; an unknown value is a `422` (validated by the `Literal`).
New levels are a one-line change to the contract + the two Gemini maps.

---

## Anticipated (not yet built)

Designed-for but intentionally out of scope (see [product.md](product.md#non-goals-for-now)):
auth/authz, request persistence, quotas/rate limits, retries + provider fallback, prompt
templates, response caching, PHI-safe audit logging, billing. The seams above (canonical
layer, provider interface, `app.state` stores, config indirection) exist so these can be
added without rework.
