# Architecture

> Layer: **architecture**. Start at [../CLAUDE.md](../CLAUDE.md). Siblings:
> [product.md](product.md) · [api-contract.md](api-contract.md) ·
> [decisions.md](decisions.md). **Read this before changing routing, providers, or
> the usage subsystem.**

## Core idea

The public API speaks the **OpenAI** contract. Internally everything speaks a
provider-agnostic **canonical** format. The HTTP route is glue that never knows
which provider it's talking to — it depends only on the `BaseLLMProvider` interface.

```
OpenAI-compatible request  (app/models/openai_contract.py)
        │
        ▼
FastAPI route              (app/api/routes.py)            ← no provider specifics
        │  normalize_request()        (app/services/normalizer.py)
        ▼
CanonicalLLMRequest        (app/models/canonical.py)      ← resolve model, flatten content
        │  ProviderRouter.get(provider)  (app/services/router.py)
        ▼
Provider adapter           (app/providers/{openai,gemini}_provider.py)
        │  complete() / stream_complete()
        ▼
Provider SDK call → provider response
        │  normalized to CanonicalLLMResponse / StreamEvent
        ▼
Route builds OpenAI response   (+ records usage → app/services/usage.py)
        ▼
OpenAI-compatible response / SSE stream to caller
```

## Module responsibilities

| Module | Responsibility |
| ------ | -------------- |
| [../app/main.py](../app/main.py) | Build the FastAPI app, configure logging, wire `app.state.provider_router` and `app.state.request_store`, register exception handlers. |
| [../app/api/routes.py](../app/api/routes.py) | HTTP endpoints only. Orchestrates normalize → route → validate → delegate → record (every request, success or failure). **Contains no OpenAI/Gemini specifics.** |
| [../app/config.py](../app/config.py) | `Settings` (env/.env), `MODEL_REGISTRY`, provider-name inference, `resolve_model`, `PRICING`. The only place tunables live. |
| [../app/models/openai_contract.py](../app/models/openai_contract.py) | Public request/response schemas (the contract). |
| [../app/models/canonical.py](../app/models/canonical.py) | Internal `CanonicalLLMRequest`/`Response`, `CanonicalUsage`, `StreamEvent`. The lingua franca. |
| [../app/models/errors.py](../app/models/errors.py) | `GatewayError` hierarchy + the JSON error envelope. |
| [../app/models/usage.py](../app/models/usage.py) | `RequestRecord` (one row per request) + usage-stats / request-listing response models. |
| [../app/providers/base.py](../app/providers/base.py) | `BaseLLMProvider` interface: `supported_content_types`, `ensure_ready`, `complete`, `stream_complete`. |
| [../app/providers/openai_provider.py](../app/providers/openai_provider.py) / [gemini_provider.py](../app/providers/gemini_provider.py) | Adapters: canonical ⇄ provider SDK. Lazy SDK imports. |
| [../app/services/normalizer.py](../app/services/normalizer.py) | OpenAI request → canonical; model selection; content-capability validation. |
| [../app/services/router.py](../app/services/router.py) | Provider name → adapter instance. |
| [../app/services/streaming.py](../app/services/streaming.py) | OpenAI-style SSE formatting; `UsageCollector` for streaming usage. |
| [../app/services/usage.py](../app/services/usage.py) | Build request records, aggregate by provider/modality/time (+ failures/latency), modality-aware cost. |
| [../app/services/request_store.py](../app/services/request_store.py) | `RequestStore` interface + `InMemoryRequestStore` and durable `PostgresRequestStore` (the `requests` table). |
| [../app/services/pricing.py](../app/services/pricing.py) | `PricingService`: hosted-JSON prices, TTL cache, static fallback. |
| [../app/utils/](../app/utils/) | `logging` (JSON, PHI-safe), `ids`, `media` (fetch URL / decode `data:` URI). |

## Request lifecycle (chat completions)

1. **Validate body** — `ChatCompletionRequest` (Pydantic). `model` optional; ≥1 message required.
2. **Normalize** — `normalize_request` selects the model and converts to `CanonicalLLMRequest`:
   - **Model selection** (`_select_model` + `resolve_model`): registered alias → else raw
     model name with provider inferred from prefix → else (no `model`) a content-based
     default (`DEFAULT_AUDIO_MODEL` if the request has audio, else `DEFAULT_MODEL`).
     Unresolvable → `UnknownModelError` (404).
   - **Content flattening**: string content → one text part; list content → typed
     `CanonicalContentPart`s.
3. **Route** — `ProviderRouter.get(provider)` returns the adapter (else `UnsupportedProviderError`).
4. **Capability check** — `validate_content_support` rejects content types the
   provider/model can't handle (`UnsupportedContentError` 400). Never silently dropped.
5. **Delegate**:
   - Non-streaming: `await provider.complete(canonical)` → `CanonicalLLMResponse`; route maps it to the OpenAI response.
   - Streaming: `provider.ensure_ready()` first (so a missing key is a normal HTTP error, not a mid-stream event), then stream via `sse_stream`.
6. **Record the request** — best-effort, wrapped so it can never break the response. The
   whole pipeline (steps 2–5) runs inside a try/except so **every routed request is
   persisted** — successes *and* failures (unknown model, unsupported content, missing key,
   provider error). Streaming records once when the stream drains (mid-stream failures are
   captured via the `UsageCollector`). Pre-route `422` body-validation errors are out of
   scope (they never reach the route). See [Requests & usage subsystem](#requests--usage-subsystem).

## Provider adapters

Each adapter translates **canonical ⇄ provider** and isolates all SDK specifics:
- `supported_content_types(provider_model)` declares capabilities (e.g. OpenAI accepts
  audio only for `*-audio` models; Gemini accepts text/image/audio).
- `complete` / `stream_complete` map messages, call the SDK, and normalize the response
  (incl. a best-effort per-modality token breakdown) back to canonical.
- **SDK imports are lazy** (inside methods) so modules and tests import without the SDKs
  or API keys present.
- Media: image URLs pass straight through to OpenAI; Gemini needs inline bytes, so URLs
  (and `data:` URIs) are fetched/decoded via [../app/utils/media.py](../app/utils/media.py).

## Streaming

`stream_complete` yields `StreamEvent`s (a text `delta`, plus an optional terminal
`usage`). `sse_stream` formats each delta as an OpenAI `chat.completion.chunk`,
terminates with `data: [DONE]`, and deposits the terminal usage into a `UsageCollector`
so streaming requests are accounted for. Errors mid-stream are emitted as a final SSE
error event (the HTTP status is already 200 by then) and deposited on the `UsageCollector`
so the request is still recorded as a failure when the stream drains.

## Requests & usage subsystem

- Every routed request → one `RequestRecord` (timestamp, request id, **status** +
  error type/code/HTTP status, provider/model, tokens + per-modality breakdown, realized
  **cost snapshot**, `has_image`/`has_audio` flags, **latency**, caller **IP / user-agent**)
  stored in `app.state.request_store`. **Both successes and failures** are recorded; failures
  carry zero tokens. Stored as PHI-safe metadata only — never prompts, media, or content (D6).
- `/v1/usage` and `/v1/usage/summary` query the store over a window and aggregate by
  provider + modality (+ optional time buckets), plus `failed_requests` and latency
  (avg / p50), estimating cost via `app.state.pricing`. `/v1/requests` lists individual
  records (newest first), filterable by provider / model / status, with a bounded `limit`.
- **Cost is two things**: a realized `cost_usd` is **snapshotted per record** at request time
  (audit), but aggregate cost is still **recomputed at query time** from token counts (so a
  price edit re-prices history — D10) and is **modality-aware** — each modality bucket is
  priced at its own rate (flat or per-modality). Prices come from `PricingService`
  ([../app/services/pricing.py](../app/services/pricing.py)): the static `config.PRICING`
  table by default, or a hosted JSON (`PRICING_SOURCE_URL`) that overrides it, TTL-cached
  with fallback to last-known/static on failure.
- Store is selectable: **Postgres** (`PostgresRequestStore`, the `requests` table) when
  `DATABASE_URL` is set, so history survives restarts and is shareable across workers/instances;
  otherwise **in-memory** (resets on restart). Postgres runs as its own `docker compose` service
  and the app waits on its healthcheck. `psycopg` is imported lazily, so the app and the default
  in-memory test run load without the driver (D3, D5).

## Error handling

Domain errors are `GatewayError` subclasses carrying `status_code`, `error_type`, `code`.
A single handler in `main.py` renders them as `{"error": {message, type, code}}`. Request
validation errors are reshaped to the same envelope (with safe field-location details, no
echoed values). See [api-contract.md](api-contract.md#errors).

## Logging

Structured JSON via [../app/utils/logging.py](../app/utils/logging.py). Each request logs
request id, model, provider, provider model, streaming flag, latency, and errors —
**never** prompts, media URLs, audio, or generated content (PHI-safety is a hard rule).

## Configuration

`Settings` (pydantic-settings) loads from environment / `.env`: `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `LOG_LEVEL`, `DEFAULT_MODEL`, `DEFAULT_AUDIO_MODEL`. `MODEL_REGISTRY`
and `PRICING` are dicts in `config.py`, intentionally isolated so they can later be backed
by a DB/config service without touching callers.

## Extension points

- **Add a provider**: implement `BaseLLMProvider` in `app/providers/<name>_provider.py`
  (keep SDK imports lazy), register it in `ProviderRouter`, add its API key to `Settings`
  + `.env.example`, and optionally add name prefixes to `PROVIDER_NAME_PREFIXES`. No route change.
- **Add a model alias**: add an entry to `MODEL_REGISTRY` (and a `PRICING` row). Available immediately.
- **Persistence**: set `DATABASE_URL` to use the built-in `PostgresRequestStore` (durable,
  shareable across workers/instances). `docker compose` provides the Postgres service. For a
  different backend, implement `RequestStore` behind the same interface and select it in
  `main.py`. Nothing else changes.
- **Change prices**: edit `config.PRICING` (static), or set `PRICING_SOURCE_URL` to a hosted
  JSON that overrides it (rates may be flat or per-modality). No code change for rate updates.
- **Future**: auth middleware, quotas, retries/fallback, prompt templates — all anticipated;
  see [decisions.md](decisions.md). (Per-request persistence is now built — D18.)

## Testing strategy

`pytest` with FastAPI's `TestClient` and a **fake provider** injected via `app.state`
(see [../tests/conftest.py](../tests/conftest.py)). Tests never call real provider APIs and
don't require SDKs or keys (lazy imports + injected fakes). Coverage spans health, models,
normalization, model routing/defaults, errors, media data-URI decoding, OpenAI-SDK response
compatibility, and the usage subsystem.
