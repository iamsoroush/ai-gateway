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

### D5. Stores are in-memory, behind interfaces, on `app.state`
**Decision.** `ProviderRouter` and `UsageStore` (`InMemoryUsageStore`) are constructed in
`main.py` and held on `app.state`.
**Why.** Matches the MVP "persistence later" scope while keeping a clean seam: tests swap
fakes via `app.state`, and a durable store can replace the in-memory one with no other change.
**Consequences.** Usage resets on restart and isn't shared across workers/instances. To make
it durable, implement `UsageStore` over SQLite/Postgres and set `app.state.usage_store`.

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

### D10. Cost via a `PRICING` config table, computed at query time
**Decision.** `config.PRICING` holds USD-per-1M-token rates per provider model; cost is
computed when usage is queried, not stored on the record.
**Why.** Pricing changes should apply to historical usage; keeping rates in config (like
`MODEL_REGISTRY`) makes them easy to edit and later move to a DB/config service.
**Consequences.** Rates are **placeholders** (notably the fictional `gpt-5.4-nano`) — set real
values. Unpriced models contribute `0`.

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

### D14. `gpt-5.4-nano` is a placeholder model name
**Decision.** `report-large` maps to `gpt-5.4-nano` per an explicit request.
**Why.** Requested by the product owner; the gateway forwards the name as-is.
**Consequences.** It may not be a real OpenAI model — a live call could `502` if rejected.
Swap it in `MODEL_REGISTRY` / `PRICING` when the real model is chosen.

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

---

## Anticipated (not yet built)

Designed-for but intentionally out of scope (see [product.md](product.md#non-goals-for-now)):
auth/authz, request persistence, quotas/rate limits, retries + provider fallback, prompt
templates, response caching, PHI-safe audit logging, billing. The seams above (canonical
layer, provider interface, `app.state` stores, config indirection) exist so these can be
added without rework.
