# CLAUDE.md — ai-gateway

Internal **OpenAI-compatible gateway**: one API for backend services to call LLMs
(OpenAI, Google Gemini) via internal model aliases — no provider-specific code or
keys at the caller. Python · FastAPI · Pydantic v2 · Docker. MVP, but built to extend.

This is the **entry point** of a layered context system. It stays short; the detail
lives in `docs/`. Read the relevant doc when your task touches that area — don't
guess from this file alone.

## Context layers — read on demand
- **[docs/product.md](docs/product.md)** — what the service is for, who uses it, scope, non-goals.
- **[docs/architecture.md](docs/architecture.md)** — request flow, module map, streaming, requests/usage, extension points. *Read before changing routing, providers, or the requests/usage subsystem.*
- **[docs/api-contract.md](docs/api-contract.md)** — endpoints, request/response schemas, content types, errors. *Read before changing the public API.*
- **[docs/decisions.md](docs/decisions.md)** — why things are the way they are (decision log). *Read before reversing a design choice.*
- **[README.md](README.md)** — user-facing docs: setup, curl/Python examples, pricing.

## Orientation — where things live
- App entry / wiring / exception handlers: [app/main.py](app/main.py)
- HTTP endpoints (provider-agnostic): [app/api/routes.py](app/api/routes.py)
- Config, `MODEL_REGISTRY`, model resolution, `PRICING`: [app/config.py](app/config.py)
- Public OpenAI contract: [app/models/openai_contract.py](app/models/openai_contract.py)
- Internal canonical models + `StreamEvent`: [app/models/canonical.py](app/models/canonical.py)
- Errors + envelope: [app/models/errors.py](app/models/errors.py) · `RequestRecord` + usage/request response models: [app/models/usage.py](app/models/usage.py)
- Provider interface + adapters: [app/providers/](app/providers/) (`base.py`, `openai_provider.py`, `gemini_provider.py`)
- Services: [app/services/](app/services/) (`normalizer.py`, `router.py`, `streaming.py`, `usage.py`, `request_store.py`)
- Utils: [app/utils/](app/utils/) (`logging.py`, `ids.py`, `media.py`)
- Tests: [tests/](tests/) · Runnable client example: [examples/openai_sdk_client.py](examples/openai_sdk_client.py)

## Run / test
- Docker: `docker compose up --build` → http://localhost:8081 (host 8081 → container 8080)
- Local: `uvicorn app.main:app --reload --port 8081`
- Tests: `pytest` — uses `TestClient` + an injected **fake provider**; never calls real LLM APIs and needs no SDKs/keys.
- Config: `.env` (see [.env.example](.env.example)) — `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEFAULT_MODEL`, `DEFAULT_AUDIO_MODEL`, `LOG_LEVEL`, `DATABASE_URL` (set → durable Postgres `requests` table; `docker compose` sets it automatically).

## Invariants — do not break these
1. **The route contains no provider-specific logic.** It depends only on `BaseLLMProvider`. New provider = new adapter + register in `ProviderRouter`; the route does not change. *(decisions D2)*
2. **Everything internal flows through the canonical models.** Translate OpenAI ⇄ canonical in the normalizer/route; providers speak canonical only. *(D2)*
3. **Never log, echo, or persist prompts, media URLs, audio, or generated content.** Logs *and* the `requests` table are metadata only (IP/UA are caller infra, not content). *(D6, D18)*
4. **Keep provider SDK imports lazy** (inside methods) so the app/tests load without `openai`/`google-genai` or keys. *(D3)*
5. **Raise `GatewayError` subclasses, not bare `HTTPException`.** The handler in `main.py` renders the `{"error": {...}}` envelope. *(D11)*
6. **Swappable stores live on `app.state`** behind interfaces (`ProviderRouter`, `RequestStore`). Every routed request (success *and* failure) is recorded; usage stats are computed from the `requests` table. Persists to Postgres when `DATABASE_URL` is set (`psycopg` imported lazily), else in-memory (resets on restart). *(D5, D18)*
7. **Tests must not hit real providers.** Inject fakes via `app.state` (see [tests/conftest.py](tests/conftest.py)).
8. **Keep responses OpenAI-SDK-compatible** — [tests/test_openai_sdk_compat.py](tests/test_openai_sdk_compat.py) must stay green. *(D1)*

## Conventions
- Pydantic v2; set `protected_namespaces=()` when field names start with `model_`.
- Async HTTP where it matters (provider SDKs, media fetch).
- MVP-simple, clean seams. Match the style/comment density of surrounding code.
- When you make a load-bearing design choice, add an entry to [docs/decisions.md](docs/decisions.md).

## Gotchas
- `model` is **optional**: alias → raw name (provider inferred by prefix) → omitted (content-based default: audio→`DEFAULT_AUDIO_MODEL`, else `DEFAULT_MODEL`). *(D4)*
- **Audio is Gemini-only** for now; audio to a GPT model → `400`. *(api-contract)*
- **Reasoning** uses OpenAI's `reasoning_effort` (`minimal|low|medium|high`): forwarded to OpenAI, translated in the Gemini adapter to `thinking_level` (Gemini 3+) or `thinking_budget` (Gemini 2.5). *(D17)*
- Every routed request → one `RequestRecord` row (success *and* failure: status, error, tokens, **cost snapshot**, audio/image flags, latency, model, client IP/UA). `/v1/usage(/summary)` adds `failed_requests` + latency; `/v1/requests` lists rows. Pre-route `422`s aren't recorded. Durable store is **Postgres** via `DATABASE_URL` (else in-memory). *(D5, D18)*
- Cost is **two things**: a realized `cost_usd` snapshot per request **and** query-time recompute from tokens for usage stats (so a price edit re-prices history). Both **modality-aware**; prices load from `PRICING_SOURCE_URL` (hosted JSON) if set, else the static `PRICING` table — provider list prices for the GPT-5+ / Gemini-2.5+ families, verified against the OpenAI/Google pricing pages (June 2026). *(D10, D16, D18)*
- `gpt-5.4-nano` (the `DEFAULT_MODEL` / `report-large` target) is a **real** OpenAI model (GPT-5.4 nano tier) — was originally a placeholder name. *(D14)*
- Docker host port is **8081**, not 8080. *(D13)*
