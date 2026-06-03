# Product definition

> Layer: **product**. Start at [../CLAUDE.md](../CLAUDE.md). Siblings:
> [architecture.md](architecture.md) · [api-contract.md](api-contract.md) ·
> [decisions.md](decisions.md).

## What it is

**ai-gateway** is an internal backend service that gives other backend services a
**single, OpenAI-compatible API** for calling large language models. Callers don't
deal with provider SDKs, provider-specific request formats, or API keys — they send
an OpenAI-style request to the gateway and it routes to the right provider.

## Problem it solves

Without a gateway, every service that needs an LLM has to:
- integrate each provider's SDK and request/response format,
- hold and rotate each provider's API key,
- hard-code concrete model names, and
- re-implement multimodal handling, streaming, errors, and usage tracking.

The gateway centralizes all of that behind one contract.

## Who uses it

Other **first-party backend services** (server-to-server, inside the trust
boundary). It is **not** a public, user-facing API. There is no authentication yet
(see non-goals); it assumes a trusted network for the MVP.

## Core capabilities

- One endpoint, OpenAI Chat Completions shape: `POST /v1/chat/completions`.
- Providers: **OpenAI** and **Google Gemini** (pluggable).
- **Model aliases**: callers use internal names (`report-fast`, `report-large`) that
  map to concrete provider models — so models can be swapped centrally. Callers may
  also pass a raw model name (provider auto-detected) or omit `model` to let the
  gateway pick one based on content.
- **Multimodal input**: text, image (URL or data URI), audio (URL or base64).
- **Streaming and non-streaming** output (OpenAI-style SSE).
- **Per-request records**: one row per request (success *and* failure) with status, tokens,
  realized cost, audio/image flags, model, latency, and caller IP/UA — queryable via
  `/v1/requests`. Metadata only; never prompts/media/content.
- **Usage & cost stats**: computed from those records — tokens by provider and modality over
  time, with estimated cost, failure counts, and latency.
- **PHI-safe logging & storage**: operational metadata only; never prompts/media/content.

The product is aimed at internal use cases like medical report generation, hence the
default aliases (`report-*`) and the strict no-content-logging stance.

## Scope (in)

- Resolving an internal model alias / raw name / default to a provider + model.
- Translating OpenAI requests to/from each provider's format.
- Multimodal, streaming, errors, and usage accounting.
- Configuration via environment / `.env`.

## Non-goals (for now)

These are deliberately **out of scope for the MVP** but the code is structured so
they can be added without rework (see [decisions.md](decisions.md)):

- Authentication / authorization / multi-tenant isolation.
- Quotas / rate limiting, retries, provider fallback/load-balancing.
- Prompt templates, caching, PHI-safe audit logging, billing integration.

## Success criteria

A caller can, through one OpenAI-compatible contract:
1. send text/image/audio, streaming or not, against an internal alias;
2. have it routed to the correct provider with provider details hidden;
3. get a clean OpenAI-shaped response (typed via the `openai` SDK if desired);
4. and have the gateway operators see per-request records and token usage, estimated
   cost, failures, and latency per provider over time — all without any service knowing
   provider keys or formats.
