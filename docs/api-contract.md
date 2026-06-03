# API contract

> Layer: **interface**. Start at [../CLAUDE.md](../CLAUDE.md). Siblings:
> [product.md](product.md) · [architecture.md](architecture.md) ·
> [decisions.md](decisions.md). **Read this before changing the public API.**
> Schemas live in [../app/models/openai_contract.py](../app/models/openai_contract.py)
> and [../app/models/usage.py](../app/models/usage.py); this is the human-facing spec.
> Full runnable examples are in [../README.md](../README.md).

The public surface mirrors the **OpenAI Chat Completions** API, so the official
`openai` SDK can talk to the gateway unchanged (point it at `<base>/v1`).

## Endpoints

| Method | Path                   | Purpose |
| ------ | ---------------------- | ------- |
| GET    | `/health`              | Liveness. |
| GET    | `/v1/models`           | List registered model aliases + their providers. |
| POST   | `/v1/chat/completions` | Chat completion (streaming or not, multimodal). |
| GET    | `/v1/usage`            | Token usage by provider + modality over a window, with cost. |
| GET    | `/v1/usage/summary`    | Overall usage totals + estimated cost. |

No authentication (MVP).

## `GET /health`
```json
{ "status": "ok", "service": "ai-gateway" }
```

## `GET /v1/models`
```json
{"object": "list", "data": [
  {"id": "report-fast", "object": "model", "provider": "gemini", "provider_model": "gemini-2.5-flash"},
  {"id": "report-large", "object": "model", "provider": "openai", "provider_model": "gpt-5.4-nano"}
]}
```
Lists only registered aliases; raw pass-through model names are not enumerated.

## `POST /v1/chat/completions`

### Request
```jsonc
{
  "model": "report-fast",        // optional — see "Model selection"
  "stream": false,               // default false
  "messages": [ /* >= 1 */ ],
  "temperature": 0.2,            // optional
  "max_tokens": 2000,            // optional
  "response_format": { },        // optional, structured output — see below
  "reasoning_effort": "medium",  // optional, "minimal"|"low"|"medium"|"high" — see below
  "metadata": { }                // optional, ignored by providers
}
```
Unknown extra fields are ignored (forward-compatible with OpenAI clients).

**Messages.** `role` ∈ `system` | `user` | `assistant`. `content` is either a string
or a list of typed parts:

| Part `type`   | Shape | Notes |
| ------------- | ----- | ----- |
| `text`        | `{"type":"text","text":"..."}` | |
| `image_url`   | `{"type":"image_url","image_url":{"url":"https://... or data:..."}}` | URL or base64 data URI. |
| `audio_url`   | `{"type":"audio_url","audio_url":{"url":"...","mime_type":"audio/wav"}}` | Gateway extension (not standard OpenAI). |
| `input_audio` | `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}` | Standard OpenAI audio part. |

### Structured output (`response_format`)
Constrain the model's output to JSON — and optionally to a specific JSON Schema —
using the standard OpenAI `response_format` field. Works the same regardless of
which provider serves the request:

| `response_format`                                                  | Effect |
| ------------------------------------------------------------------ | ------ |
| `{"type": "json_object"}`                                          | JSON mode (free-form JSON object). |
| `{"type": "json_schema", "json_schema": {"name":"...","schema":{…}}}` | JSON constrained to the given JSON Schema. |
| `{"type": "text"}` or omitted                                      | Plain text (default). |

- **OpenAI** receives the field unchanged (native Structured Outputs).
- **Gemini** translates it: `application/json` response mime type + the JSON Schema
  applied as Gemini's native response schema. Full JSON Schema (incl. `$ref`/`$defs`/
  `additionalProperties`, as emitted by Pydantic) is supported.
- Via the **OpenAI SDK**, `client.chat.completions.parse(response_format=PydanticModel)`
  works against either provider — the SDK sends the `json_schema` form and parses the
  JSON back into your model. Unrecognized `response_format` shapes impose no constraint.

### Reasoning effort (`reasoning_effort`)
Control how much a model "thinks" before answering, using the standard OpenAI
`reasoning_effort` field — `minimal` | `low` | `medium` | `high`. Set it the same way
regardless of provider; the gateway translates per provider:

| Provider | Translation |
| -------- | ----------- |
| **OpenAI** | Forwarded as-is (`reasoning_effort`) to reasoning-capable models. |
| **Gemini 3+** | Mapped to `thinking_level` (`MINIMAL`/`LOW`/`MEDIUM`/`HIGH` — a 1:1 match). |
| **Gemini 2.5** | Mapped to a `thinking_budget`: `minimal`→`0` (off where allowed), `low`→`1024`, `medium`→`8192`, `high`→`-1` (dynamic). Budgets are clamped to the model's range. |

- Omitting the field applies no thinking control (each provider's default behavior).
- Via the **OpenAI SDK**: `client.chat.completions.create(..., reasoning_effort="high")`.
- An unrecognized value is rejected with `422`. Sending an effort to a model that doesn't
  support reasoning surfaces the provider's own error.

### Model selection
1. **Registered alias** (`report-fast`, `report-large`) → `MODEL_REGISTRY`.
2. **Raw model name** → provider inferred from prefix: `gpt`/`o1`/`o3`/`o4`/`chatgpt` →
   OpenAI; `gemini`/`gemma` → Gemini. Unrecognized prefix → `404`.
3. **`model` omitted** → content-based default: audio present → `DEFAULT_AUDIO_MODEL`
   (`gemini-2.5-flash`), else `DEFAULT_MODEL` (`gpt-5.4-nano`).

### Multimodal support
- **Text**: all providers.
- **Image**: OpenAI + Gemini.
- **Audio**: **Gemini only** for now. Audio to a GPT model → `400 unsupported_content_type`.
- Unsupported content is **never silently dropped** — it returns a clear `400`.

### Non-streaming response
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "report-fast",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

### Streaming response (`stream: true`)
OpenAI-style SSE `chat.completion.chunk` events, ending with `data: [DONE]`:
```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Report"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## `GET /v1/usage`

Query params (all optional): `start`, `end` (ISO 8601; default last 30 days),
`provider`, `interval` ∈ `day` | `week` | `month`.

```json
{
  "start": "...", "end": "...", "interval": null,
  "totals": {
    "requests": 2, "input_tokens": 2000000, "output_tokens": 2000000, "total_tokens": 4000000,
    "input_by_modality": {"text": 1800000, "image": 200000},
    "output_by_modality": {"text": 2000000},
    "estimated_cost_usd": 4.25
  },
  "by_provider": { "gemini": { /* same shape */ }, "openai": { /* ... */ } },
  "buckets": null
}
```
With `interval`, `buckets` is a time-series of `{start, totals, by_provider}`.

## `GET /v1/usage/summary`
Params: `start`, `end`, `provider`.
```json
{
  "start": "...", "end": "...",
  "requests": 2, "input_tokens": 2000000, "output_tokens": 2000000, "total_tokens": 4000000,
  "estimated_cost_usd": 4.25,
  "input_cost_usd": 0.5,
  "output_cost_usd": 3.75,
  "cost_by_provider": {"gemini": 2.8, "openai": 1.45},
  "input_cost_by_provider": {"gemini": 0.3, "openai": 0.2},
  "output_cost_by_provider": {"gemini": 2.5, "openai": 1.25}
}
```
Cost is split by token direction: `input_cost_usd` / `output_cost_usd` are the
overall figures, and `*_cost_by_provider` break the same numbers down per provider.
By construction `estimated_cost_usd == input_cost_usd + output_cost_usd`.

## Errors

Consistent envelope:
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "model_not_found"}}
```

| Situation                            | HTTP | `type`                  | `code` |
| ------------------------------------ | ---- | ----------------------- | ------ |
| Unknown / unroutable model           | 404  | `invalid_request_error` | `model_not_found` |
| Unsupported content type for model   | 400  | `invalid_request_error` | `unsupported_content_type` |
| Invalid request body                 | 422  | `invalid_request_error` | `invalid_request_body` |
| Missing provider API key             | 500  | `internal_error`        | `missing_api_key` |
| Provider has no adapter              | 500  | `internal_error`        | `unsupported_provider` |
| Provider request failed              | 502  | `provider_error`        | `provider_request_failed` |

## Compatibility guarantee

Responses validate against the OpenAI SDK's `ChatCompletion` / `ChatCompletionChunk`
models — enforced by [../tests/test_openai_sdk_compat.py](../tests/test_openai_sdk_compat.py).
Changes to the response shape must keep that test green.
