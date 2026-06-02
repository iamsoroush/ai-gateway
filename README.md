# ai-gateway

A small internal service that gives other backend services **one OpenAI-compatible
API** for calling LLMs, without those services needing to know about
provider-specific APIs, API keys, or request formats.

Callers speak the OpenAI Chat Completions contract and reference either an
**internal model alias** (e.g. `report-fast`, `report-large`) or a **raw provider
model name** (e.g. `gpt-4o-mini`, `gemini-1.5-pro`). The gateway resolves that to a
concrete provider + model, translates the request into a provider-agnostic
**canonical** format, dispatches it through a provider adapter
(`OpenAIProvider` / `GeminiProvider`), and normalizes the response back into the
OpenAI shape.

```
OpenAI-compatible request
        ↓
FastAPI route
        ↓
Request normalization → canonical format   (services/normalizer.py)
        ↓
Provider router                            (services/router.py)
        ↓
Provider adapter: OpenAIProvider / GeminiProvider
        ↓
Provider-specific API request
        ↓
Normalize provider response → canonical
        ↓
OpenAI-compatible response to caller
```

The FastAPI route never contains provider-specific logic — it depends only on the
`BaseLLMProvider` interface.

> **Working on the code?** Start from [CLAUDE.md](CLAUDE.md) — the entry point to a
> layered context system: [product](docs/product.md), [architecture](docs/architecture.md),
> [API contract](docs/api-contract.md), and [technical decisions](docs/decisions.md).

## Features

- OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming)
- Providers: **OpenAI** and **Google Gemini**
- Multimodal input: text, image URL/data URI, audio URL, and base64 `input_audio`
- **Structured output**: constrain responses to JSON or a JSON Schema via OpenAI's
  `response_format` — works on **both** OpenAI and Gemini (Gemini is translated to
  native JSON mode)
- Flexible model selection: registered **aliases**, **any raw model name**
  (provider auto-detected), or **omit `model`** to auto-pick by content
- Usage stats: token usage by provider and modality (text/image/audio) over a
  time window, with **estimated cost**
- Consistent JSON error envelope
- Structured (JSON) logging that never logs prompts, media, or generated content
- No authentication (MVP) — keys are loaded from `.env`

## Project layout

```
app/
  main.py                 # FastAPI app + exception handlers
  config.py               # Settings + MODEL_REGISTRY + model resolution
  api/routes.py           # endpoints (provider-agnostic)
  models/
    openai_contract.py    # public OpenAI-shaped request/response
    canonical.py          # internal canonical request/response/stream
    errors.py             # error types + JSON envelope
    usage.py              # usage record + stats response models
  providers/
    base.py               # BaseLLMProvider interface
    openai_provider.py    # OpenAI adapter
    gemini_provider.py    # Gemini adapter
  services/
    normalizer.py         # OpenAI request -> canonical + capability checks
    router.py             # provider name -> adapter
    streaming.py          # SSE formatting
    usage.py              # usage recording, aggregation, modality-aware cost
    usage_store.py        # UsageStore interface + in-memory store
    pricing.py            # PricingService: hosted-JSON prices, TTL cache, fallback
  utils/                  # logging, ids, media fetch
examples/openai_sdk_client.py  # runnable examples using the official OpenAI SDK
tests/                    # health, models, normalizer, chat, providers, sdk-compat
```

## Running locally with Docker Compose

1. Create your env file from the template and fill in keys:

   ```bash
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY / GEMINI_API_KEY
   ```

2. Build and start:

   ```bash
   docker compose up --build
   ```

   The service is published on **http://localhost:8081** (the compose file maps
   host `8081` → container `8080`; change the host side in `docker-compose.yml`
   if 8081 is taken). The source is bind-mounted and uvicorn runs with
   `--reload`, so code changes hot-reload.

3. Check it's up:

   ```bash
   curl http://localhost:8081/health
   ```

### Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

## Configuration (`.env`)

| Variable              | Description                                              | Default            |
| --------------------- | -------------------------------------------------------- | ------------------ |
| `OPENAI_API_KEY`      | OpenAI API key (used by the OpenAI provider)             | _(empty)_          |
| `GEMINI_API_KEY`      | Google Gemini API key (used by Gemini)                   | _(empty)_          |
| `LOG_LEVEL`           | Log level (`DEBUG`/`INFO`/`WARNING`/...)                 | `INFO`             |
| `DEFAULT_MODEL`       | Model used when `model` is omitted and there is no audio | `gpt-5.4-nano`     |
| `DEFAULT_AUDIO_MODEL` | Model used when `model` is omitted and audio is present  | `gemini-2.5-flash` |
| `PRICING_SOURCE_URL`  | Optional hosted JSON of model prices (else the static table) | _(empty)_      |
| `PRICING_REFRESH_SECONDS` | How often to refresh prices from `PRICING_SOURCE_URL` | `3600`         |

> Never commit a real `.env`. It is git-ignored; commit only `.env.example`.

## Endpoints

| Method | Path                   | Description                                   |
| ------ | ---------------------- | --------------------------------------------- |
| GET    | `/health`              | Liveness check                                |
| GET    | `/v1/models`           | List internal model aliases + providers       |
| POST   | `/v1/chat/completions` | OpenAI-compatible chat completion             |
| GET    | `/v1/usage`            | Token usage by provider + modality, with cost |
| GET    | `/v1/usage/summary`    | Overall usage totals + estimated cost         |

### `GET /health`

```json
{ "status": "ok", "service": "ai-gateway" }
```

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    { "id": "report-fast", "object": "model", "provider": "gemini", "provider_model": "gemini-2.5-flash" },
    { "id": "report-large", "object": "model", "provider": "openai", "provider_model": "gpt-5.4-nano" }
  ]
}
```

## Models

The `model` field of a chat request accepts **any** of the following:

1. **A registered alias** — resolved via `MODEL_REGISTRY` in [`app/config.py`](app/config.py):

   | Alias          | Provider | Provider model     |
   | -------------- | -------- | ------------------ |
   | `report-fast`  | gemini   | `gemini-2.5-flash` |
   | `report-large` | openai   | `gpt-5.4-nano`     |

2. **Any raw provider model name** — the provider is auto-detected from the name prefix:

   | Name starts with                     | Routed to     |
   | ------------------------------------ | ------------- |
   | `gpt`, `o1`, `o3`, `o4`, `chatgpt`   | OpenAI        |
   | `gemini`, `gemma`                    | Google Gemini |

   Examples: `gpt-4o-mini`, `gemini-1.5-pro`. A name that matches no provider
   returns `404`.

3. **Omitted entirely** — the gateway picks a default based on the request content:
   - contains audio → `DEFAULT_AUDIO_MODEL` (`gemini-2.5-flash`)
   - otherwise → `DEFAULT_MODEL` (`gpt-5.4-nano`)

> **Audio note:** GPT models do not support audio for now. Sending audio to an
> OpenAI model (e.g. `gpt-5.4-nano` / `report-large`) returns `400
> unsupported_content_type`. Use a Gemini model for audio — which is exactly why
> the no-`model` audio default is `gemini-2.5-flash`.

`GET /v1/models` lists the registered aliases; raw pass-through names are not enumerated.

## Typed client (recommended)

You don't need a custom client to get clean, typed responses. Because the gateway
is OpenAI-compatible, the **official `openai` Python SDK** works against it
unchanged and returns OpenAI's own typed Pydantic models (`ChatCompletion`,
`ChatCompletionChunk`) — the same objects you'd get from OpenAI directly, whether
the gateway routes your request to OpenAI or Gemini. One standard SDK for every
backend behind the gateway.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="unused")  # gateway has no auth

# non-streaming -> typed ChatCompletion
resp = client.chat.completions.create(
    model="report-fast",                       # alias or a raw model name (e.g. "gpt-4o-mini")
    messages=[{"role": "user", "content": "Write a short report."}],
)
print(resp.choices[0].message.content)         # typed attribute access
print(resp.usage.total_tokens)

# streaming -> iterator of typed ChatCompletionChunk
for chunk in client.chat.completions.create(
    model="report-fast",
    messages=[{"role": "user", "content": "Write a short report."}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Runnable: [`examples/openai_sdk_client.py`](examples/openai_sdk_client.py)
(`pip install openai`). A few caveats:

- `image_url` and `input_audio` are standard OpenAI content parts. The gateway's
  `audio_url` extension is **not** in the OpenAI types, so via the SDK send audio
  as base64 `input_audio` (still Gemini-only — use `report-fast`).
- The "omit `model` to auto-pick by content" feature isn't reachable through the
  SDK (it requires `model`); just pass an alias or raw model name.
- You wouldn't use `google.genai` here — that speaks Gemini's protocol; the
  gateway speaks OpenAI's.

> Compatibility is covered by [`tests/test_openai_sdk_compat.py`](tests/test_openai_sdk_compat.py),
> which validates the gateway's actual responses against the SDK's `ChatCompletion`
> / `ChatCompletionChunk` models.

## Examples

Each example has a **curl** tab (the raw wire format, handy for non-Python
callers) and a **Python** tab that uses the official **OpenAI SDK**
(`pip install openai`) — returning typed `ChatCompletion` / `ChatCompletionChunk`
objects. Replace `localhost:8081` with your host. The Python tabs reuse the client
and helpers in the first group below; a runnable version of everything is in
[`examples/openai_sdk_client.py`](examples/openai_sdk_client.py).

<details>
<summary><b>▶ Shared client + helpers</b> (used by all Python tabs)</summary>

```python
import base64
import mimetypes
from openai import OpenAI

# The gateway has no auth, so api_key can be any non-empty placeholder.
client = OpenAI(base_url="http://localhost:8081/v1", api_key="unused")


def image_data_uri(path: str) -> str:
    """Read a local image file and turn it into a base64 data URI."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{data}"


def audio_base64(path: str) -> tuple[str, str]:
    """Read a local audio file -> (base64 data, format like 'wav'/'mp3')."""
    data = base64.b64encode(open(path, "rb").read()).decode()
    return data, path.rsplit(".", 1)[-1].lower()
```

</details>

### 1. Text prompt

<details>
<summary><b>curl</b></summary>

```bash
# non-streaming
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "messages": [{"role": "user", "content": "Write a short test report."}]
  }'

# streaming (set "stream": true and use -N)
curl -N -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a short test report."}]
  }'

# a raw model name works too (provider auto-detected):
#   "model": "gpt-5.4-nano"   ->  OpenAI
#   "model": "gemini-1.5-pro" ->  Gemini
```

</details>

<details>
<summary><b>Python</b></summary>

```python
msgs = [{"role": "user", "content": "Write a short test report."}]

# non-streaming -> typed ChatCompletion
resp = client.chat.completions.create(
    model="report-fast", messages=msgs, temperature=0.2, max_tokens=500
)
print(resp.choices[0].message.content)
print(resp.usage.total_tokens)

# streaming -> iterator of typed ChatCompletionChunk
for chunk in client.chat.completions.create(model="report-fast", messages=msgs, stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# a raw model name works too (provider auto-detected by the gateway):
resp = client.chat.completions.create(model="gpt-5.4-nano", messages=msgs)
print(resp.choices[0].message.content)
```

</details>

Example non-streaming response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "report-fast",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Generated report text..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

### 2. Prompt + image

Images are supported on **both** providers (`report-fast`/Gemini and `report-large`/OpenAI).

<details>
<summary><b>curl</b></summary>

```bash
# image by public URL (non-streaming)
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]}
    ]
  }'

# A local file can be sent inline as a data URI:
#   "image_url": {"url": "data:image/jpeg;base64,<base64>"}
# For streaming, add "stream": true and the -N flag (as in example 1).
```

</details>

<details>
<summary><b>Python</b></summary>

```python
# local image file -> base64 data URI
image_msgs = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": image_data_uri("scan.jpg")}},
        ],
    }
]

# non-streaming, both providers:
print(client.chat.completions.create(model="report-fast", messages=image_msgs).choices[0].message.content)   # Gemini
print(client.chat.completions.create(model="report-large", messages=image_msgs).choices[0].message.content)  # OpenAI

# streaming:
for chunk in client.chat.completions.create(model="report-fast", messages=image_msgs, stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

</details>

### 3. Prompt + audio

> **Audio is Gemini-only for now.** Use `report-fast` (`gemini-2.5-flash`).
> Sending audio to a GPT model returns `400 unsupported_content_type`. If you
> **omit** `model` on an audio request, the gateway auto-selects
> `gemini-2.5-flash`.

<details>
<summary><b>curl</b></summary>

```bash
# audio by URL (Gemini)
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Summarize this audio."},
        {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav", "mime_type": "audio/wav"}}
      ]}
    ]
  }'

# audio as base64 input_audio, with "model" OMITTED -> auto-selects gemini-2.5-flash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Summarize this audio."},
        {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
      ]}
    ]
  }'
```

</details>

<details>
<summary><b>Python</b></summary>

```python
# local audio file -> base64 input_audio (a standard OpenAI content part)
data, fmt = audio_base64("note.wav")
audio_msgs = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this audio."},
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
        ],
    }
]

# Gemini only — non-streaming and streaming
resp = client.chat.completions.create(model="report-fast", messages=audio_msgs)
print(resp.choices[0].message.content)
for chunk in client.chat.completions.create(model="report-fast", messages=audio_msgs, stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Sending audio to a GPT model raises openai.BadRequestError (HTTP 400):
#   client.chat.completions.create(model="report-large", messages=audio_msgs)
#   -> 400 unsupported_content_type
#
# Notes:
#  * The gateway's `audio_url` part is not in the OpenAI types — send audio as
#    input_audio via the SDK (or use audio_url via raw curl; see the curl tab).
#  * "Omit model to auto-pick" needs the field absent, which the SDK doesn't do
#    (model is required) — pass an explicit Gemini model for audio.
```

</details>

### 4. Structured output (JSON / JSON schema)

Use OpenAI's `response_format` to make the model return JSON — optionally
constrained to a schema. This works on **both** providers: OpenAI receives it
natively, and the gateway translates it to Gemini's native JSON mode + response
schema. Full JSON Schema (including the `$ref`/`$defs` Pydantic emits) is supported.

<details>
<summary><b>curl</b></summary>

```bash
# Free-form JSON object
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "messages": [{"role": "user", "content": "Give me a patient summary as JSON."}],
    "response_format": {"type": "json_object"}
  }'

# JSON constrained to a schema
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "report-fast",
    "messages": [{"role": "user", "content": "Summarize: BP 120/80, HR 72."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "vitals",
        "schema": {
          "type": "object",
          "properties": {"systolic": {"type": "integer"}, "diastolic": {"type": "integer"}, "hr": {"type": "integer"}},
          "required": ["systolic", "diastolic", "hr"]
        }
      }
    }
  }'
```

</details>

<details>
<summary><b>Python</b></summary>

```python
from pydantic import BaseModel


class Vitals(BaseModel):
    systolic: int
    diastolic: int
    hr: int


# .parse() sends response_format as a json_schema and parses the reply back into
# the model — works whether the gateway routes to OpenAI or Gemini.
completion = client.beta.chat.completions.parse(
    model="report-fast",  # Gemini — translated to native JSON mode + schema
    messages=[{"role": "user", "content": "Summarize: BP 120/80, HR 72."}],
    response_format=Vitals,
)
vitals = completion.choices[0].message.parsed
print(vitals.systolic, vitals.diastolic, vitals.hr)

# Or free-form JSON mode (no schema):
resp = client.chat.completions.create(
    model="report-fast",
    messages=[{"role": "user", "content": "Give me a patient summary as JSON."}],
    response_format={"type": "json_object"},
)
print(resp.choices[0].message.content)  # a JSON string
```

</details>

### Streaming response (SSE) format

Streaming responses are OpenAI-style `chat.completion.chunk` Server-Sent Events,
terminated by `data: [DONE]`:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Report"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" text"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

> For typed Python responses, use the official `openai` SDK — see
> [Typed client (recommended)](#typed-client-recommended).

## Usage & cost

Every successful chat completion (streaming and non-streaming) is recorded with
its provider, model, token counts, and a best-effort per-modality breakdown. Two
endpoints expose aggregated stats over a time window (default **last 30 days**).

### `GET /v1/usage`

Token usage broken down **by provider** and **by modality** (text / image / audio),
for input and output, plus estimated cost.

Query parameters (all optional):

| Param      | Description                                                    | Default   |
| ---------- | -------------------------------------------------------------- | --------- |
| `start`    | Window start, ISO 8601 (e.g. `2026-05-01T00:00:00Z`)           | `end`−30d |
| `end`      | Window end, ISO 8601                                           | now       |
| `provider` | Filter to one provider (`openai`, `gemini`)                    | all       |
| `interval` | Time-series bucketing: `day`, `week`, or `month`               | none      |

```bash
# default: last 30 days, totals + per-provider breakdown
curl "http://localhost:8081/v1/usage"

# a month, bucketed by day, gemini only
curl "http://localhost:8081/v1/usage?provider=gemini&interval=day&start=2026-05-01T00:00:00Z"
```

```json
{
  "start": "2026-05-03T00:00:00Z",
  "end": "2026-06-02T00:00:00Z",
  "interval": null,
  "totals": {
    "requests": 2,
    "input_tokens": 2000000,
    "output_tokens": 2000000,
    "total_tokens": 4000000,
    "input_by_modality": { "text": 1800000, "image": 200000 },
    "output_by_modality": { "text": 2000000 },
    "estimated_cost_usd": 0.825
  },
  "by_provider": {
    "gemini": { "requests": 1, "input_tokens": 1000000, "estimated_cost_usd": 0.375, "...": "..." },
    "openai": { "requests": 1, "input_tokens": 1000000, "estimated_cost_usd": 0.45,  "...": "..." }
  },
  "buckets": null
}
```

When `interval` is given, `buckets` is a time-series (one entry per period) with
the same `totals` + `by_provider` shape.

### `GET /v1/usage/summary`

Overall totals and estimated cost (plus a per-provider cost map). Accepts
`start`, `end`, `provider`.

```bash
curl "http://localhost:8081/v1/usage/summary"
```

```json
{
  "start": "2026-05-03T00:00:00Z",
  "end": "2026-06-02T00:00:00Z",
  "requests": 2,
  "input_tokens": 2000000,
  "output_tokens": 2000000,
  "total_tokens": 4000000,
  "estimated_cost_usd": 0.825,
  "cost_by_provider": { "gemini": 0.375, "openai": 0.45 }
}
```

### Pricing

Rates are USD per 1,000,000 tokens, per provider model. Each side (`input`/`output`)
is either a **flat number** (same rate for every modality) or a **per-modality map**
with an optional `default` — so models that charge more for audio/image than text are
priced correctly:

```python
# app/config.py — the static / fallback table
PRICING = {
    "gpt-5.4-nano":     {"input": 0.05, "output": 0.40},          # flat
    "gemini-2.5-flash": {                                         # per-modality
        "input":  {"text": 0.075, "image": 0.075, "audio": 0.30, "default": 0.075},
        "output": {"text": 0.30, "default": 0.30},
    },
}
```

Cost is computed **at query time** and is **modality-aware**: each modality bucket of a
record is priced at its own rate. Tokens for an unpriced model contribute `0`.

**Prices can change, so they're loadable from a hosted JSON you control** instead of only
the static table. Set `PRICING_SOURCE_URL` (and optionally `PRICING_REFRESH_SECONDS`,
default 3600) to a JSON document shaped like `PRICING` (optionally wrapped in a top-level
`"models"` key):

```json
{ "models": {
  "gemini-2.5-flash": { "input": {"text": 0.075, "audio": 0.30}, "output": {"text": 0.30} },
  "gpt-5.4-nano":     { "input": 0.05, "output": 0.40 }
} }
```

Remote rates **override** the static table per model; the document is fetched + cached on
the refresh interval, and on any fetch failure the gateway falls back to the last-known
(or static) rates so cost estimation never breaks. There is **no first-party pricing API**
from OpenAI/Google, so this is a document you host (config service / object store / mirror).
Prices are still **placeholders** — set real values. See `app/services/pricing.py`.

### Modality breakdown — how it's derived

- **Gemini** reports a full per-modality breakdown (`prompt_tokens_details` /
  `candidates_tokens_details`): text, image, audio, etc.
- **OpenAI** reports audio tokens in `*_tokens_details`; image tokens are folded
  into the prompt count, so the remainder is counted as `text`.
- Anything a provider doesn't break down is attributed to `text`.

## Error format

All errors share a consistent envelope:

```json
{
  "error": {
    "message": "Unknown model: 'foo'. ...",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| Situation                                  | HTTP | `type`                 | `code`                    |
| ------------------------------------------ | ---- | ---------------------- | ------------------------- |
| Unknown / unroutable model                 | 404  | `invalid_request_error`| `model_not_found`         |
| Unsupported content type for model         | 400  | `invalid_request_error`| `unsupported_content_type`|
| Invalid request body                       | 422  | `invalid_request_error`| `invalid_request_body`    |
| Missing provider API key                   | 500  | `internal_error`       | `missing_api_key`         |
| Provider points at no adapter              | 500  | `internal_error`       | `unsupported_provider`    |
| Provider request failed                    | 502  | `provider_error`       | `provider_request_failed` |

## Logging

Logs are emitted as JSON lines and include: request id, model, provider,
provider model, streaming flag, latency, and errors. They deliberately **do not**
include user prompts, image/audio URLs, audio data, or generated content.

## Current limitations (MVP)

- No authentication / authorization.
- No quotas, retries, or provider fallback.
- GPT models do not support audio for now; sending audio to an OpenAI model
  returns `400`. (The OpenAI adapter would accept audio only for audio-capable
  models, e.g. a `*-audio-preview` model.)
- Provider auto-detection covers OpenAI and Gemini name prefixes; a model name
  with an unrecognized prefix returns `404`.
- **Usage stats are stored in memory** (`InMemoryUsageStore`): they reset on
  restart and are not shared across workers/instances. Swap in a SQLite/Postgres
  implementation of `UsageStore` for durable, multi-instance history.
- Pricing rates are placeholders — set real values in `PRICING`.
- Streaming token usage depends on the provider returning a final usage event
  (enabled for OpenAI via `stream_options` and captured from Gemini's last chunk).

## How to add a new provider

1. Create `app/providers/<name>_provider.py` with a class subclassing
   `BaseLLMProvider` and implement:
   - `supported_content_types(provider_model)` — which canonical content types it accepts
   - `ensure_ready()` — raise `MissingAPIKeyError` if not configured
   - `async complete(request)` — return a `CanonicalLLMResponse`
   - `async stream_complete(request)` — yield text deltas (`str`)
   Keep all provider-specific translation inside this file.
2. Register it in `app/services/router.py` under its provider name and pass the
   relevant API key from `Settings`.
3. Add the key to `Settings` in `app/config.py` and to `.env.example`.
4. (Optional) Add its model-name prefixes to `PROVIDER_NAME_PREFIXES` in
   `app/config.py` so raw model names route to it automatically.

No changes to the route are needed.

## How to add a new model alias

Edit `MODEL_REGISTRY` in [`app/config.py`](app/config.py):

```python
MODEL_REGISTRY = {
    "report-fast":  {"provider": "gemini", "provider_model": "gemini-2.5-flash"},
    "report-large": {"provider": "openai", "provider_model": "gpt-5.4-nano"},
    "report-vision": {"provider": "openai", "provider_model": "gpt-5.4-nano"},  # new
}
```

The alias is immediately available via `/v1/models` and `/v1/chat/completions`.
The registry is isolated here so it can later be backed by a database or config
service without touching callers. (You can also just pass a raw model name without
adding an alias at all — see [Models](#models).)

## Tests

Tests use FastAPI's `TestClient` and a fake provider — they never call a real
provider API.

```bash
pip install ".[dev]"
pytest
```

Covered: `/health`, `/v1/models`, request normalization (string and multimodal),
raw model-name routing, content-based default model selection, unknown model,
unsupported provider, unsupported content type, the data-URI media helper,
OpenAI-SDK response compatibility, usage recording + aggregation + cost + the
`/v1/usage` endpoints, and both streaming and non-streaming chat completions.
