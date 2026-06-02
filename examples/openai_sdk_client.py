"""Talk to ai-gateway using the official OpenAI Python SDK.

Because the gateway is OpenAI-compatible, the standard `openai` SDK works against
it unchanged and returns clean, typed Pydantic models (`ChatCompletion`,
`ChatCompletionChunk`) — the same objects you'd get from OpenAI directly, whether
the gateway routes your request to OpenAI or Gemini behind the scenes.

This is the recommended way for caller services to get typed responses: there is
no custom client to build or maintain.

Usage:
    pip install openai
    # start the gateway first (docker compose up --build), then:
    python examples/openai_sdk_client.py --image path/to/scan.jpg --audio path/to/note.wav

Notes:
  * The gateway needs no auth, so api_key can be any non-empty placeholder.
  * `image_url` and `input_audio` are standard OpenAI content parts. The gateway's
    `audio_url` extension is NOT part of the OpenAI types, so for audio via the
    SDK use base64 `input_audio` (shown below).
  * Audio is Gemini-only for now — send audio to `report-fast`, not a GPT model.
  * The "omit model to auto-pick" gateway feature isn't reachable via the SDK
    (the SDK requires `model`); just pass an alias or a raw model name.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AI_GATEWAY_URL", "http://localhost:8081") + "/v1",
    api_key="unused",  # gateway has no auth; SDK just requires a non-empty value
)


def image_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def audio_base64(path: str) -> tuple[str, str]:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode(), path.rsplit(".", 1)[-1].lower()


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_pair(label: str, model: str, messages: list, **kwargs) -> None:
    banner(f"{label}  (model={model})")

    # Non-streaming -> typed ChatCompletion
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    print("-- non-streaming --")
    print(resp.choices[0].message.content)        # typed attribute access
    if resp.usage:
        print(f"[usage] total_tokens={resp.usage.total_tokens}")

    # Streaming -> iterator of typed ChatCompletionChunk
    print("\n-- streaming --")
    for chunk in client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def text_examples() -> None:
    msgs = [{"role": "user", "content": "Write a short test report."}]
    run_pair("Text prompt", "report-fast", msgs, temperature=0.2, max_tokens=500)
    run_pair("Text prompt", "report-large", msgs)
    # A raw model name works too (provider auto-detected by the gateway):
    run_pair("Text prompt (raw name)", "gemini-1.5-pro", msgs)


def image_examples(image_path: str | None) -> None:
    if not image_path or not os.path.exists(image_path):
        print("\n[skip] image examples — pass --image <file> to run them")
        return
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
            ],
        }
    ]
    run_pair("Image file", "report-fast", msgs)   # Gemini
    run_pair("Image file", "report-large", msgs)  # OpenAI (images OK)


def audio_examples(audio_path: str | None) -> None:
    if not audio_path or not os.path.exists(audio_path):
        print("\n[skip] audio examples — pass --audio <file> to run them")
        return
    data, fmt = audio_base64(audio_path)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this audio."},
                {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
            ],
        }
    ]
    # Audio is Gemini-only.
    run_pair("Audio file", "report-fast", msgs)


def structured_output_examples() -> None:
    """Constrain the reply to JSON / a JSON schema via `response_format`.

    Works on both providers — OpenAI gets it natively, Gemini is translated to its
    native JSON mode + response schema by the gateway.
    """
    from pydantic import BaseModel

    class Vitals(BaseModel):
        systolic: int
        diastolic: int
        hr: int

    banner("Structured output — JSON schema via .parse()  (model=report-fast)")
    completion = client.beta.chat.completions.parse(
        model="report-fast",  # Gemini, translated by the gateway
        messages=[{"role": "user", "content": "Summarize vitals: BP 120/80, HR 72."}],
        response_format=Vitals,
    )
    print(completion.choices[0].message.parsed)  # a typed Vitals instance

    banner("Structured output — free-form JSON mode  (model=report-large)")
    resp = client.chat.completions.create(
        model="report-large",  # OpenAI, native
        messages=[{"role": "user", "content": "Give me a one-line patient summary as JSON."}],
        response_format={"type": "json_object"},
    )
    print(resp.choices[0].message.content)  # a JSON string


def main() -> None:
    parser = argparse.ArgumentParser(description="ai-gateway via the OpenAI SDK")
    parser.add_argument("--image", help="path to a local image file")
    parser.add_argument("--audio", help="path to a local audio file (wav/mp3/...)")
    args = parser.parse_args()

    print(f"Using gateway at {client.base_url}")
    text_examples()
    image_examples(args.image)
    audio_examples(args.audio)
    structured_output_examples()


if __name__ == "__main__":
    main()
