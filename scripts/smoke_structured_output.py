"""Live smoke test for structured output (response_format) against a real provider.

Exercises the full stack — OpenAI SDK → gateway HTTP → real Gemini — to confirm
that `response_format` actually constrains the model's output. This makes a REAL
provider call, so it needs a working GEMINI_API_KEY (loaded by the gateway from
`.env`) and a running gateway.

Usage:
    # start the gateway (it reads GEMINI_API_KEY from .env), then:
    AI_GATEWAY_URL=http://localhost:8081 python scripts/smoke_structured_output.py

Only the Gemini path is exercised: the default `report-large`/`gpt-5.4-nano` alias
is a placeholder model name (see decisions D14) and would fail a live OpenAI call.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys

from pydantic import BaseModel
from openai import OpenAI

BASE_URL = os.environ.get("AI_GATEWAY_URL", "http://localhost:8081").rstrip("/") + "/v1"
MODEL = os.environ.get("SMOKE_MODEL", "report-fast")  # -> gemini-2.5-flash

client = OpenAI(base_url=BASE_URL, api_key="unused")  # gateway has no auth


class Vitals(BaseModel):
    systolic: int
    diastolic: int
    heart_rate: int


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def test_json_schema_parse() -> bool:
    """`.parse()` sends a json_schema response_format and parses the reply back."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract the vitals as structured data."},
            {"role": "user", "content": "Patient vitals: BP 128 over 84, pulse 76 bpm."},
        ],
        response_format=Vitals,
    )
    parsed = completion.choices[0].message.parsed
    print(f"       parsed={parsed!r}")
    return check(
        "json_schema via .parse() -> typed Vitals",
        isinstance(parsed, Vitals) and all(
            isinstance(v, int) for v in (parsed.systolic, parsed.diastolic, parsed.heart_rate)
        ),
    )


def test_json_object_mode() -> bool:
    """`{"type": "json_object"}` should yield content that is valid JSON."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": 'Return a JSON object {"ok": true} and nothing else.'},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    print(f"       content={content!r}")
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError as exc:
        return check("json_object -> valid JSON content", False, f"not JSON: {exc}")
    return check("json_object -> valid JSON content", isinstance(loaded, dict))


def main() -> int:
    print(f"Smoke-testing structured output against {BASE_URL} (model={MODEL})\n")
    results = []
    for test in (test_json_schema_parse, test_json_object_mode):
        try:
            results.append(test())
        except Exception as exc:  # surface provider/HTTP errors as a failed check
            results.append(check(test.__name__, False, f"{type(exc).__name__}: {exc}"))
        print()
    ok = all(results)
    print("RESULT:", "all checks passed" if ok else "one or more checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
