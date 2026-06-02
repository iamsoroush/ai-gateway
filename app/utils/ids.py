"""Small helpers for generating opaque identifiers."""

from __future__ import annotations

import secrets


def new_request_id() -> str:
    """Identifier used for log correlation of a single inbound request."""
    return f"req_{secrets.token_hex(8)}"


def new_completion_id() -> str:
    """OpenAI-style chat completion id (``chatcmpl-...``)."""
    return f"chatcmpl-{secrets.token_hex(12)}"
