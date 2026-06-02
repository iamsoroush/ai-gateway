"""Helpers for fetching remote media (images / audio) referenced by URL.

Some providers (e.g. Gemini) need the actual bytes inline rather than a URL, so
we download and hand back the raw content plus a best-effort MIME type.
"""

from __future__ import annotations

import base64
from urllib.parse import unquote_to_bytes

import httpx

# Keep downloads bounded so a malicious/huge URL can't exhaust memory.
_DOWNLOAD_TIMEOUT = httpx.Timeout(30.0)


def _parse_data_uri(uri: str) -> tuple[bytes, str | None]:
    """Decode an RFC 2397 ``data:`` URI into ``(content, mime_type)``.

    Supports both base64 (``data:image/png;base64,...``) and URL-encoded data.
    This lets callers send a local image/audio file inline without exposing a
    public URL.
    """
    header, sep, data = uri[len("data:"):].partition(",")
    if not sep:
        raise ValueError("Malformed data URI")
    is_base64 = header.endswith(";base64")
    mediatype = header[: -len(";base64")] if is_base64 else header
    mime = mediatype.split(";", 1)[0].strip() or None
    content = base64.b64decode(data) if is_base64 else unquote_to_bytes(data)
    return content, mime


async def fetch_bytes(url: str) -> tuple[bytes, str | None]:
    """Resolve ``url`` to ``(content, mime_type)``.

    Handles inline ``data:`` URIs directly and otherwise downloads over HTTP(S).
    ``mime_type`` comes from the data URI or the response ``Content-Type`` header
    when present. Raises ``ValueError``/``httpx.HTTPError`` on failure; callers
    translate that into a provider error.
    """
    if url.startswith("data:"):
        return _parse_data_uri(url)

    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type")
        if mime:
            mime = mime.split(";", 1)[0].strip()
        return resp.content, mime
