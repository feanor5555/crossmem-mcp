"""Broken-JSON fault-injection scenario (task 27.3c).

Posts a request to ``${CROSSMEM_E2E_MOCK_URL}/v1/messages`` and
expects the response body to be **not** valid JSON. Returns ``0``
when JSON decoding fails (the upstream client must surface this as
an error), non-zero when the body parses cleanly (regression).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def run() -> int:
    base_url = os.environ.get("CROSSMEM_E2E_MOCK_URL", "").strip()
    if not base_url:
        return 10  # misconfiguration — runner must export the URL

    request_body = json.dumps(
        {
            "model": "mock",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "please trigger broken-json fixture"}
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - loopback only, test harness
        base_url.rstrip("/") + "/v1/messages",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
            body = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() or b""
    try:
        json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 0  # expected: malformed JSON observed
    return 1  # unexpected: server returned valid JSON


__all__ = ["run"]
