"""Empty tool-call response fault-injection scenario (task 27.3c).

Posts a request to ``${CROSSMEM_E2E_MOCK_URL}/v1/messages`` and
expects a valid JSON response with **no** tool-use blocks — the
upstream client must treat this as "the LLM didn't ask for any
tools" and not crash. Returns ``0`` when the response decodes to
JSON with an empty ``content`` list, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import urllib.request


def run() -> int:
    base_url = os.environ.get("CROSSMEM_E2E_MOCK_URL", "").strip()
    if not base_url:
        return 10

    request_body = json.dumps(
        {
            "model": "mock",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "please trigger empty-tool-calls fixture"}
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - loopback only, test harness
        base_url.rstrip("/") + "/v1/messages",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
        body = resp.read()
    parsed = json.loads(body.decode("utf-8"))
    content = parsed.get("content")
    if isinstance(content, list) and content == []:
        return 0
    return 1


__all__ = ["run"]
