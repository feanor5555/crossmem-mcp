"""HTTP-5xx fault-injection scenario (task 27.3c).

Posts a request to ``${CROSSMEM_E2E_MOCK_URL}/v1/messages`` and
expects a 5xx response. Returns ``0`` when the server answers with
any status in ``[500, 600)``, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import urllib.error
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
                {"role": "user", "content": "please trigger http-5xx fixture"}
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
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    if 500 <= status < 600:
        return 0
    return 1


__all__ = ["run"]
