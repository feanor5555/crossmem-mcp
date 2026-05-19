"""Slow-response / timeout fault-injection scenario (task 27.3c).

Posts a request to ``${CROSSMEM_E2E_MOCK_URL}/v1/messages`` with a
client-side timeout shorter than the fixture's ``delay_s``. Returns
``0`` when the client raises a timeout error, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

#: Client-side timeout (seconds). Must be tighter than the slowest
#: timeout fixture (currently ``delay_s = 0.4``) so the request
#: actually hits a timeout on every run. Keep it short — CI doesn't
#: need to wait around for a deliberately slow mock.
_CLIENT_TIMEOUT_S = 0.05


def run() -> int:
    base_url = os.environ.get("CROSSMEM_E2E_MOCK_URL", "").strip()
    if not base_url:
        return 10

    request_body = json.dumps(
        {
            "model": "mock",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "please trigger timeout fixture"}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - loopback only, test harness
        base_url.rstrip("/") + "/v1/messages",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_CLIENT_TIMEOUT_S) as resp:  # noqa: S310
            resp.read()
    except (urllib.error.URLError, TimeoutError):
        # ``socket.timeout`` is now an alias for the builtin
        # ``TimeoutError`` since Python 3.10 — the URLError branch
        # catches the timeout flavour urllib raises in practice.
        return 0  # expected: client timed out before the slow mock replied
    return 1  # unexpected: server replied within the (tiny) budget


__all__ = ["run"]
