"""Source-adapter test fixtures.

All source adapters now route HTTP through the SSRF guard in
``crossmem.sources._http`` which performs a real ``socket.getaddrinfo``
lookup before dispatching. CI runners and offline test environments
have no DNS for the fictional hosts used by ``pytest-httpx``, so the
guard would reject every call.

The ``_safe_dns`` autouse fixture redirects all DNS lookups to a known
*public* IP (``8.8.8.8``) so the guard sees a safe resolution and lets
``pytest-httpx`` intercept the actual request. Tests that need a
specific IP (e.g. to assert the SSRF guard rejects an internal address)
override the patch in their own scope before calling ``fetch``.

The same fixture also wraps :func:`crossmem.sources._http.safe_get` in
each importing adapter module so that production calls run with
``pin_dns=False`` during the test suite. ``pytest-httpx`` matches outgoing
requests by URL; with pinning on every test mock would need to register
against the pinned IP literal (``https://8.8.8.8/...``) instead of the
original hostname. The dedicated rebind regression suite in
``test_dns_pinning.py`` calls ``safe_get`` directly with ``pin_dns=True``.
"""

from __future__ import annotations

import socket
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

from crossmem.sources import _http as _http_module
from crossmem.sources import github as _github_module
from crossmem.sources import web as _web_module
from crossmem.sources.adapters import context7 as _context7_module

if TYPE_CHECKING:
    from collections.abc import Iterable


def _patch_safe_get_no_pin(
    monkeypatch: pytest.MonkeyPatch, modules: Iterable[Any]
) -> None:
    """Force ``safe_get`` in each module's namespace to run with ``pin_dns=False``.

    Adapters import ``safe_get`` at module load (``from ... import safe_get``);
    each ends up with its own binding. Patching only the source module would
    not affect callers, so every importing module is patched explicitly.
    """
    no_pin = partial(_http_module.safe_get, pin_dns=False)
    for module in modules:
        monkeypatch.setattr(module, "safe_get", no_pin)


@pytest.fixture(autouse=True)
def _safe_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    # Keep URL-based pytest-httpx mocks matching the original hostname.
    # See module docstring.
    _patch_safe_get_no_pin(
        monkeypatch,
        (_web_module, _github_module, _context7_module),
    )
