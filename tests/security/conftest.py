"""Shared helpers for SSRF / security tests."""

from __future__ import annotations

import socket
from functools import partial
from typing import TYPE_CHECKING

import pytest

from crossmem.sources import _http as _http_module
from crossmem.sources import github as _github_module
from crossmem.sources import web as _web_module
from crossmem.sources.adapters import context7 as _context7_module

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def _disable_dns_pinning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable DNS-rebind URL rewrite for URL-matched ``httpx_mock`` tests.

    See ``tests/sources/conftest.py`` for the rationale — pinning rewrites
    the request URL to the resolved IP literal which would defeat the
    URL-based pytest-httpx matcher used by these tests. The dedicated
    rebind regression suite in ``tests/sources/test_dns_pinning.py``
    re-enables pinning by calling ``safe_get`` with ``pin_dns=True``
    directly.
    """
    no_pin = partial(_http_module.safe_get, pin_dns=False)
    for module in (
        _web_module,
        _github_module,
        _context7_module,
    ):
        monkeypatch.setattr(module, "safe_get", no_pin)


@pytest.fixture
def patch_dns(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Return a callable that forces ``socket.getaddrinfo`` to resolve to ``ip``.

    Usage:
        def test_x(patch_dns):
            patch_dns("127.0.0.1")
            ...

    The fixture itself does not patch anything until the returned callable is
    invoked, so a single test can re-patch as needed (e.g., for redirect tests
    that need different resolutions for different hops).
    """

    def _apply(ip: str) -> None:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET

        def fake_getaddrinfo(host, *_args, **_kwargs):
            return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    return _apply
