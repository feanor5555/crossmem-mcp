"""Unit tests for the centralised SSRF guard helpers in ``sources/_http.py``.

These cover the ``safe_get``-specific edge cases (redirect cap, redirect
without Location, redirect-to-internal) that are not exercised by the
per-source SSRF parametrisation.
"""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from crossmem.sources._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpResponseTooLargeError,
    SSRFError,
    _resolve_host,
    safe_get,
    validate_url,
)


def _patch_dns(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# ---------- validate_url ----------------------------------------------------


def test_validate_url_returns_scheme_and_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dns(monkeypatch, "8.8.8.8")
    scheme, host = validate_url("https://example.com/path")
    assert scheme == "https"
    assert host == "example.com"


def test_validate_url_literal_public_ip_ok() -> None:
    scheme, host = validate_url("http://8.8.8.8/")
    assert (scheme, host) == ("http", "8.8.8.8")


def test_validate_url_blocked_scheme() -> None:
    with pytest.raises(SSRFError):
        validate_url("ftp://example.com")


def test_validate_url_missing_host() -> None:
    with pytest.raises(SSRFError):
        validate_url("https:///path")


def test_validate_url_literal_blocked_ip() -> None:
    with pytest.raises(SSRFError):
        validate_url("http://127.0.0.1/")


# ---------- _resolve_host ---------------------------------------------------


def test_resolve_host_skips_invalid_sockaddrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``getaddrinfo`` entries with garbage IPs are silently skipped."""

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not-an-ip", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    addrs = _resolve_host("example.com")
    assert len(addrs) == 1
    assert str(addrs[0]) == "8.8.8.8"


# ---------- safe_get: redirect handling -------------------------------------


def test_safe_get_blocks_redirect_to_internal_ip(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://hop.example.com/start",
        status_code=302,
        headers={"Location": "http://127.0.0.1/admin"},
    )
    with httpx.Client(follow_redirects=False) as client, pytest.raises(SSRFError):
        safe_get(client, "https://hop.example.com/start", pin_dns=False)


def test_safe_get_redirect_without_location_returns_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """A 302 with no Location header is returned verbatim, not chased."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://hop.example.com/x",
        status_code=302,
        headers={},
    )
    with httpx.Client(follow_redirects=False) as client:
        resp = safe_get(client, "https://hop.example.com/x", pin_dns=False)
    assert resp.status_code == 302


def test_safe_get_too_many_redirects(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://loop.example.com/x",
        status_code=302,
        headers={"Location": "/x"},
        is_reusable=True,
    )
    with httpx.Client(follow_redirects=False) as client, pytest.raises(httpx.HTTPError):
        safe_get(client, "https://loop.example.com/x", max_redirects=2, pin_dns=False)


def test_safe_get_follows_safe_redirect_chain(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://safe.example.com/a",
        status_code=302,
        headers={"Location": "/b"},
    )
    httpx_mock.add_response(
        url="https://safe.example.com/b",
        status_code=200,
        text="final body",
    )
    with httpx.Client(follow_redirects=False) as client:
        resp = safe_get(client, "https://safe.example.com/a", pin_dns=False)
    assert resp.status_code == 200
    assert resp.text == "final body"


def test_safe_get_passes_params_and_headers(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Initial params and headers reach the wire on the first hop."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url=httpx.URL("https://api.example.com/search", params={"q": "hello"}),
        status_code=200,
        json={"ok": True},
    )
    with httpx.Client(follow_redirects=False) as client:
        resp = safe_get(
            client,
            "https://api.example.com/search",
            params={"q": "hello"},
            headers={"X-Test": "yes"},
            pin_dns=False,
        )
    assert resp.status_code == 200
    req = httpx_mock.get_request()
    assert req.headers.get("X-Test") == "yes"
    assert req.url.params["q"] == "hello"


def test_safe_get_rejects_initial_blocked_url() -> None:
    with httpx.Client(follow_redirects=False) as client, pytest.raises(SSRFError):
        safe_get(client, "http://127.0.0.1/")


# ---------- safe_get: response size limit -----------------------------------


def test_safe_get_default_max_response_bytes_is_reasonable() -> None:
    """The default cap should be in the megabyte range — large enough for real
    docs (READMEs, search payloads) but small enough to prevent a hostile
    server from filling memory with a multi-gigabyte response."""
    assert 1 * 1024 * 1024 <= DEFAULT_MAX_RESPONSE_BYTES <= 64 * 1024 * 1024


def test_safe_get_rejects_oversize_by_content_length(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """A response advertising a Content-Length above the cap must be rejected
    with a clear error before the body is consumed."""
    _patch_dns(monkeypatch, "8.8.8.8")
    huge = DEFAULT_MAX_RESPONSE_BYTES + 1
    httpx_mock.add_response(
        url="https://big.example.com/file",
        status_code=200,
        headers={"Content-Length": str(huge)},
        content=b"x" * huge,
    )
    with (
        httpx.Client(follow_redirects=False) as client,
        pytest.raises(HttpResponseTooLargeError) as exc_info,
    ):
        safe_get(client, "https://big.example.com/file", pin_dns=False)
    msg = str(exc_info.value).lower()
    assert "size" in msg or "large" in msg or "bytes" in msg
    assert str(huge) in str(exc_info.value) or str(DEFAULT_MAX_RESPONSE_BYTES) in str(
        exc_info.value
    )


def test_safe_get_rejects_oversize_body_when_content_length_lies(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """A server that lies about Content-Length (claims small, sends huge)
    must still be rejected — the actual byte count is the source of truth."""
    _patch_dns(monkeypatch, "8.8.8.8")
    huge = DEFAULT_MAX_RESPONSE_BYTES + 1024
    httpx_mock.add_response(
        url="https://sneaky.example.com/file",
        status_code=200,
        headers={"Content-Length": "10"},  # lie
        content=b"x" * huge,
    )
    with (
        httpx.Client(follow_redirects=False) as client,
        pytest.raises(HttpResponseTooLargeError) as exc_info,
    ):
        safe_get(client, "https://sneaky.example.com/file", pin_dns=False)
    # Message must reference the actual byte count, not the advertised one.
    assert str(huge) in str(exc_info.value)


def test_safe_get_ignores_malformed_content_length(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """A non-numeric Content-Length must not crash the guard; the actual
    body size still determines pass/fail."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://weird.example.com/file",
        status_code=200,
        headers={"Content-Length": "not-a-number"},
        content=b"ok",
    )
    with httpx.Client(follow_redirects=False) as client:
        resp = safe_get(client, "https://weird.example.com/file", pin_dns=False)
    assert resp.status_code == 200
    assert resp.content == b"ok"


def test_safe_get_allows_response_at_or_below_cap(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """A response exactly at the cap must still pass through."""
    _patch_dns(monkeypatch, "8.8.8.8")
    body = b"x" * DEFAULT_MAX_RESPONSE_BYTES
    httpx_mock.add_response(
        url="https://ok.example.com/file",
        status_code=200,
        content=body,
    )
    with httpx.Client(follow_redirects=False) as client:
        resp = safe_get(client, "https://ok.example.com/file", pin_dns=False)
    assert resp.status_code == 200
    assert len(resp.content) == DEFAULT_MAX_RESPONSE_BYTES


def test_safe_get_custom_max_response_bytes(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Callers can lower the cap per request (e.g. for cheap probes)."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://small.example.com/file",
        status_code=200,
        content=b"x" * 1024,
    )
    with (
        httpx.Client(follow_redirects=False) as client,
        pytest.raises(HttpResponseTooLargeError),
    ):
        safe_get(
            client,
            "https://small.example.com/file",
            pin_dns=False,
            max_response_bytes=512,
        )
