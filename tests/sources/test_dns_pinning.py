"""DNS-rebind regression tests for the shared HTTP layer.

These tests exercise the IP-pinning rewrite added to
:func:`crossmem.sources._http.safe_get`. The parent ``conftest.py``
autouse fixture wraps the adapters' ``safe_get`` import with
``pin_dns=False`` so URL-based ``httpx_mock`` mocks keep working for the
rest of the source suite. This module imports ``safe_get`` directly from
``crossmem.sources._http`` (not via an adapter) and exercises the default
``pin_dns=True`` behaviour, verifying:

* a rebind attempt — where ``socket.getaddrinfo`` returns a public IP on
  the first call and a loopback IP on the second — does not redirect
  the connection at httpx's connect-time lookup, because ``safe_get``
  pins to the *first* resolution and rewrites the URL to that IP;
* hosts that have only a single IP keep working normally and produce
  the expected ``Host`` header + TLS SNI extension;
* URLs that already use an IP literal are not rewritten (no spurious
  ``Host`` header / SNI extension is injected);
* the rewrite uses bracket syntax for IPv6 pinned addresses.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

import httpx

from crossmem.sources._http import (
    _pinned_request_args,
    _rewrite_to_pinned_ip,
    safe_get,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


# ---------- DNS patch helpers ----------------------------------------------


def _single_ip(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _rebind_sequence(
    monkeypatch: pytest.MonkeyPatch, ips: list[str]
) -> Callable[[], int]:
    """Make ``socket.getaddrinfo`` return the IPs from ``ips`` in order.

    Returns a callable that yields the current invocation count. Once
    ``ips`` is exhausted the last entry is repeated indefinitely — that
    matches a real attacker whose authoritative DNS keeps flipping
    answers after the validator passed.
    """
    state = {"calls": 0}

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        idx = min(state["calls"], len(ips) - 1)
        state["calls"] += 1
        ip = ips[idx]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return lambda: state["calls"]


# ---------- _rewrite_to_pinned_ip unit tests --------------------------------


def test_rewrite_ipv4_basic() -> None:
    import ipaddress

    out = _rewrite_to_pinned_ip(
        "https://example.com/path?q=1#frag",
        ipaddress.IPv4Address("8.8.8.8"),
    )
    assert out == "https://8.8.8.8/path?q=1#frag"


def test_rewrite_ipv4_preserves_port_and_userinfo() -> None:
    import ipaddress

    out = _rewrite_to_pinned_ip(
        "http://alice:secret@example.com:8443/api?x=1",
        ipaddress.IPv4Address("1.2.3.4"),
    )
    assert out == "http://alice:secret@1.2.3.4:8443/api?x=1"


def test_rewrite_ipv6_uses_brackets() -> None:
    import ipaddress

    out = _rewrite_to_pinned_ip(
        "https://example.com:443/x",
        ipaddress.IPv6Address("2606:4700::1111"),
    )
    assert out == "https://[2606:4700::1111]:443/x"


def test_rewrite_userinfo_percent_encodes_special_chars() -> None:
    """Username/password with ``@``, ``:`` and ``/`` must be re-quoted.

    A literal ``:`` inside the password (everything after the first ``:``
    of the userinfo block) reaches ``parsed.password`` unmodified, while
    ``@`` and ``/`` are typically delivered percent-encoded. In either
    case, reusing the raw substring verbatim in the rewritten netloc
    risks an ambiguous authority (``user:a:b@1.2.3.4`` has two ``:``
    separators). Percent-encoding every userinfo subcomponent via
    ``quote(s, safe="")`` is the spec — special chars become ``%40`` /
    ``%3A`` / ``%2F`` and the rebuilt netloc is unambiguous.
    """
    import ipaddress

    # ``:`` literal in password; ``@`` and ``/`` already percent-encoded.
    out = _rewrite_to_pinned_ip(
        "http://alice:pw%40ss:more%2Fpath@example.com:8443/api",
        ipaddress.IPv4Address("1.2.3.4"),
    )
    # quote(s, safe="") re-encodes the existing %-sequences (% -> %25),
    # and encodes the literal ``:`` (which urlparse left intact in
    # parsed.password) to %3A. The resulting netloc has exactly one
    # ``:`` between username and password and exactly one ``@`` before
    # the host, so httpx will parse it correctly.
    assert out == "http://alice:pw%2540ss%3Amore%252Fpath@1.2.3.4:8443/api"


# ---------- _pinned_request_args unit tests ---------------------------------


def test_pinned_request_rewrites_https_with_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _single_ip(monkeypatch, "8.8.8.8")
    pinned_url, headers, extensions = _pinned_request_args(
        "https://example.com/page", {"User-Agent": "crossmem-test"}
    )
    assert pinned_url == "https://8.8.8.8/page"
    assert headers["Host"] == "example.com"
    assert headers["User-Agent"] == "crossmem-test"
    assert extensions == {"sni_hostname": "example.com"}


def test_pinned_request_http_omits_sni(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain HTTP must not set a TLS SNI extension."""
    _single_ip(monkeypatch, "8.8.8.8")
    pinned_url, headers, extensions = _pinned_request_args("http://example.com/x", None)
    assert pinned_url == "http://8.8.8.8/x"
    assert headers["Host"] == "example.com"
    assert extensions == {}


def test_pinned_request_literal_ip_url_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URLs that already use a public IP literal are not rewritten.

    The SSRF guard runs (literal-IP block check) but no ``Host`` header or
    SNI extension is forced — there is nothing to pin against, and we
    must not invent a hostname for a URL the caller wrote with an IP.
    """
    # _single_ip not strictly needed (no DNS happens for IP literals)
    # but applied for consistency with the other tests.
    _single_ip(monkeypatch, "8.8.8.8")
    pinned_url, headers, extensions = _pinned_request_args("https://8.8.8.8/x", None)
    assert pinned_url == "https://8.8.8.8/x"
    assert "Host" not in headers
    assert extensions == {}


def test_pinned_request_disabled_via_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``pin_dns=False`` the URL/headers/extensions pass through."""
    _single_ip(monkeypatch, "8.8.8.8")
    pinned_url, headers, extensions = _pinned_request_args(
        "https://example.com/x", {"X-Test": "1"}, pin_dns=False
    )
    assert pinned_url == "https://example.com/x"
    assert headers == {"X-Test": "1"}
    assert extensions == {}


# ---------- End-to-end rebind test through safe_get -------------------------


def test_safe_get_pins_to_first_resolved_ip_against_rebind(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """The rebind regression test mandated by TODO 20.7.

    Scenario: ``socket.getaddrinfo`` is rigged to flip its answer after the
    SSRF guard has cleared the first lookup — first call returns a public
    IP (``1.2.3.4``), every subsequent call returns ``127.0.0.1`` (the
    classic rebind payload that would expose internal services if httpx
    re-resolved at connect time).

    With pinning the request URL handed to ``client.get`` is rewritten to
    ``http://1.2.3.4/...`` *before* the second lookup is even relevant —
    httpx never resolves the hostname a second time because the URL host
    is now a literal IP. We assert this via the URL that pytest-httpx
    receives on the wire.
    """
    call_count = _rebind_sequence(monkeypatch, ["1.2.3.4", "127.0.0.1"])
    # The pinning rewrite makes the outgoing URL use the pinned IP.
    httpx_mock.add_response(url="http://1.2.3.4/secret", status_code=200, text="ok")

    with httpx.Client(follow_redirects=False) as client:
        response = safe_get(client, "http://victim.example.com/secret")

    assert response.status_code == 200
    assert response.text == "ok"

    # Exactly one DNS lookup happened — the validation lookup. httpx itself
    # never re-resolved because the URL it was handed already contained the
    # IP literal. The rebind answer (127.0.0.1) was therefore never used.
    assert call_count() == 1

    # Wire-level assertions: original hostname preserved in Host header,
    # request URL points at the pinned IP.
    request = httpx_mock.get_request()
    assert request.url.host == "1.2.3.4"
    assert request.headers.get("Host") == "victim.example.com"


def test_safe_get_preserves_host_header_and_sni_for_https(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Single-IP regression: pinning must not break normal HTTPS requests.

    Asserts the rewritten URL uses the pinned IP, the Host header carries
    the original hostname so the remote vhost lookup still works, and the
    TLS SNI extension is set so cert validation against the hostname
    keeps working.
    """
    _single_ip(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(url="https://8.8.8.8/api", status_code=200, json={"ok": 1})

    with httpx.Client(follow_redirects=False) as client:
        response = safe_get(client, "https://api.example.com/api")

    assert response.status_code == 200
    request = httpx_mock.get_request()
    assert request.url.host == "8.8.8.8"
    assert request.headers.get("Host") == "api.example.com"
    assert request.extensions.get("sni_hostname") == "api.example.com"


def test_safe_get_literal_ip_url_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Calling safe_get with a public IP literal already passes through.

    No Host header forced (we have no hostname to put there) and no SNI
    extension set. The mock matches the URL as-is.
    """
    httpx_mock.add_response(url="https://8.8.8.8/health", status_code=200, text="up")

    with httpx.Client(follow_redirects=False) as client:
        response = safe_get(client, "https://8.8.8.8/health")

    assert response.status_code == 200
    request = httpx_mock.get_request()
    assert request.url.host == "8.8.8.8"
    # httpx auto-inserts a Host header for HTTP/1.1; if present it must
    # reflect the IP literal itself, NOT a fabricated hostname.
    host_header = request.headers.get("Host")
    assert host_header in (None, "8.8.8.8", "8.8.8.8:443")
    assert "sni_hostname" not in request.extensions


def test_safe_get_redirect_repins_per_hop(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Each redirect hop must trigger a fresh resolve+pin cycle.

    Two-hop chain: first hop resolves to ``1.2.3.4`` (single-call DNS
    fixture per hop), second hop on a different hostname resolves to
    ``5.6.7.8``. Both hops must be rewritten correctly.
    """
    by_host = {"hop-one.example.com": "1.2.3.4", "hop-two.example.com": "5.6.7.8"}

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        ip = by_host.get(host, "8.8.8.8")
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    httpx_mock.add_response(
        url="https://1.2.3.4/start",
        status_code=302,
        headers={"Location": "https://hop-two.example.com/finish"},
    )
    httpx_mock.add_response(url="https://5.6.7.8/finish", status_code=200, text="done")

    with httpx.Client(follow_redirects=False) as client:
        response = safe_get(client, "https://hop-one.example.com/start")

    assert response.status_code == 200
    assert response.text == "done"
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    assert requests[0].url.host == "1.2.3.4"
    assert requests[0].headers.get("Host") == "hop-one.example.com"
    assert requests[1].url.host == "5.6.7.8"
    assert requests[1].headers.get("Host") == "hop-two.example.com"


def test_safe_get_pin_dns_false_passes_url_through(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: Any
) -> None:
    """Explicit ``pin_dns=False`` skips the IP rewrite (test-suite contract).

    The SSRF guard still runs — the host must resolve and pass the
    private/loopback checks — but the outgoing URL keeps the original
    hostname so URL-based mocks keep matching.
    """
    _single_ip(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/passthrough", status_code=200, text="ok"
    )

    with httpx.Client(follow_redirects=False) as client:
        response = safe_get(client, "https://example.com/passthrough", pin_dns=False)

    assert response.status_code == 200
    request = httpx_mock.get_request()
    assert request.url.host == "example.com"
    assert "sni_hostname" not in request.extensions
