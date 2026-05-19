"""Tests for the web source: SSRF guard, robots.txt, and HTML extraction."""

from __future__ import annotations

import socket

import httpx
import pytest
from bs4 import BeautifulSoup

from crossmem.sources._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpResponseTooLargeError,
)
from crossmem.sources.web import (
    RobotsDisallowedError,
    SSRFError,
    WebSource,
    _extract_title,
    _is_blocked_ip,
    _validate_url,
)

# ---------- helpers ---------------------------------------------------------


def _patch_dns(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    """Force ``socket.getaddrinfo`` to resolve every host to ``ip``."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def fake_getaddrinfo(host, *_args, **_kwargs):
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# ---------- can_handle ------------------------------------------------------


def test_can_handle_http_https():
    src = WebSource()
    assert src.can_handle("https://example.com") is True
    assert src.can_handle("http://example.com/page") is True


def test_can_handle_rejects_other_schemes():
    src = WebSource()
    assert src.can_handle("file:///etc/passwd") is False
    assert src.can_handle("gopher://example.com") is False
    assert src.can_handle("ftp://example.com") is False
    assert src.can_handle("data:text/plain,abc") is False
    assert src.can_handle("javascript:alert(1)") is False


def test_can_handle_requires_host():
    src = WebSource()
    assert src.can_handle("https://") is False


def test_can_handle_invalid_url():
    src = WebSource()
    # bracketed IPv6 with garbage triggers urlparse ValueError on some versions
    assert src.can_handle("http://[bad") is False


# ---------- SSRF: scheme-based ---------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/_",
        "ftp://example.com/file",
        "data:text/plain,abc",
        "javascript:alert(1)",
    ],
)
def test_fetch_rejects_disallowed_schemes(url):
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(url)


# ---------- SSRF: IP-based --------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # RFC-1918
        "172.16.0.1",  # RFC-1918
        "192.168.1.1",  # RFC-1918
        "169.254.169.254",  # AWS metadata / link-local
        "0.0.0.0",  # unspecified
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 unique-local
    ],
)
def test_fetch_blocks_internal_dns_resolutions(monkeypatch, ip):
    _patch_dns(monkeypatch, ip)
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://attacker.example.com/")


def test_fetch_rejects_literal_blocked_ip():
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://127.0.0.1/secret")


def test_validate_url_no_host():
    with pytest.raises(SSRFError):
        _validate_url("https:///path")


def test_validate_url_unresolvable(monkeypatch):
    def fail(*_a, **_kw):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(SSRFError):
        _validate_url("https://nowhere.invalid")


def test_validate_url_no_addresses(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(SSRFError):
        _validate_url("https://void.example.com")


def test_is_blocked_ip_helper_passes_public():
    import ipaddress

    assert _is_blocked_ip(ipaddress.ip_address("8.8.8.8")) is False
    assert _is_blocked_ip(ipaddress.ip_address("2606:4700:4700::1111")) is False


# ---------- Redirect SSRF ---------------------------------------------------


def test_redirect_to_internal_ip_blocked(monkeypatch, httpx_mock):
    """A redirect Location pointing at an internal host must be re-validated."""
    # Public DNS for the first host, but a subsequent literal 127.0.0.1
    # must be rejected by the literal-IP guard regardless of DNS patching.
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://evil.example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://evil.example.com/",
        status_code=302,
        headers={"Location": "http://127.0.0.1/admin"},
    )
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://evil.example.com/")


# ---------- robots.txt ------------------------------------------------------


def test_robots_disallow_blocks_fetch(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt",
        status_code=200,
        text="User-agent: *\nDisallow: /private\n",
    )
    src = WebSource()
    with pytest.raises(RobotsDisallowedError):
        src.fetch("https://example.com/private/secret")


def test_robots_allow_lets_fetch_proceed(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt",
        status_code=200,
        text="User-agent: *\nAllow: /\n",
    )
    httpx_mock.add_response(
        url="https://example.com/page",
        status_code=200,
        text="<html><head><title>Hi</title></head><body>Body text</body></html>",
    )
    src = WebSource()
    docs = src.fetch("https://example.com/page")
    assert len(docs) == 1
    assert "Body text" in docs[0].content


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_robots_5xx_disallows_per_rfc_9309(monkeypatch, httpx_mock, status):
    """RFC 9309 sec. 2.3.1.3: 5xx on robots.txt MUST be treated as complete disallow."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=status, text="boom"
    )
    src = WebSource()
    with pytest.raises(RobotsDisallowedError):
        src.fetch("https://example.com/")


def test_robots_4xx_other_than_404_defaults_to_allow(monkeypatch, httpx_mock):
    """RFC 9309 sec. 2.3.1.4: 4xx (Unavailable) means no rules apply -> allow."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=403, text="forbidden"
    )
    httpx_mock.add_response(
        url="https://example.com/",
        status_code=200,
        text="<html><body><main>Main only</main></body></html>",
    )
    src = WebSource()
    docs = src.fetch("https://example.com/")
    assert docs and "Main only" in docs[0].content


def test_robots_transport_error_defaults_to_allow(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_exception(
        httpx.ConnectError("offline"), url="https://example.com/robots.txt"
    )
    httpx_mock.add_response(
        url="https://example.com/",
        status_code=200,
        text="<html><body>Hello world</body></html>",
    )
    src = WebSource()
    docs = src.fetch("https://example.com/")
    assert docs and "Hello world" in docs[0].content


def test_robots_dns_rebind_to_internal_ip_propagates_ssrf(monkeypatch, httpx_mock):
    """A DNS rebind between the initial host validation and the robots.txt
    fetch must surface as :class:`SSRFError`, not be silently swallowed.

    Scenario: the first ``getaddrinfo`` (in :func:`validate_url` from
    ``WebSource.fetch``) returns a public IP, so the URL passes the guard.
    The next call (for ``robots.txt`` inside :func:`_check_robots`) flips
    to an RFC-1918 address — a classic rebind. The previous behaviour
    caught ``SSRFError`` alongside ``httpx.HTTPError`` and defaulted to
    "allow", which let the fetch proceed past robots and made another
    network call against a now-internal target. The fix narrows the
    except clause to ``httpx.HTTPError`` only.

    To pin the failure to the robots-path specifically (not the later
    page fetch that would also re-resolve), this test caps
    ``getaddrinfo`` at two calls and raises on a third. If the bug
    regresses, ``safe_get`` for the page URL would trigger the third
    lookup and the test would surface a :class:`RuntimeError` instead of
    the expected :class:`SSRFError`.
    """
    calls: list[str] = []

    def fake_getaddrinfo(host, *_args, **_kwargs):
        calls.append(host)
        n = len(calls)
        if n == 1:
            ip = "8.8.8.8"
        elif n == 2:
            ip = "10.0.0.5"
        else:
            raise RuntimeError(
                f"unexpected third getaddrinfo (host={host!r}); "
                "robots-path SSRFError was swallowed"
            )
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    # If the bug regresses, _check_robots would swallow the SSRFError and
    # the code would carry on, eventually attempting the page request.
    # The mock is registered as optional so its absence does not mask the
    # regression — the third getaddrinfo above is what makes the test
    # surface a RuntimeError in that case.
    httpx_mock.add_response(
        url="https://example.com/page",
        status_code=200,
        text="<html><body>should not be reached</body></html>",
        is_optional=True,
    )
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://example.com/page")
    assert len(calls) == 2


# ---------- Happy path / extraction -----------------------------------------


HTML_PAGE = """
<html>
  <head><title>Example Page</title></head>
  <body>
    <header>Site Header</header>
    <nav>nav nav nav</nav>
    <aside>side</aside>
    <main>
      <h1>Hello</h1>
      <p>Important <b>article</b> content.</p>
      <script>alert('x')</script>
      <style>.x{}</style>
    </main>
    <footer>copyright</footer>
  </body>
</html>
"""


def test_fetch_extracts_main_content_and_strips_chrome(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://example.com/article", status_code=200, text=HTML_PAGE
    )
    src = WebSource()
    docs = src.fetch("https://example.com/article")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.title == "Example Page"
    assert doc.metadata.source_url == "https://example.com/article"
    assert doc.metadata.source_type == "web"
    assert doc.metadata.content_hash
    assert doc.id and len(doc.id) == 32
    assert "Important article content." in doc.content
    # Stripped:
    assert "nav nav nav" not in doc.content
    assert "Site Header" not in doc.content
    assert "copyright" not in doc.content
    assert "alert" not in doc.content


def test_extract_title_concatenates_children():
    """Multi-child ``<title>`` (where ``.string`` is None) yields joined text."""
    soup = BeautifulSoup(
        "<html><head><title></title></head><body></body></html>", "html.parser"
    )
    soup.title.append("Foo ")
    span = soup.new_tag("span")
    span.string = "bar"
    soup.title.append(span)
    soup.title.append(" baz")
    assert soup.title.string is None  # precondition: bug-trigger shape
    assert _extract_title(soup) == "Foo bar baz"


def test_extract_title_falls_back_to_h1_when_title_empty():
    soup = BeautifulSoup(
        "<html><head><title>   </title></head><body><h1>Headline</h1></body></html>",
        "html.parser",
    )
    assert _extract_title(soup) == "Headline"


def test_extract_title_returns_empty_when_nothing_available():
    soup = BeautifulSoup("<html><body><p>no titles</p></body></html>", "html.parser")
    assert _extract_title(soup) == ""


def test_fetch_uses_h1_when_no_title(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    html = "<html><body><h1>Headline</h1><article>Body here.</article></body></html>"
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(url="https://example.com/", status_code=200, text=html)
    src = WebSource()
    docs = src.fetch("https://example.com/")
    assert docs[0].metadata.title == "Headline"
    assert "Body here." in docs[0].content


def test_fetch_returns_empty_for_empty_body(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://example.com/",
        status_code=200,
        text="<html><body></body></html>",
    )
    src = WebSource()
    docs = src.fetch("https://example.com/")
    assert docs == []


def test_fetch_raises_on_4xx(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://example.com/missing", status_code=404, text="not found"
    )
    src = WebSource()
    with pytest.raises(httpx.HTTPStatusError):
        src.fetch("https://example.com/missing")


def test_fetch_follows_safe_redirects(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://example.com/old",
        status_code=302,
        headers={"Location": "/new"},
    )
    httpx_mock.add_response(
        url="https://example.com/new",
        status_code=200,
        text="<html><body>Redirected ok</body></html>",
    )
    src = WebSource()
    docs = src.fetch("https://example.com/old")
    assert docs and "Redirected ok" in docs[0].content


def test_fetch_too_many_redirects(monkeypatch, httpx_mock):
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://example.com/loop",
        status_code=302,
        headers={"Location": "/loop"},
        is_reusable=True,
    )
    src = WebSource()
    with pytest.raises(httpx.HTTPError):
        src.fetch("https://example.com/loop")


def test_name():
    assert WebSource().name() == "web"


# ---------- response size limit --------------------------------------------


def test_fetch_rejects_oversize_response(monkeypatch, httpx_mock):
    """A page body above the cap must abort with HttpResponseTooLargeError
    and return no documents — nothing must reach the store."""
    _patch_dns(monkeypatch, "8.8.8.8")
    httpx_mock.add_response(
        url="https://huge.example.com/robots.txt", status_code=404, text=""
    )
    huge = DEFAULT_MAX_RESPONSE_BYTES + 1
    httpx_mock.add_response(
        url="https://huge.example.com/page",
        status_code=200,
        headers={"Content-Length": str(huge)},
        content=b"x" * huge,
    )
    src = WebSource()
    with pytest.raises(HttpResponseTooLargeError):
        src.fetch("https://huge.example.com/page")
