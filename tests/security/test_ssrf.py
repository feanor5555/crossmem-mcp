"""Exhaustive SSRF protection tests for the web source.

Covers every vector listed in the project security requirements:
- Literal-IP blocks (loopback, link-local, RFC-1918, IPv6 loopback / ULA /
  link-local, unspecified, broadcast)
- Scheme blocks (file, gopher, ftp, data, javascript, mixed-case)
- DNS-based blocks (host resolves to any blocked IP family)
- Redirect-target blocks (302 Location pointing at internal IPs)
- Edge cases (IPv4-mapped IPv6, decimal IP encoding, trailing-dot hostnames)

These tests are *purely defensive* — they assert that ``WebSource.fetch``
refuses to make any request when the target is internal/blocked. They do
NOT modify production code; if a real bypass is found, the test is marked
``xfail`` with a TODO instead so the gap is visible.
"""

from __future__ import annotations

import socket

import pytest

from crossmem.sources.web import (
    SSRFError,
    WebSource,
    _validate_url,
)

# ---------------------------------------------------------------------------
# 1. Literal-IP blocks (no DNS, the URL itself contains the address)
# ---------------------------------------------------------------------------

LITERAL_BLOCKED_IPS: list[tuple[str, str]] = [
    # AWS metadata / IPv4 link-local
    ("169.254.169.254", "AWS metadata / IPv4 link-local"),
    ("169.254.0.1", "IPv4 link-local lower boundary"),
    ("169.254.255.254", "IPv4 link-local upper boundary"),
    # IPv4 loopback (whole /8)
    ("127.0.0.1", "IPv4 loopback (canonical)"),
    ("127.0.0.0", "IPv4 loopback network address"),
    ("127.5.5.5", "IPv4 loopback (mid /8)"),
    ("127.255.255.254", "IPv4 loopback upper boundary"),
    # IPv6 loopback / link-local / ULA
    ("::1", "IPv6 loopback"),
    ("fe80::1", "IPv6 link-local"),
    ("fe80::dead:beef", "IPv6 link-local (other)"),
    ("fc00::1", "IPv6 unique-local (fc00::/8)"),
    ("fd00::1", "IPv6 unique-local (fd00::/8)"),
    ("fdab:cdef::1", "IPv6 unique-local (random)"),
    # RFC-1918 — 10/8
    ("10.0.0.0", "RFC-1918 10/8 lower boundary"),
    ("10.0.0.1", "RFC-1918 10/8 typical"),
    ("10.255.255.254", "RFC-1918 10/8 upper boundary"),
    # RFC-1918 — 172.16/12
    ("172.16.0.1", "RFC-1918 172.16/12 lower"),
    ("172.20.10.1", "RFC-1918 172.16/12 middle"),
    ("172.31.255.254", "RFC-1918 172.16/12 upper"),
    # RFC-1918 — 192.168/16
    ("192.168.0.1", "RFC-1918 192.168/16 lower"),
    ("192.168.1.1", "RFC-1918 192.168/16 typical"),
    ("192.168.255.254", "RFC-1918 192.168/16 upper"),
    # Unspecified / broadcast
    ("0.0.0.0", "IPv4 unspecified"),
    ("255.255.255.255", "IPv4 broadcast"),
    ("::", "IPv6 unspecified"),
]


@pytest.mark.parametrize(
    "ip",
    [p[0] for p in LITERAL_BLOCKED_IPS],
    ids=[p[1] for p in LITERAL_BLOCKED_IPS],
)
def test_literal_blocked_ip_rejected_http(ip: str) -> None:
    """A literal blocked IP in the URL must be rejected before any network call."""
    # IPv6 literals need bracket syntax in URLs.
    host = f"[{ip}]" if ":" in ip else ip
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(f"http://{host}/anything")


@pytest.mark.parametrize(
    "ip",
    [p[0] for p in LITERAL_BLOCKED_IPS],
    ids=[p[1] for p in LITERAL_BLOCKED_IPS],
)
def test_literal_blocked_ip_rejected_https(ip: str) -> None:
    """Same as above but via https — the scheme must not bypass the IP guard."""
    host = f"[{ip}]" if ":" in ip else ip
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(f"https://{host}/anything")


# ---------------------------------------------------------------------------
# 2. Scheme blocks — only http/https are allowed
# ---------------------------------------------------------------------------

BLOCKED_SCHEME_URLS: list[tuple[str, str]] = [
    ("file:///etc/passwd", "file (POSIX)"),
    ("file://C:/Windows/System32/drivers/etc/hosts", "file (Windows)"),
    ("gopher://example.com/_", "gopher"),
    ("gopher://127.0.0.1:11211/_stat", "gopher (memcached attack)"),
    ("ftp://example.com/file", "ftp"),
    ("ftps://example.com/file", "ftps"),
    ("data:text/plain,hello", "data text"),
    ("data:text/html;base64,PHNjcmlwdD4=", "data html base64"),
    ("javascript:alert(1)", "javascript"),
    ("vbscript:msgbox(1)", "vbscript"),
    ("ws://example.com/", "ws"),
    ("wss://example.com/", "wss"),
    ("ldap://example.com/dc=x", "ldap"),
    ("dict://example.com:11211/stat", "dict"),
    ("ssh://example.com:22", "ssh"),
    ("telnet://example.com:23", "telnet"),
    ("sftp://example.com/file", "sftp"),
    ("smb://server/share", "smb"),
    ("jar:file:///x.jar!/y", "jar"),
    ("blob:https://example.com/abc", "blob"),
    ("about:blank", "about"),
    ("chrome://settings", "chrome"),
    ("view-source:http://example.com", "view-source"),
]


@pytest.mark.parametrize(
    "url",
    [p[0] for p in BLOCKED_SCHEME_URLS],
    ids=[p[1] for p in BLOCKED_SCHEME_URLS],
)
def test_blocked_scheme(url: str) -> None:
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(url)


@pytest.mark.parametrize(
    "url",
    [
        "FILE:///etc/passwd",
        "File:///etc/passwd",
        "fILe:///etc/passwd",
        "FTP://example.com/x",
        "GOPHER://example.com/_",
        "JaVaScRiPt:alert(1)",
        "DATA:text/plain,abc",
    ],
    ids=[
        "FILE upper",
        "File mixed",
        "fILe weird",
        "FTP upper",
        "GOPHER upper",
        "JaVaScRiPt mixed",
        "DATA upper",
    ],
)
def test_blocked_scheme_case_insensitive(url: str) -> None:
    """``urlparse`` lowercases the scheme, so mixed-case must be rejected too."""
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(url)


def test_scheme_empty_rejected() -> None:
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("//example.com/no-scheme")


# ---------------------------------------------------------------------------
# 3. DNS-based blocks (hostname resolves to an internal IP)
# ---------------------------------------------------------------------------

DNS_BLOCKED_IPS: list[tuple[str, str]] = [
    ("169.254.169.254", "AWS metadata via DNS"),
    ("127.0.0.1", "loopback via DNS"),
    ("127.5.5.5", "loopback /8 via DNS"),
    ("10.0.0.1", "10/8 via DNS"),
    ("10.255.255.254", "10/8 upper via DNS"),
    ("172.16.0.1", "172.16/12 via DNS"),
    ("172.31.255.254", "172.16/12 upper via DNS"),
    ("192.168.0.1", "192.168/16 via DNS"),
    ("192.168.255.254", "192.168/16 upper via DNS"),
    ("0.0.0.0", "unspecified via DNS"),
    ("255.255.255.255", "broadcast via DNS"),
    ("::1", "IPv6 loopback via DNS"),
    ("fe80::1", "IPv6 link-local via DNS"),
    ("fc00::1", "IPv6 ULA via DNS"),
    ("fd00::1", "IPv6 ULA (fd) via DNS"),
    ("::", "IPv6 unspecified via DNS"),
]


@pytest.mark.parametrize(
    "ip",
    [p[0] for p in DNS_BLOCKED_IPS],
    ids=[p[1] for p in DNS_BLOCKED_IPS],
)
def test_dns_resolves_to_blocked_ip(patch_dns, ip: str) -> None:
    """A public-looking hostname that resolves to an internal IP must be blocked."""
    patch_dns(ip)
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://attacker.example.com/loot")


def test_dns_failure_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gaierror`` (NXDOMAIN, etc.) must surface as SSRFError, not network err."""

    def fail(*_args, **_kwargs):
        raise socket.gaierror("forced failure")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://does-not-exist.invalid/")


def test_dns_returns_empty_address_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``getaddrinfo`` returns no usable addresses we must refuse the request."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://emptydns.example.com/")


# ---------------------------------------------------------------------------
# 4. Redirect-target blocks (302 Location -> internal IP must be re-validated)
# ---------------------------------------------------------------------------

REDIRECT_TARGETS: list[tuple[str, str]] = [
    ("http://127.0.0.1/admin", "redirect -> loopback"),
    ("http://169.254.169.254/latest/meta-data/", "redirect -> AWS metadata"),
    ("http://10.0.0.1/admin", "redirect -> RFC-1918 10/8"),
    ("http://172.16.0.1/", "redirect -> RFC-1918 172.16/12"),
    ("http://192.168.0.1/", "redirect -> RFC-1918 192.168/16"),
    ("https://[::1]/", "redirect -> IPv6 loopback"),
    ("https://[fe80::1]/", "redirect -> IPv6 link-local"),
    ("https://[fc00::1]/", "redirect -> IPv6 ULA"),
    ("http://0.0.0.0/", "redirect -> unspecified"),
]


@pytest.mark.parametrize(
    "target",
    [p[0] for p in REDIRECT_TARGETS],
    ids=[p[1] for p in REDIRECT_TARGETS],
)
def test_redirect_to_internal_blocked(patch_dns, httpx_mock, target: str) -> None:
    """Initial URL resolves to a public IP; the 302 Location is internal."""
    patch_dns("8.8.8.8")  # makes evil.example.com look benign
    httpx_mock.add_response(
        url="https://evil.example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://evil.example.com/start",
        status_code=302,
        headers={"Location": target},
    )
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://evil.example.com/start")


def test_redirect_chain_blocked_on_second_hop(patch_dns, httpx_mock) -> None:
    """Public -> public -> internal must be blocked at the final hop."""
    patch_dns("8.8.8.8")
    httpx_mock.add_response(
        url="https://evil.example.com/robots.txt", status_code=404, text=""
    )
    httpx_mock.add_response(
        url="https://evil.example.com/a",
        status_code=302,
        headers={"Location": "https://evil.example.com/b"},
    )
    httpx_mock.add_response(
        url="https://evil.example.com/b",
        status_code=302,
        headers={"Location": "http://169.254.169.254/"},
    )
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://evil.example.com/a")


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
        "::ffff:192.168.1.1",
    ],
    ids=[
        "v4-mapped loopback",
        "v4-mapped RFC-1918 10/8",
        "v4-mapped AWS metadata",
        "v4-mapped RFC-1918 192.168/16",
    ],
)
def test_ipv4_mapped_ipv6_literal_blocked(ip: str) -> None:
    """``::ffff:a.b.c.d`` literals must be rejected just like the IPv4 address."""
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch(f"http://[{ip}]/")


@pytest.mark.parametrize(
    "ip",
    [
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
    ],
    ids=[
        "v4-mapped loopback via DNS",
        "v4-mapped RFC-1918 via DNS",
        "v4-mapped AWS metadata via DNS",
    ],
)
def test_ipv4_mapped_ipv6_dns_blocked(patch_dns, ip: str) -> None:
    """DNS resolution returning an IPv4-mapped IPv6 must also be blocked."""
    patch_dns(ip)
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://victim.example.com/")


def test_decimal_ip_encoding_current_behavior() -> None:
    """``http://2130706433/`` should be loopback (127.0.0.1) in decimal.

    Python's ``ipaddress.ip_address`` does NOT decode decimal-IPv4 literals,
    so the validator falls through to ``socket.getaddrinfo``. Behavior is
    platform/OS dependent:
      - On Windows ``getaddrinfo("2130706433", ...)`` raises gaierror -> SSRFError
      - On some Linux glibc versions it would resolve to 127.0.0.1 -> SSRFError

    Either way it must NOT succeed. This test documents the current behavior
    and fails if a future change starts allowing the request.
    """
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://2130706433/")


def test_decimal_ip_via_patched_dns_blocked(patch_dns) -> None:
    """If decimal IP gets resolved to 127.0.0.1 (forced via DNS patch), block."""
    patch_dns("127.0.0.1")
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://2130706433/")


def test_trailing_dot_loopback_blocked(patch_dns) -> None:
    """``localhost.`` (trailing dot) must still resolve through SSRF check."""
    patch_dns("127.0.0.1")
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://localhost./")


def test_trailing_dot_internal_blocked(patch_dns) -> None:
    patch_dns("10.0.0.1")
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://internal.corp./api")


def test_uppercase_hostname_internal_blocked(patch_dns) -> None:
    """Mixed-case hostnames must follow the same rules."""
    patch_dns("127.0.0.1")
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("https://LOCALHOST/x")


def test_userinfo_in_url_does_not_bypass(patch_dns) -> None:
    """``http://user:pass@host/`` must still hit the hostname guard."""
    patch_dns("127.0.0.1")
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://user:pass@victim.example.com/")


def test_port_does_not_bypass() -> None:
    """A non-standard port on a blocked IP must still be rejected."""
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://127.0.0.1:8080/admin")


def test_ipv6_with_zone_id_literal_rejected() -> None:
    """IPv6 with a zone identifier (fe80::1%eth0) is link-local -> blocked.

    Python's ``ipaddress.ip_address`` accepts ``fe80::1%eth0`` as IPv6Address
    on 3.9+; the validator's ``is_link_local`` must catch it. If ``urlparse``
    fails to extract the hostname for any reason we still get SSRFError via
    the "URL has no host" path.
    """
    src = WebSource()
    with pytest.raises(SSRFError):
        src.fetch("http://[fe80::1%25eth0]/")


# ---------------------------------------------------------------------------
# 6. Direct unit tests on _validate_url for completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
    ],
    ids=[
        "loopback",
        "RFC-1918 10/8",
        "RFC-1918 192.168/16",
        "AWS metadata",
        "IPv6 loopback",
        "IPv6 link-local",
        "IPv6 ULA",
    ],
)
def test_validate_url_rejects_literal(url: str) -> None:
    with pytest.raises(SSRFError):
        _validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/",
        "gopher://example.com/",
        "file:///etc/passwd",
        "data:text/plain,abc",
        "javascript:alert(1)",
    ],
)
def test_validate_url_rejects_scheme(url: str) -> None:
    with pytest.raises(SSRFError):
        _validate_url(url)


def test_validate_url_no_host_field() -> None:
    with pytest.raises(SSRFError):
        _validate_url("https:///path-no-host")


def test_validate_url_allows_public_literal() -> None:
    """8.8.8.8 is a clearly public IP and must pass the literal-IP check."""
    scheme, host = _validate_url("https://8.8.8.8/")
    assert scheme == "https"
    assert host == "8.8.8.8"


def test_validate_url_allows_public_hostname(patch_dns) -> None:
    """A hostname resolving to a public IP must pass."""
    patch_dns("8.8.8.8")
    scheme, host = _validate_url("https://example.com/")
    assert scheme == "https"
    assert host == "example.com"
