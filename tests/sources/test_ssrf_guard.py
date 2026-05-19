"""Cross-source SSRF guard coverage.

Asserts that every source adapter rejects the same SSRF payload list
(disallowed schemes + blocked IPs + redirect-to-internal). Each scenario
is parametrised across all registered sources so a regression in any
adapter shows up as a failure here, not as a latent vulnerability.
"""

from __future__ import annotations

import re
import socket
from typing import Any

import pytest

from crossmem.sources._http import SSRFError
from crossmem.sources.adapters.context7 import Context7Adapter
from crossmem.sources.github import GitHubSource
from crossmem.sources.web import WebSource

# ---------- DNS patching ----------------------------------------------------


def _patch_dns(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    """Force ``socket.getaddrinfo`` to resolve every host to ``ip``."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def fake_getaddrinfo(host: str, *_a: Any, **_kw: Any) -> list[Any]:
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# ---------- Per-source "fetch a URL" callables ------------------------------
#
# Each callable triggers exactly one outbound GET that *should* be blocked
# by the SSRF guard. Adapters that hard-code their endpoint (Context7,
# GitHub) cannot fetch arbitrary URLs, so we construct adapter variants
# whose endpoints already point at the SSRF target.


def _fetch_web(url: str) -> None:
    WebSource().fetch(url)


def _fetch_github(url: str) -> None:
    # Re-aim the GitHub adapter at an attacker-controlled host by mutating
    # the module-level _API_BASE for the lifetime of one call. The adapter
    # builds its request URL from that prefix.
    from crossmem.sources import github as github_module

    original = github_module._API_BASE
    github_module._API_BASE = url.rstrip("/")
    try:
        GitHubSource().fetch("octocat/Hello-World readme")
    finally:
        github_module._API_BASE = original


def _fetch_context7(url: str) -> None:
    Context7Adapter(base_url=url.rstrip("/")).fetch("query")


ALL_SOURCES = pytest.mark.parametrize(
    ("name", "fetch_fn"),
    [
        ("web", _fetch_web),
        ("github", _fetch_github),
        ("context7", _fetch_context7),
    ],
    ids=["web", "github", "context7"],
)


# ---------- Disallowed schemes ---------------------------------------------


@ALL_SOURCES
@pytest.mark.parametrize(
    "scheme_url",
    [
        "file:///etc/passwd",
        "gopher://attacker.example.com/_",
        "ftp://attacker.example.com/x",
        "data:text/plain,abc",
        "javascript:alert(1)",
    ],
    ids=["file", "gopher", "ftp", "data", "javascript"],
)
def test_all_sources_reject_disallowed_schemes(
    name: str, fetch_fn: Any, scheme_url: str
) -> None:
    with pytest.raises(SSRFError):
        fetch_fn(scheme_url)


# ---------- Blocked IPs via DNS --------------------------------------------


BLOCKED_IPS = [
    "127.0.0.1",  # loopback
    "10.0.0.5",  # RFC-1918
    "172.16.0.1",  # RFC-1918
    "192.168.1.1",  # RFC-1918
    "169.254.169.254",  # AWS metadata / link-local
    "0.0.0.0",  # unspecified
    "::1",  # IPv6 loopback
    "fe80::1",  # IPv6 link-local
    "fc00::1",  # IPv6 unique-local
]


@ALL_SOURCES
@pytest.mark.parametrize("ip", BLOCKED_IPS)
def test_all_sources_block_internal_dns_resolutions(
    monkeypatch: pytest.MonkeyPatch, name: str, fetch_fn: Any, ip: str
) -> None:
    _patch_dns(monkeypatch, ip)
    with pytest.raises(SSRFError):
        fetch_fn("https://attacker.example.com/")


# ---------- Literal blocked IPs --------------------------------------------


@ALL_SOURCES
def test_all_sources_reject_literal_internal_ip(name: str, fetch_fn: Any) -> None:
    with pytest.raises(SSRFError):
        fetch_fn("http://127.0.0.1/")


# ---------- Redirect to internal IP ----------------------------------------
#
# Each adapter must re-validate the Location header. We pre-arm pytest-httpx
# with a 302 -> http://127.0.0.1/admin and a robots.txt 404 (only relevant
# for WebSource). The SSRF guard must trip on the redirect target.


@pytest.mark.parametrize(
    ("name", "fetch_fn", "primary_url"),
    [
        ("web", _fetch_web, "https://evil.example.com/"),
        ("github", _fetch_github, "https://evil.example.com"),
        ("context7", _fetch_context7, "https://evil.example.com"),
    ],
    ids=["web", "github", "context7"],
)
def test_all_sources_block_redirect_to_internal_ip(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: Any,
    name: str,
    fetch_fn: Any,
    primary_url: str,
) -> None:
    # External DNS resolves to a safe public IP so the *primary* request
    # would pass the guard; the redirect Location is a literal 127.0.0.1,
    # which the guard must reject on the second hop.
    _patch_dns(monkeypatch, "8.8.8.8")

    # WebSource also fetches /robots.txt first.
    if name == "web":
        httpx_mock.add_response(
            url="https://evil.example.com/robots.txt",
            status_code=404,
            text="",
        )
        redirect_matcher: Any = "https://evil.example.com/"
    elif name == "github":
        redirect_matcher = "https://evil.example.com/repos/octocat/Hello-World/readme"
    else:
        # context7 hits /search with query params; match any URL on the
        # /search path.
        redirect_matcher = re.compile(r"^https://evil\.example\.com/search(\?.*)?$")

    httpx_mock.add_response(
        url=redirect_matcher,
        status_code=302,
        headers={"Location": "http://127.0.0.1/admin"},
    )

    with pytest.raises(SSRFError):
        fetch_fn(primary_url)
