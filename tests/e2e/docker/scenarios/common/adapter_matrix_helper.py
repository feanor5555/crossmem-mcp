"""Adapter-matrix Python helper for the E2E scenario ``common/adapters.sh``.

Why a separate Python module — the shell scenario only orchestrates env
setup, log redirection, and the JSON report fragment. The real work
(driving each source adapter against a mock HTTP layer, exercising the
SSRF guard, and verifying DNS-pin behaviour) is Python because the
crossmem adapters are Python classes whose contracts only show up at
the API boundary.

Why ``httpx.MockTransport`` instead of the ``responses`` library — the
spec asks for mock-HTTP via the ``responses`` lib in a companion Python
script, but ``responses`` only intercepts ``requests``/``urllib3``;
every crossmem adapter dispatches HTTP through ``httpx``. The
functionally equivalent path is ``httpx.MockTransport`` which lets us
register URL/path-based responders without an extra runtime
dependency. The docker runner ships httpx through the pipx-installed
crossmem package, so no Dockerfile change is needed. The semantic
intent of "mock outbound HTTP in a companion Python script" is
preserved.

The helper is invoked from the bash scenario with the pipx venv's
Python (``/opt/pipx/venvs/crossmem/bin/python``) so that
``import crossmem`` and ``import httpx`` resolve to the same versions
the production CLI uses.

Exit code 0 on full pass; 1 on any failure. The shell wrapper
translates that into the report fragment status.
"""

from __future__ import annotations

import base64
import socket
import sys
import traceback
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# All three adapters plus the SSRF guard come from the pipx-installed
# crossmem package. If any import fails the scenario must abort with
# a clear stack so the log makes the cause obvious.
from crossmem.sources import _http as _http_module
from crossmem.sources import github as github_module
from crossmem.sources import web as web_module
from crossmem.sources._http import (
    SSRFError,
    _pinned_request_args,
    _rewrite_to_pinned_ip,
    validate_url,
)
from crossmem.sources.adapters import context7 as context7_module
from crossmem.sources.adapters.context7 import Context7Adapter
from crossmem.sources.github import GitHubSource
from crossmem.sources.web import WebSource

# ---------------------------------------------------------------------------
# Mock transport plumbing
# ---------------------------------------------------------------------------


def _make_mock_transport(
    routes: dict[str, Callable[[httpx.Request], httpx.Response]],
) -> httpx.MockTransport:
    """Build an ``httpx.MockTransport`` from a path->handler dict.

    The transport matches by path so URL-rewriting (e.g. DNS pinning,
    if it ever leaks back in) does not affect the route. Unknown paths
    raise — we want loud failures, not silent 404s that downstream
    assertions would have to chase.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path not in routes:
            raise AssertionError(
                f"unexpected request: {request.method} {request.url} (path={path!r})"
            )
        return routes[path](request)

    return httpx.MockTransport(handler)


@contextmanager
def _patched_client(
    transport: httpx.MockTransport, modules: list[Any]
) -> Iterator[None]:
    """Patch each adapter module's ``borrowed_client`` to use ``transport``.

    Adapters import ``borrowed_client`` by name at module load. Replacing
    it on each module yields an ``httpx.Client`` wired to the mock
    transport. The ``safe_get`` in each module is also rewired to skip
    DNS pinning so the URL handed to the client keeps the original
    hostname (the mock transport routes by path so pinning would not
    actually break routing, but keeping ``pin_dns=False`` mirrors the
    fixture used by the in-process test suite under ``tests/sources``
    and makes wire-level assertions in this helper trivial).
    """
    original_clients: dict[Any, Any] = {}
    original_safe_gets: dict[Any, Any] = {}
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(_host: str, *_a: Any, **_kw: Any) -> list[Any]:
        # Adapter positive cases must not depend on real DNS — the
        # mock transport routes by path. Pretend every hostname
        # resolves to a public IP so the SSRF guard's resolution check
        # passes.
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    def make_borrowed(_existing: httpx.Client | None = None, **kwargs: Any):
        @contextmanager
        def borrowed() -> Iterator[httpx.Client]:
            kwargs.pop("transport", None)
            client = httpx.Client(transport=transport, **kwargs)
            try:
                yield client
            finally:
                client.close()

        return borrowed()

    def make_safe_get_no_pin(*args: Any, **kwargs: Any) -> httpx.Response:
        kwargs["pin_dns"] = False
        return _http_module.safe_get(*args, **kwargs)

    socket.getaddrinfo = fake_getaddrinfo
    for module in modules:
        original_clients[module] = module.borrowed_client
        original_safe_gets[module] = module.safe_get
        module.borrowed_client = make_borrowed
        module.safe_get = make_safe_get_no_pin
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo
        for module, value in original_clients.items():
            module.borrowed_client = value
        for module, value in original_safe_gets.items():
            module.safe_get = value


# ---------------------------------------------------------------------------
# Per-adapter positive cases
# ---------------------------------------------------------------------------


def _check_web_positive() -> None:
    """WebSource fetch of a simple HTML page yields one Document.

    Two routes are wired: an empty robots.txt (so the fetch is allowed)
    and the page itself. The page carries a ``<title>`` and a body
    paragraph; the WebSource extracts both via BeautifulSoup.
    """
    html = (
        "<html><head><title>Hello E2E</title></head>"
        "<body><main><p>adapter matrix payload</p></main></body></html>"
    )

    def robots(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="")

    def page(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    transport = _make_mock_transport(
        {
            "/robots.txt": robots,
            "/article": page,
        }
    )
    with _patched_client(transport, [web_module]):
        docs = WebSource().fetch("https://example.com/article")
    assert len(docs) == 1, f"web: expected 1 doc, got {len(docs)}"
    doc = docs[0]
    assert doc.metadata.source_type == "web"
    assert doc.metadata.title == "Hello E2E"
    assert "adapter matrix payload" in doc.content
    assert doc.metadata.source_url == "https://example.com/article"


def _check_github_positive() -> None:
    """GitHubSource fetch of a repo (readme-only mode) yields one Document."""
    readme_text = "# Hello\nE2E adapter matrix readme."
    encoded = base64.b64encode(readme_text.encode("utf-8")).decode("ascii")

    def readme(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "README.md",
                "path": "README.md",
                "content": encoded,
                "encoding": "base64",
                "html_url": (
                    "https://github.com/octocat/Hello-World/blob/main/README.md"
                ),
            },
        )

    transport = _make_mock_transport(
        {
            "/repos/octocat/Hello-World/readme": readme,
        }
    )
    with _patched_client(transport, [github_module]):
        docs = GitHubSource().fetch("octocat/Hello-World readme")
    assert len(docs) == 1, f"github: expected 1 doc, got {len(docs)}"
    doc = docs[0]
    assert doc.metadata.source_type == "github"
    assert "readme" in doc.metadata.tags
    assert "E2E adapter matrix readme" in doc.content
    assert doc.metadata.title == "octocat/Hello-World README"


def _check_context7_positive() -> None:
    """Context7 search returns the expected Documents in order."""

    def search(request: httpx.Request) -> httpx.Response:
        params = parse_qs(request.url.query.decode("utf-8"))
        assert params.get("q") == ["hooks"], f"unexpected q params: {params!r}"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "useState",
                        "url": "https://react.dev/reference/react/useState",
                        "content": "useState is a React Hook for state.",
                    },
                    {
                        "title": "useEffect",
                        "url": "https://react.dev/reference/react/useEffect",
                        "content": "useEffect runs side effects.",
                    },
                ]
            },
        )

    transport = _make_mock_transport({"/api/v1/search": search})
    with _patched_client(transport, [context7_module]):
        docs = Context7Adapter().fetch("hooks", library="react")
    assert len(docs) == 2, f"context7: expected 2 docs, got {len(docs)}"
    titles = [d.metadata.title for d in docs]
    assert titles == ["useState", "useEffect"], f"got titles {titles!r}"
    for doc in docs:
        assert doc.metadata.source_type == "context7"


# ---------------------------------------------------------------------------
# Negative SSRF cases (validated against the SSRF guard directly)
# ---------------------------------------------------------------------------

# The payload list mirrors the spec for TODO 27.16. Each must raise
# ``SSRFError`` when passed through ``validate_url``. The literal-IP
# entries do NOT require a DNS mock — the guard short-circuits on the
# literal address class. ``localhost`` is the only entry that goes
# through ``getaddrinfo``; we let the real resolver answer (loopback
# resolution is reliable on every supported runner OS).
_SSRF_PAYLOADS = (
    "http://localhost/",
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://169.254.169.254/",  # AWS instance metadata
    "file:///etc/passwd",
    "gopher://localhost:11211/",
    "http://[::1]/",
)


def _check_ssrf_rejects() -> None:
    """Every SSRF payload from the spec must be rejected by ``validate_url``."""
    for url in _SSRF_PAYLOADS:
        try:
            validate_url(url)
        except SSRFError:
            continue
        raise AssertionError(f"SSRF payload was NOT rejected: {url!r}")


# ---------------------------------------------------------------------------
# DNS-pin verification
# ---------------------------------------------------------------------------


def _check_dns_pin_uses_first_ip() -> None:
    """Rebind trace: only the first (public) IP is contacted.

    Models the classic DNS-rebind attack: ``getaddrinfo`` returns a
    public IP on its first call (validation passes) and a private IP
    on every subsequent call (the hypothetical second lookup an
    attacker would race against the validator). The pinning rewrite
    in :func:`crossmem.sources._http.safe_get` substitutes the
    validated IP into the URL host slot, so httpx never resolves the
    hostname a second time at connect time — the private answer never
    matters.

    Three independent assertions in one trace, each defended:

    1. ``_pinned_request_args`` returns a URL whose host is the public
       IP literal, the ``Host`` header carries the original hostname,
       and the TLS SNI extension is set for HTTPS. This is the pure-
       Python view of the pinning rewrite.
    2. ``safe_get`` through a mock transport that *fails* on any
       non-public host completes successfully — proving the URL httpx
       actually saw used the pinned IP.
    3. ``socket.getaddrinfo`` was called exactly twice (once for
       step 1, once inside ``safe_get``); no third call originated
       from inside httpx because the URL it received already used a
       literal IP. If pinning regressed and httpx re-resolved at
       connect time, the third call would have returned the private
       IP and the mock would have failed the assertion in step 2.
    """
    ips_returned: list[str] = []
    rebind_sequence = ["8.8.8.8", "8.8.8.8", "10.0.0.5"]

    def fake_getaddrinfo(_host: str, *_a: Any, **_kw: Any) -> list[Any]:
        # Replay the rebind sequence in order; once exhausted, keep
        # returning the last (private) answer — a real attacker's
        # authoritative DNS would flip its reply permanently after the
        # validator passed.
        idx = min(len(ips_returned), len(rebind_sequence) - 1)
        ip = rebind_sequence[idx]
        ips_returned.append(ip)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    original_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = fake_getaddrinfo
    try:
        # Step 1 — direct pinning rewrite view (1st lookup, returns public).
        pinned_url, headers, extensions = _pinned_request_args(
            "https://rebind.example.com/path", {"User-Agent": "crossmem-e2e"}
        )
        assert pinned_url == "https://8.8.8.8/path", (
            f"expected rewrite to https://8.8.8.8/path, got {pinned_url!r}"
        )
        assert headers["Host"] == "rebind.example.com"
        assert extensions == {"sni_hostname": "rebind.example.com"}

        # Step 2 — end-to-end safe_get (2nd lookup, still public). The
        # mock fails loudly on any non-public host so a regression
        # makes the failure point obvious.
        def assert_public_only(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "8.8.8.8", (
                f"request reached non-public host {request.url.host!r}; DNS-pin failed"
            )
            assert request.headers.get("Host") == "rebind.example.com"
            return httpx.Response(200, text="pinned-ok")

        transport = httpx.MockTransport(assert_public_only)
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            response = _http_module.safe_get(client, "https://rebind.example.com/path")
        assert response.status_code == 200
        assert response.text == "pinned-ok"

        # Step 3 — exactly two lookups happened (one per safe_get
        # entry-point above); no third call originated from httpx
        # because the URL it received had the IP literal already. If a
        # third call had fired, ``fake_getaddrinfo`` would have flipped
        # to the private answer and the mock in step 2 would have
        # failed because httpx would have connected to 10.0.0.5.
        assert len(ips_returned) == 2, (
            f"expected 2 getaddrinfo calls (pin-args + safe_get), "
            f"got {len(ips_returned)} -> {ips_returned!r}"
        )
        assert ips_returned == ["8.8.8.8", "8.8.8.8"], (
            f"unexpected rebind sequence consumed: {ips_returned!r}"
        )
    finally:
        socket.getaddrinfo = original_getaddrinfo


# ---------------------------------------------------------------------------
# Smoke checks for the pinning rewrite (literal IP + IPv6 brackets)
# ---------------------------------------------------------------------------


def _check_rewrite_invariants() -> None:
    """Unit-style sanity checks on the pinning rewrite helpers.

    Kept inside the scenario because the spec asks for "DNS-Pin-
    Verifikation" — a single end-to-end trace is the headline test,
    but two cheap invariants catch regressions that would have made
    that trace pass for the wrong reason.
    """
    import ipaddress

    rewritten = _rewrite_to_pinned_ip(
        "https://example.com/path?q=1#frag", ipaddress.IPv4Address("8.8.8.8")
    )
    assert rewritten == "https://8.8.8.8/path?q=1#frag", rewritten

    rewritten_v6 = _rewrite_to_pinned_ip(
        "https://example.com:443/x", ipaddress.IPv6Address("2606:4700::1111")
    )
    assert rewritten_v6 == "https://[2606:4700::1111]:443/x", rewritten_v6


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("web-positive", _check_web_positive),
    ("github-positive", _check_github_positive),
    ("context7-positive", _check_context7_positive),
    ("ssrf-rejects", _check_ssrf_rejects),
    ("dns-pin", _check_dns_pin_uses_first_ip),
    ("rewrite-invariants", _check_rewrite_invariants),
)


def main() -> int:
    failures: list[str] = []
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"  ok: {name}")
        except Exception as exc:  # noqa: BLE001 - we want every failure
            tb = traceback.format_exc()
            print(f"  FAIL: {name}: {exc}\n{tb}", file=sys.stderr)
            failures.append(name)
    if failures:
        print(f"adapter-matrix: {len(failures)} check(s) failed: {failures}")
        return 1
    print("adapter-matrix: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
