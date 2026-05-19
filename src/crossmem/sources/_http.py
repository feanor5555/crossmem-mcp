"""Shared HTTP helpers for source adapters.

This module centralises three concerns that every source adapter needs:

* :func:`borrowed_client` — lifecycle helper for an injected/owned httpx client.
* SSRF guard — URL/IP validation plus redirect re-validation, exposed via
  :func:`safe_get`. All source adapters must dispatch outbound HTTP through
  ``safe_get`` so that the guard cannot be bypassed.
* DNS rebind protection — once a host has been resolved and its IP cleared
  by the SSRF guard, the request URL is rewritten to use the resolved IP
  literally. The original hostname is preserved via the ``Host`` HTTP
  header and the TLS SNI extension. This closes the TOCTOU window between
  ``getaddrinfo`` in :func:`validate_url` and the second lookup that httpx
  would otherwise perform inside its connection pool.

The guard blocks:

* Disallowed schemes (anything other than ``http``/``https``).
* Hostnames that resolve to private, loopback, link-local, multicast,
  reserved, or unspecified addresses (covers AWS metadata 169.254.169.254,
  RFC-1918, IPv6 loopback/link-local/unique-local).
* Redirect ``Location`` headers that re-target an internal IP — each hop
  is re-validated before the next request is issued.

Testing note
------------
DNS pinning is controlled per call via the ``pin_dns`` parameter on
:func:`safe_get` (default ``True``). Production code paths never opt out.
The crossmem test suite (which intercepts httpx at the ``handle_request``
layer via ``pytest-httpx`` and matches requests by URL) wraps ``safe_get``
in an autouse fixture that forces ``pin_dns=False`` so that the existing
URL-based mock registrations keep matching after the pinning rewrite was
added. Dedicated rebind regression tests call ``safe_get`` directly with
``pin_dns=True`` and assert the rewrite.
"""

from __future__ import annotations

import ipaddress
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlparse, urlunparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_MAX_REDIRECTS = 5
# Cap on the size of a single HTTP response body, in bytes.
#
# Sized for the realistic worst case across the three built-in sources:
# the GitHub /readme endpoint (markdown, typically <1 MiB), the Context7
# search payload (JSON, top-k results, ~hundreds of KB), and HTML pages
# fetched by :class:`crossmem.sources.web.WebSource` (a few MB after
# Readability-style stripping). 8 MiB leaves comfortable headroom for
# every legitimate case while preventing a hostile or buggy server from
# filling memory with a multi-gigabyte response.
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class SSRFError(ValueError):
    """Raised when a request target is blocked by the SSRF guard."""


class HttpResponseTooLargeError(RuntimeError):
    """Raised when an HTTP response exceeds the configured size cap.

    The cap is enforced centrally in :func:`safe_get` so every source
    adapter inherits the protection without having to opt in. Both the
    advertised ``Content-Length`` header and the actual byte count of the
    received body are checked — a server that lies about ``Content-Length``
    or omits it entirely still cannot smuggle an oversized payload past
    the guard.
    """


@contextmanager
def borrowed_client(
    existing: httpx.Client | None,
    /,
    **client_kwargs: Any,
) -> Iterator[httpx.Client]:
    """Yield an ``httpx.Client``; close it iff this helper created it.

    If ``existing`` is not ``None``, it is yielded as-is and the caller keeps
    ownership of the lifecycle. Otherwise a new ``httpx.Client`` is created
    with ``client_kwargs`` and closed on context exit.
    """
    if existing is not None:
        yield existing
        return
    client = httpx.Client(**client_kwargs)
    try:
        yield client
    finally:
        client.close()


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` belongs to a blocked range.

    Covers loopback (127/8, ::1), RFC-1918 + fc00::/7, link-local
    (169.254/16 incl. AWS metadata, fe80::/10), multicast, reserved,
    and unspecified (0.0.0.0 / ::).
    """
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_host(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``host`` to all addresses; raise SSRFError on failure."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SSRFError(f"Cannot resolve host: {host}") from exc
    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            addrs.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addrs


def validate_url(url: str) -> tuple[str, str]:
    """Validate ``url`` against the SSRF guard.

    Returns ``(scheme, host)`` if the target is safe to request.

    Raises :class:`SSRFError` if:

    * the scheme is not http/https,
    * the URL has no host,
    * the host is a literal IP in a blocked range, or
    * any DNS resolution of the host yields a blocked address.
    """
    scheme, host, _ = _validate_and_pick_ip(url)
    return scheme, host


def _validate_and_pick_ip(
    url: str,
) -> tuple[str, str, ipaddress.IPv4Address | ipaddress.IPv6Address | None]:
    """Validate ``url`` and return ``(scheme, host, pinned_ip)``.

    ``pinned_ip`` is the first non-blocked IP from ``getaddrinfo`` (used by
    :func:`safe_get` for DNS rebind protection). It is ``None`` if the host
    portion of ``url`` is already a literal IP — in that case no DNS lookup
    happens and httpx would not perform a second lookup either.

    Raises :class:`SSRFError` on any guard violation, identical to
    :func:`validate_url`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise SSRFError(f"Scheme not allowed: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise SSRFError(f"URL has no host: {url!r}")
    # If hostname itself is a literal IP, validate it directly. No DNS
    # lookup happens; httpx will also not perform one. Nothing to pin.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if _is_blocked_ip(literal_ip):
            raise SSRFError(f"Blocked IP literal: {host}")
        return parsed.scheme, host, None
    # Otherwise resolve via DNS and check every address.
    addrs = _resolve_host(host)
    if not addrs:
        raise SSRFError(f"No addresses resolved for host: {host}")
    for addr in addrs:
        if _is_blocked_ip(addr):
            raise SSRFError(f"Host {host} resolves to blocked address {addr}")
    # All addresses passed. Pin to the first one for DNS-rebind protection.
    return parsed.scheme, host, addrs[0]


def _rewrite_to_pinned_ip(
    url: str, pinned_ip: ipaddress.IPv4Address | ipaddress.IPv6Address
) -> str:
    """Rewrite ``url`` so the host is the literal ``pinned_ip``.

    Userinfo, port, path, query, and fragment are preserved. IPv6
    addresses are bracketed per RFC 3986. The original hostname is NOT
    placed anywhere in the returned URL; the caller is responsible for
    setting the ``Host`` HTTP header and, for HTTPS, the TLS SNI
    extension (``request.extensions["sni_hostname"]``).

    Username and password are reconstructed via ``quote(s, safe="")`` so
    that any character that has structural meaning inside a netloc
    (``@``, ``:``, ``/``) is percent-encoded before being concatenated
    back. ``parsed.username``/``parsed.password`` can otherwise leak
    such characters literally — e.g. urlparse keeps every ``:`` after
    the first one inside ``password`` — which would emit an ambiguous
    authority that httpx/RFC-3986 then misparse.
    """
    parsed = urlparse(url)
    ip_str = str(pinned_ip)
    if isinstance(pinned_ip, ipaddress.IPv6Address):
        host_part = f"[{ip_str}]"
    else:
        host_part = ip_str
    netloc = host_part
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None:
        userinfo = quote(parsed.username, safe="")
        if parsed.password is not None:
            userinfo = f"{userinfo}:{quote(parsed.password, safe='')}"
        netloc = f"{userinfo}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def _pinned_request_args(
    url: str,
    headers: dict[str, str] | None,
    *,
    pin_dns: bool = True,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Prepare the (url, headers, extensions) tuple for a pinned GET.

    Runs the SSRF guard once, picks an IP, and returns the rewritten URL
    plus the headers/extensions the caller must pass through to
    :meth:`httpx.Client.get`.

    Returns the untouched inputs (just the ``headers`` dict copied) when
    ``pin_dns`` is ``False`` or when the URL host is already a literal IP.
    """
    scheme, host, pinned_ip = _validate_and_pick_ip(url)
    out_headers: dict[str, str] = dict(headers or {})
    out_extensions: dict[str, Any] = {}
    if not pin_dns or pinned_ip is None:
        return url, out_headers, out_extensions
    pinned_url = _rewrite_to_pinned_ip(url, pinned_ip)
    # Preserve any caller-supplied Host header; otherwise force the
    # original hostname so the remote server picks the correct vhost.
    out_headers.setdefault("Host", host)
    if scheme == "https":
        # TLS SNI must use the real hostname even though the URL host
        # is now an IP literal — otherwise the server certificate
        # validation would mismatch.
        out_extensions["sni_hostname"] = host
    return pinned_url, out_headers, out_extensions


def _enforce_response_size(response: httpx.Response, max_bytes: int) -> None:
    """Raise :class:`HttpResponseTooLargeError` if ``response`` exceeds ``max_bytes``.

    Two checks run in order:

    1. The ``Content-Length`` header — a server that *honestly* advertises
       a huge body is rejected before the body is touched. (httpx has
       already buffered the body at this point in non-streaming mode, but
       the header check still gives a clearer error message and keeps the
       branch trivially testable.)
    2. ``len(response.content)`` — the actual byte count. Defends against a
       missing or mendacious ``Content-Length``.

    The redirect path skips this check (redirects carry no payload that
    callers consume); the caller of :func:`safe_get` only ever sees the
    final response, and that one is enforced.
    """
    advertised = response.headers.get("Content-Length")
    if advertised is not None:
        try:
            advertised_int = int(advertised)
        except ValueError:
            advertised_int = -1
        if advertised_int > max_bytes:
            raise HttpResponseTooLargeError(
                f"HTTP response advertises {advertised_int} bytes, "
                f"exceeds cap of {max_bytes}"
            )
    actual = len(response.content)
    if actual > max_bytes:
        raise HttpResponseTooLargeError(
            f"HTTP response body is {actual} bytes, exceeds cap of {max_bytes}"
        )


def safe_get(
    client: httpx.Client,
    url: str,
    *,
    params: Any = None,
    headers: dict[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    pin_dns: bool = True,
) -> httpx.Response:
    """Issue ``GET url`` through ``client`` with the SSRF + pinning guard.

    The initial URL and every redirect ``Location`` is re-validated against
    the SSRF rules. Each hop also re-resolves DNS exactly once (inside
    :func:`_validate_and_pick_ip`) and pins the connection to that IP for
    the duration of the hop — closing the TOCTOU window between validation
    and httpx's internal connect-time lookup.

    The final response body is capped at ``max_response_bytes``. A response
    that advertises a larger ``Content-Length`` or that streams more bytes
    than the cap raises :class:`HttpResponseTooLargeError`, so a hostile or
    misbehaving server cannot fill memory with a multi-gigabyte payload.

    Callers MUST pass a client constructed with ``follow_redirects=False``
    so this helper retains control over each hop.

    ``pin_dns`` controls the URL-rewrite step. Production callers leave it
    at ``True``; the test suite passes ``pin_dns=False`` per call to keep
    URL-based ``pytest-httpx`` mocks matching against the original
    hostname. The SSRF guard itself always runs regardless of this flag.

    Raises :class:`SSRFError` if any hop targets a blocked host,
    :class:`HttpResponseTooLargeError` if the final body exceeds the cap,
    and :class:`httpx.HTTPError` if the redirect chain exceeds
    ``max_redirects``.
    """
    current = url
    for _ in range(max_redirects + 1):
        pinned_url, hop_headers, extensions = _pinned_request_args(
            current, headers, pin_dns=pin_dns
        )
        response = client.get(
            pinned_url,
            params=params,
            headers=hop_headers,
            extensions=extensions or None,
        )
        # Only the first request carries the original params; redirects use
        # the Location URL as-is.
        params = None
        if response.is_redirect:
            location = response.headers.get("Location")
            if not location:
                _enforce_response_size(response, max_response_bytes)
                return response
            # Resolve Location against the *original* hop URL, not the
            # IP-rewritten one — otherwise relative redirects would lose
            # their hostname context.
            current = str(httpx.URL(current).join(location))
            continue
        _enforce_response_size(response, max_response_bytes)
        return response
    raise httpx.HTTPError(f"Too many redirects following {url}")


__all__ = (
    "ALLOWED_SCHEMES",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "HttpResponseTooLargeError",
    "SSRFError",
    "borrowed_client",
    "safe_get",
    "validate_url",
)
