"""Web source: fetch HTML, extract main content, respect robots.txt, block SSRF."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from crossmem.core.models import Document
from crossmem.sources._http import (
    ALLOWED_SCHEMES,
    SSRFError,
    _is_blocked_ip,
    borrowed_client,
    safe_get,
    validate_url,
)
from crossmem.sources.base import SourceBase

if TYPE_CHECKING:
    from collections.abc import Iterable

# Re-exported for back-compat with callers/tests that imported these from
# ``crossmem.sources.web``.
_validate_url = validate_url

USER_AGENT = "crossmem/0.1 (+https://github.com/crossmem/crossmem)"
DEFAULT_TIMEOUT = 10.0


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows fetching the target URL."""


def _check_robots(client: httpx.Client, scheme: str, host: str, url: str) -> None:
    """Fetch and evaluate ``robots.txt``. Raise ``RobotsDisallowedError`` if denied.

    Follows RFC 9309 sec. 2.3.1:

    - 2xx with body: parse rules, deny if the rules disallow ``url``.
    - 4xx "Unavailable" (incl. 404): no rules apply, fetch is allowed.
    - 5xx "Unreachable": MUST be treated as complete disallow.
    - Transport failure: default-allow (treat as no response).

    :class:`SSRFError` is deliberately **not** caught: a guard rejection
    on the robots.txt hop is the first observable signal of a DNS-rebind
    attempt between the initial host validation in ``WebSource.fetch``
    and the second resolution inside ``safe_get``. Swallowing it would
    let the subsequent page fetch attempt another resolution and either
    succeed against a now-internal target or surface a confusing error
    from a different code path. Propagating aborts the fetch cleanly.
    """
    robots_url = f"{scheme}://{host}/robots.txt"
    try:
        resp = safe_get(client, robots_url)
    except httpx.HTTPError:
        return  # default-allow on transport issues
    if 500 <= resp.status_code < 600:
        raise RobotsDisallowedError(
            f"robots.txt unreachable ({resp.status_code}); "
            f"RFC 9309 requires complete disallow for {url}"
        )
    if resp.status_code != 200:
        return  # 4xx Unavailable -> no rules apply, allow
    parser = RobotFileParser()
    try:
        parser.parse(resp.text.splitlines())
    except (ValueError, IndexError):
        return  # default-allow on parse error
    if not parser.can_fetch(USER_AGENT, url):
        raise RobotsDisallowedError(f"robots.txt disallows fetching {url}")


def _extract_title(soup: BeautifulSoup) -> str:
    """Return the page title, robust against ``<title>`` nodes with children.

    ``soup.title.string`` returns ``None`` when ``<title>`` contains multiple
    children (e.g. ``<title>Foo <span>bar</span></title>``); ``get_text``
    concatenates them safely.
    """
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
        if title:
            return title
    if soup.h1:
        return soup.h1.get_text(" ", strip=True)
    return ""


def _strip_html(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` extracted from ``html`` via BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "aside", "footer", "header", "noscript"]
    ):
        tag.decompose()
    title = _extract_title(soup)
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(" ", strip=True)
    return title, text


class WebSource(SourceBase):
    """Fetch a web page and return a single ``Document`` with its main text."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._client = client

    def name(self) -> str:
        return "web"

    def can_handle(self, uri: str) -> bool:
        try:
            parsed = urlparse(uri)
        except ValueError:
            return False
        if parsed.scheme not in ALLOWED_SCHEMES:
            return False
        return bool(parsed.hostname)

    def fetch(self, query: str, **kwargs: object) -> list[Document]:
        url = query
        scheme, host = validate_url(url)
        with borrowed_client(
            self._client,
            follow_redirects=False,
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            _check_robots(client, scheme, host, url)
            response = safe_get(client, url)
            response.raise_for_status()
            html = response.text
        title, text = _strip_html(html)
        if not text:
            return []
        return [
            Document.from_payload(
                content=text,
                source_url=url,
                title=title,
                source_type="web",
            )
        ]


__all__: Iterable[str] = (
    "RobotsDisallowedError",
    "SSRFError",
    "WebSource",
    "_is_blocked_ip",
    "_validate_url",
)
