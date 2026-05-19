"""GitHub source adapter.

Fetches public GitHub data (READMEs, issues, code search) without
authentication. Intended only for public repositories — unauthenticated
calls share GitHub's anonymous rate limit (60 req/hour/IP).

Limitations:
    The /search/code endpoint requires authentication on the live GitHub
    API (returns 422/403 unauthenticated). Tests mock the response, but
    real-world callers must supply a token via a future authenticated
    variant of this source. The current implementation issues the request
    unauthenticated and surfaces any error to the caller.
"""

from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from crossmem.core.models import Document
from crossmem.sources._http import borrowed_client, safe_get
from crossmem.sources.base import SourceBase

if TYPE_CHECKING:
    import httpx

_API_BASE = "https://api.github.com"
_USER_AGENT = "crossmem/0.1"
_ACCEPT = "application/vnd.github+json"
_TIMEOUT = 10.0

_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*)(?:[/?#].*)?$"
)


class GitHubRateLimitError(RuntimeError):
    """Raised when the GitHub API signals a rate-limit (403 or 429)."""


class GitHubSource(SourceBase):
    """Adapter for public GitHub repositories and code search.

    Supported query forms passed to :meth:`fetch`:

    * ``"owner/repo"`` — README plus the 10 most recent open issues.
    * ``"owner/repo readme"`` — README only.
    * ``"owner/repo issues"`` — open issues only (top 10).
    * ``"https://github.com/owner/repo [readme|issues]"`` — same as above.
    * ``"search:KEYWORDS"`` — code search via ``/search/code``.
    """

    def name(self) -> str:
        return "github"

    def can_handle(self, uri: str) -> bool:
        if not uri:
            return False
        if _GITHUB_URL_RE.match(uri):
            return True
        return bool(_OWNER_REPO_RE.match(uri))

    def fetch(self, query: str, **_: Any) -> list[Document]:
        query = (query or "").strip()
        if not query:
            raise ValueError("empty query")

        if query.startswith("search:"):
            keywords = query[len("search:") :].strip()
            if not keywords:
                raise ValueError("empty search keywords")
            return self._search_code(keywords)

        repo, mode = self._parse_repo_query(query)
        if mode == "readme":
            return [self._fetch_readme(repo)]
        if mode == "issues":
            return self._fetch_issues(repo)
        return [self._fetch_readme(repo), *self._fetch_issues(repo)]

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _parse_repo_query(query: str) -> tuple[str, str]:
        """Return ``(owner/repo, mode)`` from a query string.

        ``mode`` is one of ``"readme"``, ``"issues"``, or ``"all"``.
        """
        parts = query.split()
        if not parts:
            raise ValueError("empty query")

        head = parts[0]
        url_match = _GITHUB_URL_RE.match(head)
        if url_match:
            repo = f"{url_match.group(1)}/{url_match.group(2)}"
        elif _OWNER_REPO_RE.match(head):
            repo = head
        else:
            raise ValueError(
                f"unsupported query form: {query!r}; expected 'owner/repo', "
                "a github.com URL, or 'search:KEYWORDS'"
            )

        if len(parts) == 1:
            return repo, "all"
        suffix = parts[1].lower()
        if suffix == "readme":
            return repo, "readme"
        if suffix == "issues":
            return repo, "issues"
        raise ValueError(f"unknown query mode {suffix!r}; use readme or issues")

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        """Return True if the response signals a GitHub rate-limit.

        Treats HTTP 429 unconditionally, and HTTP 403 with the header
        ``X-RateLimit-Remaining: 0`` as a rate-limit. All other statuses
        return False.
        """
        status = response.status_code
        if status == 429:
            return True
        if status == 403:
            return response.headers.get("X-RateLimit-Remaining") == "0"
        return False

    @staticmethod
    def _check_response(response: httpx.Response) -> None:
        """Raise on rate-limit or generic HTTP errors."""
        status = response.status_code
        if GitHubSource._is_rate_limited(response):
            reset = response.headers.get("X-RateLimit-Reset", "unknown")
            raise GitHubRateLimitError(
                f"GitHub rate limit hit (status {status}); resets at {reset}"
            )
        if status == 403:
            raise RuntimeError(
                f"GitHub API forbidden (status {status}): {_safe_message(response)}"
            )
        if status >= 400:
            raise RuntimeError(
                f"GitHub API error (status {status}): {_safe_message(response)}"
            )

    @staticmethod
    def _request(url: str) -> httpx.Response:
        with borrowed_client(
            None,
            timeout=_TIMEOUT,
            follow_redirects=False,
        ) as client:
            return safe_get(
                client,
                url,
                headers={"User-Agent": _USER_AGENT, "Accept": _ACCEPT},
            )

    # ------------------------------------------------------------------ ops

    def _fetch_readme(self, repo: str) -> Document:
        url = f"{_API_BASE}/repos/{repo}/readme"
        response = self._request(url)
        self._check_response(response)
        payload = response.json() or {}

        content = _decode_readme(payload)
        html_url = payload.get("html_url") or f"https://github.com/{repo}"
        title = f"{repo} README"
        return _build_document(
            content=content,
            source_url=html_url,
            title=title,
            extra_tags=["readme"],
        )

    def _fetch_issues(self, repo: str) -> list[Document]:
        url = f"{_API_BASE}/repos/{repo}/issues?state=open&per_page=10"
        response = self._request(url)
        self._check_response(response)
        items = response.json() or []
        if not isinstance(items, list):
            return []
        docs: list[Document] = []
        for issue in items:
            title = issue.get("title") or f"Issue #{issue.get('number', '?')}"
            body = issue.get("body") or ""
            html_url = issue.get("html_url") or f"https://github.com/{repo}/issues"
            docs.append(
                _build_document(
                    content=body,
                    source_url=html_url,
                    title=title,
                    extra_tags=["issue"],
                )
            )
        return docs

    def _search_code(self, keywords: str) -> list[Document]:
        url = f"{_API_BASE}/search/code?q={quote_plus(keywords)}"
        response = self._request(url)
        self._check_response(response)
        payload = response.json() or {}
        items = payload.get("items") or []
        docs: list[Document] = []
        for item in items:
            path = item.get("path") or item.get("name") or "<unknown>"
            repo_full = (item.get("repository") or {}).get("full_name", "")
            title = f"{repo_full} {path}".strip()
            html_url = item.get("html_url") or "https://github.com/"
            snippet = ""
            matches = item.get("text_matches") or []
            if matches:
                snippet = matches[0].get("fragment") or ""
            docs.append(
                _build_document(
                    content=snippet,
                    source_url=html_url,
                    title=title,
                    extra_tags=["code"],
                )
            )
        return docs


# ---------------------------------------------------------------------- helpers


def _safe_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return ""
    if isinstance(body, dict):
        return str(body.get("message", ""))
    return ""


def _decode_readme(payload: dict[str, Any]) -> str:
    """Return README text from a /readme payload (base64 or download_url)."""
    content = payload.get("content")
    encoding = payload.get("encoding")
    if content and encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return ""
    if isinstance(content, str):
        return content
    return ""


def _build_document(
    content: str,
    source_url: str,
    title: str,
    extra_tags: list[str] | None = None,
) -> Document:
    return Document.from_payload(
        content=content,
        source_url=source_url,
        title=title,
        source_type="github",
        tags=["github", *(extra_tags or [])],
    )
