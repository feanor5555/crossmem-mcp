"""Tests for the GitHub source adapter."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from crossmem.core.models import Document
from crossmem.sources._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpResponseTooLargeError,
)
from crossmem.sources.github import GitHubRateLimitError, GitHubSource


def _readme_payload(content: str) -> dict[str, Any]:
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return {
        "name": "README.md",
        "path": "README.md",
        "content": encoded,
        "encoding": "base64",
        "html_url": "https://github.com/octocat/Hello-World/blob/master/README.md",
        "download_url": (
            "https://raw.githubusercontent.com/octocat/Hello-World/master/README.md"
        ),
    }


def _issue_payload(number: int, title: str, body: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": "open",
        "html_url": f"https://github.com/octocat/Hello-World/issues/{number}",
    }


def _code_search_item(path: str, snippet: str = "") -> dict[str, Any]:
    return {
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "html_url": f"https://github.com/octocat/Hello-World/blob/main/{path}",
        "repository": {"full_name": "octocat/Hello-World"},
        "text_matches": [{"fragment": snippet}] if snippet else [],
    }


class TestCanHandle:
    def test_owner_repo_shorthand(self) -> None:
        src = GitHubSource()
        assert src.can_handle("octocat/Hello-World") is True

    def test_github_https_url(self) -> None:
        src = GitHubSource()
        assert src.can_handle("https://github.com/octocat/Hello-World") is True

    def test_github_http_url(self) -> None:
        src = GitHubSource()
        assert src.can_handle("http://github.com/octocat/Hello-World") is True

    def test_non_github_url(self) -> None:
        src = GitHubSource()
        assert src.can_handle("https://gitlab.com/x/y") is False

    def test_plain_text(self) -> None:
        src = GitHubSource()
        assert src.can_handle("not a repo") is False

    def test_single_segment(self) -> None:
        src = GitHubSource()
        assert src.can_handle("octocat") is False


class TestName:
    def test_returns_github(self) -> None:
        assert GitHubSource().name() == "github"


class TestFetchReadme:
    def test_fetch_readme_only(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            json=_readme_payload("# Hello\nWorld"),
        )

        docs = src.fetch("octocat/Hello-World readme")

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.metadata.source_type == "github"
        assert "README" in doc.metadata.title
        assert "Hello" in doc.content
        assert "World" in doc.content
        assert doc.metadata.source_url.startswith(
            "https://github.com/octocat/Hello-World"
        )
        assert "github" in doc.metadata.tags
        assert doc.embedding == ()
        assert doc.metadata.embedding_dim == 0
        assert doc.metadata.embedding_model == ""
        assert doc.id != ""
        assert doc.metadata.content_hash != ""

        request = httpx_mock.get_request()
        assert (
            str(request.url)
            == "https://api.github.com/repos/octocat/Hello-World/readme"
        )

    def test_fetch_readme_user_agent_header(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            json=_readme_payload("body"),
        )

        src.fetch("octocat/Hello-World readme")

        request = httpx_mock.get_request()
        assert request.headers.get("User-Agent") == "crossmem/0.1"
        assert request.headers.get("Accept", "").startswith("application/vnd.github")


class TestFetchIssues:
    def test_fetch_issues_only(self, httpx_mock) -> None:
        src = GitHubSource()
        issues = [
            _issue_payload(1, "Bug A", "details A"),
            _issue_payload(2, "Bug B", "details B"),
        ]
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=10",
            status_code=200,
            json=issues,
        )

        docs = src.fetch("octocat/Hello-World issues")

        assert len(docs) == 2
        titles = {d.metadata.title for d in docs}
        assert "Bug A" in titles
        assert "Bug B" in titles
        for doc in docs:
            assert doc.metadata.source_type == "github"
            assert doc.metadata.source_url.startswith(
                "https://github.com/octocat/Hello-World/issues/"
            )
            assert "github" in doc.metadata.tags

        request = httpx_mock.get_request()
        called_url = str(request.url)
        assert called_url.startswith(
            "https://api.github.com/repos/octocat/Hello-World/issues"
        )
        assert "state=open" in called_url
        assert "per_page=10" in called_url

    def test_fetch_issues_empty(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=10",
            status_code=200,
            json=[],
        )
        docs = src.fetch("octocat/Hello-World issues")
        assert docs == []


class TestFetchCombined:
    def test_fetch_repo_returns_readme_plus_issues(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            json=_readme_payload("# Combined"),
        )
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=10",
            status_code=200,
            json=[_issue_payload(1, "Combined Issue", "body")],
        )

        docs = src.fetch("octocat/Hello-World")

        assert len(docs) == 2
        kinds = {d.metadata.title for d in docs}
        assert any("README" in k for k in kinds)
        assert "Combined Issue" in kinds


class TestFetchSearch:
    def test_search_code(self, httpx_mock) -> None:
        src = GitHubSource()
        items = [
            _code_search_item("src/foo.py", "def foo():\n    return 1"),
            _code_search_item("src/bar.py"),
        ]
        httpx_mock.add_response(
            url="https://api.github.com/search/code?q=foo+bar",
            status_code=200,
            json={"items": items, "total_count": 2},
        )

        docs = src.fetch("search:foo bar")

        assert len(docs) == 2
        assert docs[0].metadata.title.endswith("src/foo.py")
        assert "def foo" in docs[0].content
        assert docs[1].content == ""
        for doc in docs:
            assert doc.metadata.source_type == "github"
            assert "github" in doc.metadata.tags
            assert doc.metadata.source_url.startswith("https://github.com/")

        request = httpx_mock.get_request()
        called_url = str(request.url)
        assert called_url.startswith("https://api.github.com/search/code")
        assert "q=" in called_url
        assert "foo" in called_url and "bar" in called_url

    def test_search_no_items(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/search/code?q=nothing",
            status_code=200,
            json={"items": [], "total_count": 0},
        )
        docs = src.fetch("search:nothing")
        assert docs == []


class TestRateLimit:
    def test_rate_limit_403(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=403,
            json={"message": "API rate limit exceeded"},
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1700000000",
            },
        )
        with pytest.raises(GitHubRateLimitError) as exc_info:
            src.fetch("octocat/Hello-World readme")
        assert "rate limit" in str(exc_info.value).lower()
        assert "1700000000" in str(exc_info.value)

    def test_429_too_many_requests(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=429,
            json={"message": "Too Many Requests"},
            headers={"X-RateLimit-Reset": "1700000001"},
        )
        with pytest.raises(GitHubRateLimitError):
            src.fetch("octocat/Hello-World readme")

    def test_403_without_zero_remaining_is_not_ratelimit(self, httpx_mock) -> None:
        """A 403 that is NOT a rate limit should raise a different error."""
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=403,
            json={"message": "Forbidden"},
            headers={"X-RateLimit-Remaining": "42"},
        )
        with pytest.raises(RuntimeError) as exc_info:
            src.fetch("octocat/Hello-World readme")
        assert not isinstance(exc_info.value, GitHubRateLimitError)

    def test_is_rate_limited_helper_403_zero_remaining(self) -> None:
        """The `_is_rate_limited` helper flags 403 + Remaining=0 as rate-limit."""

        class _FakeResp:
            status_code = 403
            headers = {"X-RateLimit-Remaining": "0"}

        assert GitHubSource._is_rate_limited(_FakeResp()) is True

    def test_is_rate_limited_helper_429_any_remaining(self) -> None:
        """The `_is_rate_limited` helper flags 429 regardless of Remaining."""

        class _FakeResp:
            status_code = 429
            headers = {"X-RateLimit-Remaining": "99"}

        assert GitHubSource._is_rate_limited(_FakeResp()) is True

    def test_is_rate_limited_helper_403_with_remaining_quota(self) -> None:
        """A 403 with non-zero Remaining is NOT a rate limit."""

        class _FakeResp:
            status_code = 403
            headers = {"X-RateLimit-Remaining": "42"}

        assert GitHubSource._is_rate_limited(_FakeResp()) is False

    def test_is_rate_limited_helper_non_rate_status(self) -> None:
        """Any non-403/429 status is NOT a rate limit, even with Remaining=0."""

        class _FakeResp:
            status_code = 500
            headers = {"X-RateLimit-Remaining": "0"}

        assert GitHubSource._is_rate_limited(_FakeResp()) is False


class TestErrorHandling:
    def test_404_raises(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=404,
            json={"message": "Not Found"},
        )
        with pytest.raises(RuntimeError):
            src.fetch("octocat/Hello-World readme")

    def test_invalid_query_unknown_form(self) -> None:
        src = GitHubSource()
        with pytest.raises(ValueError):
            src.fetch("not a valid query string with no slash")


class TestParseQuery:
    def test_url_form_normalised(self, httpx_mock) -> None:
        """URL form 'https://github.com/owner/repo' is accepted."""
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            json=_readme_payload("hi"),
        )
        docs = src.fetch("https://github.com/octocat/Hello-World readme")
        assert len(docs) == 1
        request = httpx_mock.get_request()
        assert "octocat/Hello-World" in str(request.url)

    def test_empty_query_raises(self) -> None:
        src = GitHubSource()
        with pytest.raises(ValueError):
            src.fetch("")
        with pytest.raises(ValueError):
            src.fetch("   ")

    def test_empty_search_keywords_raises(self) -> None:
        src = GitHubSource()
        with pytest.raises(ValueError):
            src.fetch("search:   ")

    def test_unknown_mode_raises(self) -> None:
        src = GitHubSource()
        with pytest.raises(ValueError):
            src.fetch("octocat/Hello-World wat")

    def test_can_handle_empty(self) -> None:
        assert GitHubSource().can_handle("") is False


class TestEdgeCases:
    def test_readme_plain_content_no_base64(self, httpx_mock) -> None:
        """A README payload without base64 encoding is returned verbatim."""
        src = GitHubSource()
        payload = {
            "name": "README.md",
            "content": "raw text body",
            "encoding": "none",
            "html_url": "https://github.com/octocat/Hello-World/blob/master/README.md",
        }
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            json=payload,
        )
        docs = src.fetch("octocat/Hello-World readme")
        assert docs[0].content == "raw text body"

    def test_issues_payload_not_list(self, httpx_mock) -> None:
        """If the issues endpoint returns a non-list (200 with object body),
        the source returns an empty list."""
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=10",
            status_code=200,
            json={"message": "weird"},
        )
        docs = src.fetch("octocat/Hello-World issues")
        assert docs == []

    def test_500_server_error(self, httpx_mock) -> None:
        src = GitHubSource()
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=500,
            json={"message": "boom"},
        )
        with pytest.raises(RuntimeError):
            src.fetch("octocat/Hello-World readme")


class TestResponseSizeLimit:
    def test_oversize_readme_aborts(self, httpx_mock) -> None:
        """A README payload above the cap must abort with a clear error,
        and no document must be returned."""
        src = GitHubSource()
        huge = DEFAULT_MAX_RESPONSE_BYTES + 1
        httpx_mock.add_response(
            url="https://api.github.com/repos/octocat/Hello-World/readme",
            status_code=200,
            headers={"Content-Length": str(huge)},
            content=b"x" * huge,
        )
        with pytest.raises(HttpResponseTooLargeError):
            src.fetch("octocat/Hello-World readme")
