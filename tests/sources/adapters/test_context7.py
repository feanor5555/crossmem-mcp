"""Tests for the Context7 adapter."""

from __future__ import annotations

import re

import httpx
import pytest

from crossmem.sources._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpResponseTooLargeError,
)
from crossmem.sources.adapters.context7 import Context7Adapter


def test_name():
    assert Context7Adapter().name() == "context7"


def test_can_handle_context7_scheme():
    adapter = Context7Adapter()
    assert adapter.can_handle("context7:react/hooks") is True


def test_can_handle_context7_url():
    adapter = Context7Adapter()
    assert adapter.can_handle("https://context7.com/api/v1/search") is True


def test_can_handle_rejects_other():
    adapter = Context7Adapter()
    assert adapter.can_handle("https://example.com") is False
    assert adapter.can_handle("file:///etc/passwd") is False


def test_fetch_returns_documents_for_each_result(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
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

    adapter = Context7Adapter()
    docs = adapter.fetch("hooks", library="react")

    assert len(docs) == 2
    titles = [d.metadata.title for d in docs]
    assert titles == ["useState", "useEffect"]
    for doc in docs:
        assert doc.metadata.source_type == "context7"
        assert doc.embedding == ()
        assert doc.metadata.embedding_model == ""
        assert doc.metadata.embedding_dim == 0
        assert doc.metadata.tags == ()
        assert doc.metadata.content_hash
        assert doc.metadata.stored_at
        assert len(doc.id) == 32

    # Verify request URL and params
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    assert req.url.path == "/api/v1/search"
    assert req.url.params["q"] == "hooks"
    assert req.url.params["library"] == "react"


def test_fetch_without_library_omits_param(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
        json={"results": []},
    )

    adapter = Context7Adapter()
    docs = adapter.fetch("anything")

    assert docs == []
    req = httpx_mock.get_request()
    assert "library" not in req.url.params


def test_fetch_passes_top_k_param(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
        json={"results": []},
    )

    adapter = Context7Adapter()
    adapter.fetch("foo", top_k=25)

    req = httpx_mock.get_request()
    assert req.url.params["top_k"] == "25"


def test_fetch_uses_custom_base_url(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://my\.context7\.test/v2/search\?.*"),
        status_code=200,
        json={"results": []},
    )

    adapter = Context7Adapter(base_url="https://my.context7.test/v2")
    adapter.fetch("q")

    req = httpx_mock.get_request()
    assert str(req.url).startswith("https://my.context7.test/v2/search")


def test_fetch_raises_on_http_error(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=500,
        text="boom",
    )

    adapter = Context7Adapter()
    with pytest.raises(httpx.HTTPStatusError):
        adapter.fetch("q")


def test_fetch_handles_missing_optional_fields(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
        json={"results": [{"content": "only content here"}]},
    )

    adapter = Context7Adapter()
    docs = adapter.fetch("q")

    assert len(docs) == 1
    assert docs[0].content == "only content here"
    assert docs[0].metadata.title == ""
    assert docs[0].metadata.source_url == ""


def test_fetch_rejects_oversize_response(httpx_mock):
    """A Context7 payload above the cap must abort with a clear error,
    and no document must be returned."""
    huge = DEFAULT_MAX_RESPONSE_BYTES + 1
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
        headers={"Content-Length": str(huge)},
        content=b"x" * huge,
    )
    adapter = Context7Adapter()
    with pytest.raises(HttpResponseTooLargeError):
        adapter.fetch("q")


def test_fetch_creates_internal_client_when_none_passed(httpx_mock):
    """Ensure the adapter constructs its own httpx.Client if none supplied."""
    httpx_mock.add_response(
        url=re.compile(r"^https://context7\.com/api/v1/search\?.*"),
        status_code=200,
        json={"results": []},
    )
    adapter = Context7Adapter()
    docs = adapter.fetch("q")
    assert docs == []
    # If we got here without a NoResponseFound error, the adapter built its own
    # httpx.Client and dispatched the request through pytest-httpx's mock transport.
    assert httpx_mock.get_request() is not None
