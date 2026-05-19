"""Tests for the ``source=`` parameter of the MCP ``query`` tool.

The query tool supports a ``source`` argument that activates a
source-adapter dispatch path:

* No source -> behaves like the cache-only query (existing behaviour).
* Unknown source -> structured ``value_error`` payload that includes
  the available source names so the LLM can retry with a valid name.
* Known source + cache hit -> adapter is NOT called.
* Known source + cache miss -> adapter is invoked, its documents are
  ingested via ``store.store(...)``, the query is retried, and the
  retried hits are returned.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from crossmem.core.models import Document, Metadata
from crossmem.server import create_server
from crossmem.sources.base import SourceBase
from crossmem.sources.registry import SourceRegistry


def _make_doc(
    doc_id: str = "abc123",
    content: str = "hello world",
    source_url: str = "https://example.com/x",
    title: str = "Example",
    source_type: str = "context7",
    tags: list[str] | None = None,
) -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=source_url,
            title=title,
            source_type=source_type,
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=384,
            namespace="default",
            tags=tags or ["python"],
            content_hash="hash",
        ),
    )


class _RecordingAdapter(SourceBase):
    """Source adapter that records every fetch and returns canned documents."""

    def __init__(self, adapter_name: str, docs: list[Document]) -> None:
        self._name = adapter_name
        self._docs = docs
        self.calls: list[tuple[str, dict]] = []

    def name(self) -> str:
        return self._name

    def can_handle(self, uri: str) -> bool:  # pragma: no cover - unused
        return False

    def fetch(self, query: str, **kwargs: object) -> list[Document]:
        self.calls.append((query, dict(kwargs)))
        return list(self._docs)


def test_query_tool_without_source_keeps_existing_behaviour() -> None:
    """``source=None`` -> the registry is never touched, the store is."""
    store = MagicMock()
    store.query.return_value = [_make_doc("a")]
    registry = SourceRegistry()
    registry.register(_RecordingAdapter("context7", []))

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="hello", top_k=5, tags=["python"])

    store.query.assert_called_once_with("hello", top_k=5, tags=["python"])
    assert len(result) == 1
    assert result[0]["id"] == "a"


def test_query_tool_unknown_source_returns_value_error_with_available() -> None:
    store = MagicMock()
    store.query.return_value = []
    registry = SourceRegistry()
    registry.register(_RecordingAdapter("context7", []))
    registry.register(_RecordingAdapter("web", []))

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="anything", source="bogus")

    assert isinstance(result, dict)
    assert result["code"] == "value_error"
    assert "bogus" in result["error"]
    # The known sources are listed so the caller can retry.
    assert "context7" in result["error"]
    assert "web" in result["error"]
    # No fetch was attempted.
    store.store.assert_not_called()


def test_query_tool_source_cache_hit_skips_adapter() -> None:
    """When the cache already has results, the adapter is never called."""
    cached = _make_doc("cached-1")
    store = MagicMock()
    store.query.return_value = [cached]
    adapter = _RecordingAdapter("context7", [_make_doc("from-net")])
    registry = SourceRegistry()
    registry.register(adapter)

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="cached", source="context7")

    # store.query is called exactly once (no retry needed).
    store.query.assert_called_once_with("cached", top_k=10, tags=None)
    # Adapter not invoked, no ingest.
    assert adapter.calls == []
    store.store.assert_not_called()
    # Cache hit is what we return.
    assert len(result) == 1
    assert result[0]["id"] == "cached-1"


def test_query_tool_source_cache_miss_dispatches_to_adapter() -> None:
    """Cache miss -> adapter.fetch + store.store per doc + retry query."""
    fetched_a = _make_doc(
        "net-a",
        content="adapter content A",
        source_url="https://context7.com/a",
        title="A",
    )
    fetched_b = _make_doc(
        "net-b",
        content="adapter content B",
        source_url="https://context7.com/b",
        title="B",
    )
    adapter = _RecordingAdapter("context7", [fetched_a, fetched_b])

    # First query returns []; the retry after ingest returns the fetched docs.
    final_a = _make_doc("net-a", content="adapter content A")
    final_b = _make_doc("net-b", content="adapter content B")
    store = MagicMock()
    store.query.side_effect = [[], [final_a, final_b]]
    store.store.return_value = ["chunk-id"]

    registry = SourceRegistry()
    registry.register(adapter)

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="hooks", top_k=3, source="context7")

    # Adapter was called with the original query.
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == "hooks"

    # Each fetched document is persisted via the store API. We assert on
    # the ingest kwargs that the store needs to chunk + embed + tag.
    assert store.store.call_count == 2
    seen_titles = {call.kwargs["title"] for call in store.store.call_args_list}
    assert seen_titles == {"A", "B"}
    for call in store.store.call_args_list:
        assert call.kwargs["source_type"] == "context7"
        assert call.kwargs["content"]  # non-empty
        assert call.kwargs["source_url"].startswith("https://context7.com/")

    # store.query was called twice: once for the initial cache lookup, once
    # for the retry after ingest.
    assert store.query.call_count == 2

    # The retry returned the cached docs, which is what the tool returns.
    assert {doc["id"] for doc in result} == {"net-a", "net-b"}


def test_query_tool_source_cache_miss_skips_empty_content_docs() -> None:
    """Adapter docs with empty ``content`` are not forwarded to ``store.store``."""
    keeper = _make_doc(
        "keeper",
        content="real content",
        source_url="https://context7.com/keep",
        title="Keep",
    )
    placeholder = _make_doc(
        "placeholder",
        content="",
        source_url="https://context7.com/skip",
        title="Skip",
    )
    adapter = _RecordingAdapter("context7", [placeholder, keeper])
    store = MagicMock()
    store.query.side_effect = [[], [keeper]]
    registry = SourceRegistry()
    registry.register(adapter)

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    tool.fn(query="anything", source="context7")

    # Only the keeper document was ingested.
    assert store.store.call_count == 1
    assert store.store.call_args.kwargs["title"] == "Keep"


def test_query_tool_source_cache_miss_returns_empty_when_adapter_empty() -> None:
    """Adapter returns no docs -> tool returns [] without a retry."""
    adapter = _RecordingAdapter("context7", [])
    store = MagicMock()
    store.query.return_value = []
    registry = SourceRegistry()
    registry.register(adapter)

    app = create_server(store, source_registry=registry)
    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="nothing", source="context7")

    assert result == []
    # Only the initial cache-miss call; no retry needed if nothing was ingested.
    assert store.query.call_count == 1
    store.store.assert_not_called()


def test_query_tool_lists_source_param_in_tool_schema() -> None:
    """The MCP ``query`` tool advertises a ``source`` parameter to clients."""
    store = MagicMock()
    app = create_server(store)
    tool = asyncio.run(app.get_tool("query"))

    schema = tool.parameters
    properties = schema.get("properties", {})
    assert "source" in properties


def test_create_server_uses_default_registry_when_none_supplied() -> None:
    """A factory call without ``source_registry`` falls back to default_registry()."""
    store = MagicMock()
    store.query.return_value = []
    app = create_server(store)
    tool = asyncio.run(app.get_tool("query"))

    # An unknown source produces the structured error, proving a registry is in place.
    result = tool.fn(query="x", source="definitely-not-there")
    assert isinstance(result, dict)
    assert result["code"] == "value_error"
    # And the default registry advertises at least context7.
    assert "context7" in result["error"]
