"""End-to-end source-dispatch integration test for the MCP ``query`` tool.

Covers the DoD of TODO 20.3:

* ``query(..., source="context7")`` -> cache miss -> mock adapter is
  invoked -> doc lands in the DB -> the tool's response contains the
  newly cached doc.
* The second call (same query, same source) hits the cache: the adapter
  is not invoked a second time.

Uses a real :class:`SQLiteBackend` + :class:`KnowledgeStore` so the test
exercises the full store / embed / chunk / query path; only the source
adapter is faked.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document
from crossmem.core.store import KnowledgeStore
from crossmem.server import create_server
from crossmem.sources.base import SourceBase
from crossmem.sources.registry import SourceRegistry
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from pathlib import Path


class _MockContext7Adapter(SourceBase):
    """Stand-in for the Context7 adapter that records every fetch."""

    def __init__(self) -> None:
        self.fetches: list[str] = []

    def name(self) -> str:
        return "context7"

    def can_handle(self, uri: str) -> bool:  # pragma: no cover - unused
        return False

    def fetch(self, query: str, **_kwargs: object) -> list[Document]:
        self.fetches.append(query)
        return [
            Document.from_payload(
                content=(
                    "useState is the React Hook that lets a component "
                    "hold local state. Call useState at the top level "
                    "of your component to declare a state variable."
                ),
                source_url="https://react.dev/reference/react/useState",
                title="useState",
                source_type="context7",
            ),
            Document.from_payload(
                content=(
                    "useEffect lets you synchronize a component with an "
                    "external system. Call useEffect at the top level of "
                    "your component to declare an Effect."
                ),
                source_url="https://react.dev/reference/react/useEffect",
                title="useEffect",
                source_type="context7",
            ),
        ]


def _build_app(tmp_path: Path, adapter: _MockContext7Adapter):
    backend = SQLiteBackend(tmp_path / "knowledge.db")
    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-source-dispatch"))
    registry = SourceRegistry()
    registry.register(adapter)
    return create_server(store, source_registry=registry), store


def test_query_source_cache_miss_invokes_adapter_then_caches(tmp_path: Path) -> None:
    adapter = _MockContext7Adapter()
    app, store = _build_app(tmp_path, adapter)
    tool = asyncio.run(app.get_tool("query"))

    # 1. First call: empty DB -> adapter is invoked, results are ingested,
    #    the retry returns the freshly cached documents.
    first = tool.fn(query="useState", top_k=5, source="context7")

    assert isinstance(first, list)
    assert len(first) >= 1
    titles_first = {doc["metadata"]["title"] for doc in first}
    assert "useState" in titles_first
    # Adapter was called exactly once for the miss.
    assert adapter.fetches == ["useState"]

    # The store has at least one chunk per ingested doc.
    assert store.stats()["document_count"] >= 2

    # 2. Second call with the same query + source: cache hit, no adapter call.
    second = tool.fn(query="useState", top_k=5, source="context7")
    assert isinstance(second, list)
    assert len(second) >= 1
    # Adapter was NOT invoked a second time — the cache served the request.
    assert adapter.fetches == ["useState"]
    titles_second = {doc["metadata"]["title"] for doc in second}
    assert "useState" in titles_second
