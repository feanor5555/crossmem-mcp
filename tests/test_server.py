"""Tests for the FastMCP server (query + store tools)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from fastmcp import FastMCP

from crossmem.core.models import Document, Metadata
from crossmem.server import _serialize_document, create_server

if TYPE_CHECKING:
    import pytest


def _make_doc(
    doc_id: str = "abc123",
    content: str = "hello world",
    source_url: str = "https://example.com/x",
    title: str = "Example",
    tags: list[str] | None = None,
) -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=source_url,
            title=title,
            source_type="manual",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=384,
            namespace="default",
            tags=tags or ["python"],
            content_hash="hash",
        ),
    )


def test_serialize_document_excludes_embedding() -> None:
    doc = _make_doc()
    out = _serialize_document(doc)
    assert out["id"] == "abc123"
    assert out["content"] == "hello world"
    assert "embedding" not in out
    assert out["metadata"]["source_url"] == "https://example.com/x"
    assert out["metadata"]["tags"] == ["python"]


def test_create_server_returns_fastmcp_with_tools() -> None:
    store = MagicMock()
    app = create_server(store)
    assert isinstance(app, FastMCP)

    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "query" in names
    assert "store" in names


def test_query_tool_forwards_args_and_serializes() -> None:
    store = MagicMock()
    store.query.return_value = [_make_doc("id-1"), _make_doc("id-2", "more")]
    app = create_server(store)

    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="hello", top_k=5, tags=["python"])

    store.query.assert_called_once_with("hello", top_k=5, tags=["python"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["id"] == "id-1"
    assert "embedding" not in result[0]
    assert result[1]["id"] == "id-2"


def test_query_tool_default_args() -> None:
    store = MagicMock()
    store.query.return_value = []
    app = create_server(store)

    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="anything")

    store.query.assert_called_once_with("anything", top_k=10, tags=None)
    assert result == []


def test_store_tool_forwards_args_and_returns_chunk_ids() -> None:
    store = MagicMock()
    store.store.return_value = ["chunk-1", "chunk-2"]
    app = create_server(store)

    tool = asyncio.run(app.get_tool("store"))
    result = tool.fn(
        content="new content",
        source_url="https://example.com/n",
        title="N",
        source_type="manual",
        namespace="user1",
        tags=["go"],
    )

    store.store.assert_called_once_with(
        content="new content",
        source_url="https://example.com/n",
        title="N",
        source_type="manual",
        namespace="user1",
        tags=["go"],
    )
    assert result == {"ids": ["chunk-1", "chunk-2"], "count": 2}


def test_store_tool_default_args() -> None:
    store = MagicMock()
    store.store.return_value = ["only-id"]
    app = create_server(store)

    tool = asyncio.run(app.get_tool("store"))
    result = tool.fn(content="c", source_url="https://example.com")

    store.store.assert_called_once_with(
        content="c",
        source_url="https://example.com",
        title="",
        source_type="manual",
        namespace="default",
        tags=None,
    )
    assert result == {"ids": ["only-id"], "count": 1}


def test_main_builds_default_store_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """main() routes through build_backend + EmbeddingService and runs app."""
    from crossmem import server as server_module

    fake_backend = MagicMock(name="backend")
    fake_embedder = MagicMock(name="embedder")
    fake_store = MagicMock(name="store")
    fake_app = MagicMock(name="app")

    build_backend_mock = MagicMock(return_value=fake_backend)
    embedder_factory = MagicMock(return_value=fake_embedder)
    store_factory = MagicMock(return_value=fake_store)
    create_server_mock = MagicMock(return_value=fake_app)

    monkeypatch.setattr(server_module, "build_backend", build_backend_mock)
    monkeypatch.setattr(server_module, "EmbeddingService", embedder_factory)
    monkeypatch.setattr(server_module, "KnowledgeStore", store_factory)
    monkeypatch.setattr(server_module, "create_server", create_server_mock)
    monkeypatch.setattr(
        server_module, "load_config", lambda: {"backend": {"name": "sqlite"}}
    )
    # Redirect default DB path to a tmp directory so mkdir does not touch $HOME.
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(tmp_path / "knowledge.db"))

    server_module.main()

    build_backend_mock.assert_called_once()
    embedder_factory.assert_called_once()
    store_factory.assert_called_once_with(fake_backend, fake_embedder)
    create_server_mock.assert_called_once_with(fake_store)
    fake_app.run.assert_called_once()


def test_main_respects_env_var_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from crossmem import server as server_module

    build_backend_mock = MagicMock()
    monkeypatch.setattr(server_module, "build_backend", build_backend_mock)
    monkeypatch.setattr(server_module, "EmbeddingService", MagicMock())
    monkeypatch.setattr(server_module, "KnowledgeStore", MagicMock())
    monkeypatch.setattr(server_module, "create_server", MagicMock())
    monkeypatch.setattr(
        server_module, "load_config", lambda: {"backend": {"name": "sqlite"}}
    )

    custom = tmp_path / "custom.db"
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(custom))

    server_module.main()

    _, kwargs = build_backend_mock.call_args
    assert kwargs["sqlite_path"] == custom


def test_default_sqlite_path_uses_home(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from crossmem import configure as configure_module

    monkeypatch.delenv("CROSSMEM_DB_PATH", raising=False)
    db_path = configure_module._default_sqlite_path()
    assert isinstance(db_path, Path)
    assert db_path == Path.home() / ".crossmem" / "knowledge.db"


def test_default_sqlite_path_respects_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pathlib import Path

    from crossmem import configure as configure_module

    custom = tmp_path / "custom.db"
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(custom))
    db_path = configure_module._default_sqlite_path()
    assert db_path == Path(str(custom))
