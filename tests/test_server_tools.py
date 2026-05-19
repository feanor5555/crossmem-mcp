"""Tests for delete, status, and cleanup MCP tools registered by create_server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar
from unittest.mock import MagicMock, patch

import pytest

from crossmem.core.models import Document, Metadata
from crossmem.server import DESTRUCTIVE_MCP_ENV, create_server

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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


def _store_returning(docs: list[Document]) -> MagicMock:
    """MagicMock store whose lookup APIs return ``docs`` and ``delete`` returns 1.

    Tag-mode cleanup (TODO 26.1) drives :meth:`KnowledgeStore.find_by_tag`,
    semantic-mode cleanup drives :meth:`KnowledgeStore.query`; we wire both
    so a single helper covers every cleanup-tool test variant. ``find_by_tag``
    returns a fresh iterator on every call so repeated invocations within a
    single test do not exhaust the generator.
    """
    store = MagicMock()
    store.query.return_value = docs
    store.find_by_tag.side_effect = lambda _tag: iter(docs)
    store.delete.return_value = 1
    return store


T = TypeVar("T")


def _run_with_trash_path(trash_path: Path, fn: Callable[[], T]) -> T:
    """Run ``fn`` with the cleanup module's default trash path pinned."""
    with patch("crossmem.cleanup._default_trash_path", lambda: trash_path):
        return fn()


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


# -------- delete tool --------


def test_delete_tool_by_doc_id_forwards_args() -> None:
    store = MagicMock()
    store.delete.return_value = 1
    app = create_server(store)

    tool = asyncio.run(app.get_tool("delete"))
    result = tool.fn(doc_id="abc123")

    store.delete.assert_called_once_with(
        doc_id="abc123", source_url=None, permanent=False
    )
    assert result == {"deleted": 1}


def test_delete_tool_by_source_url_forwards_args() -> None:
    store = MagicMock()
    store.delete.return_value = 2
    app = create_server(store)

    tool = asyncio.run(app.get_tool("delete"))
    result = tool.fn(source_url="https://example.com/x")

    store.delete.assert_called_once_with(
        doc_id=None, source_url="https://example.com/x", permanent=False
    )
    assert result == {"deleted": 2}


def test_delete_tool_permanent_forwards_flag() -> None:
    store = MagicMock()
    store.delete.return_value = 1
    app = create_server(store)

    tool = asyncio.run(app.get_tool("delete"))
    result = tool.fn(doc_id="abc123", permanent=True)

    store.delete.assert_called_once_with(
        doc_id="abc123", source_url=None, permanent=True
    )
    assert result == {"deleted": 1}


# -------- status tool --------


def test_status_tool_returns_store_stats() -> None:
    store = MagicMock()
    expected = {
        "document_count": 42,
        "db_size_bytes": 1024,
        "top_tags": [("python", 10)],
        "backend": "sqlite",
    }
    store.stats.return_value = expected
    app = create_server(store)

    tool = asyncio.run(app.get_tool("status"))
    result = tool.fn()

    store.stats.assert_called_once_with()
    assert result == expected


# -------- cleanup tool --------


def test_cleanup_tool_default_mode_is_tag(monkeypatch) -> None:
    """The default ``mode`` parameter is ``"tag"``."""
    from crossmem import server as server_module

    store = MagicMock()
    captured: dict = {}

    def fake_cleanup(s, q, *, dry_run, mode):
        captured["store"] = s
        captured["query"] = q
        captured["dry_run"] = dry_run
        captured["mode"] = mode
        from crossmem.cleanup import CleanupResult

        return CleanupResult(mode=mode)

    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(store)

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="anything")

    assert captured == {
        "store": store,
        "query": "anything",
        "dry_run": True,
        "mode": "tag",
    }
    assert result == {"matched": [], "deleted": 0, "dry_run": True, "mode": "tag"}


def test_cleanup_tool_tag_dry_run_delegates_and_returns_preview(monkeypatch) -> None:
    """``mode="tag"`` + ``dry_run=True`` returns matched ids, deleted=0."""
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    store = MagicMock()
    fake_cleanup = MagicMock(
        return_value=CleanupResult(mode="tag", previewed_ids=["id-1", "id-2"])
    )
    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(store)

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="python", dry_run=True, mode="tag")

    fake_cleanup.assert_called_once_with(store, "python", dry_run=True, mode="tag")
    store.delete.assert_not_called()
    assert result == {
        "matched": ["id-1", "id-2"],
        "deleted": 0,
        "dry_run": True,
        "mode": "tag",
    }


def test_cleanup_tool_semantic_dry_run_delegates(monkeypatch) -> None:
    """``mode="semantic"`` is forwarded verbatim to the cleanup API."""
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    store = MagicMock()
    fake_cleanup = MagicMock(
        return_value=CleanupResult(mode="semantic", previewed_ids=["x"])
    )
    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(store)

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="outdated", dry_run=True, mode="semantic")

    fake_cleanup.assert_called_once_with(
        store, "outdated", dry_run=True, mode="semantic"
    )
    assert result == {
        "matched": ["x"],
        "deleted": 0,
        "dry_run": True,
        "mode": "semantic",
    }


def test_cleanup_tool_tag_executes_when_dry_run_false(tmp_path, monkeypatch) -> None:
    """``mode="tag"`` + ``dry_run=False`` populates trash and reports deletions.

    Behavior parity: invoking the MCP tool yields the same result as calling
    ``crossmem.cleanup.cleanup`` directly with the same arguments. The
    destructive path requires ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``.
    """
    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, "1")
    from crossmem.cleanup import cleanup as direct_cleanup

    trash_via_tool = tmp_path / "trash_tool.jsonl"
    trash_via_api = tmp_path / "trash_api.jsonl"

    matches = [
        _make_doc("id-1", source_url="https://example.com/a", tags=["python"]),
        _make_doc("id-2", source_url="https://example.com/b", tags=["python"]),
    ]

    store_tool = _store_returning(matches)
    store_api = _store_returning(matches)

    app = create_server(store_tool)
    tool = asyncio.run(app.get_tool("cleanup"))

    via_tool = _run_with_trash_path(
        trash_via_tool, lambda: tool.fn(query="python", dry_run=False, mode="tag")
    )
    via_api = _run_with_trash_path(
        trash_via_api,
        lambda: direct_cleanup(store_api, "python", dry_run=False, mode="tag"),
    )

    # MCP tool result mirrors the direct cleanup() API.
    assert via_tool == {
        "matched": via_api.previewed_ids,
        "deleted": len(via_api.deleted_ids),
        "dry_run": False,
        "mode": "tag",
    }
    assert via_tool["deleted"] == 2
    assert set(via_tool["matched"]) == {"id-1", "id-2"}

    # Trash side-effect: both calls produced identical JSONL line counts.
    assert trash_via_tool.exists()
    assert trash_via_api.exists()
    assert _count_lines(trash_via_tool) == _count_lines(trash_via_api) == 2

    # store.delete was called permanently, by doc_id, for every match.
    assert store_tool.delete.call_count == 2
    for call in store_tool.delete.call_args_list:
        assert call.kwargs.get("permanent") is True
        assert call.kwargs.get("doc_id") in {"id-1", "id-2"}


def test_cleanup_tool_semantic_executes_and_matches_direct_api(
    tmp_path, monkeypatch
) -> None:
    """``mode="semantic"`` delegates and yields the same result as the direct API.

    Destructive path requires ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``.
    """
    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, "1")
    from crossmem.cleanup import cleanup as direct_cleanup

    trash_via_tool = tmp_path / "trash_tool.jsonl"
    trash_via_api = tmp_path / "trash_api.jsonl"

    matches = [
        _make_doc(f"sem-{i}", source_url=f"https://example.com/{i}", tags=["misc"])
        for i in range(3)
    ]

    store_tool = _store_returning(matches)
    store_api = _store_returning(matches)

    app = create_server(store_tool)
    tool = asyncio.run(app.get_tool("cleanup"))

    via_tool = _run_with_trash_path(
        trash_via_tool,
        lambda: tool.fn(query="outdated docs", dry_run=False, mode="semantic"),
    )
    via_api = _run_with_trash_path(
        trash_via_api,
        lambda: direct_cleanup(
            store_api, "outdated docs", dry_run=False, mode="semantic"
        ),
    )

    assert via_tool == {
        "matched": via_api.previewed_ids,
        "deleted": len(via_api.deleted_ids),
        "dry_run": False,
        "mode": "semantic",
    }
    assert via_tool["deleted"] == 3
    assert set(via_tool["matched"]) == {"sem-0", "sem-1", "sem-2"}

    # store.query was called WITHOUT a tags=[query] pre-filter (semantic mode).
    assert store_tool.query.call_args.kwargs.get("tags") in (None,)
    assert "tags" not in store_tool.query.call_args.kwargs or (
        store_tool.query.call_args.kwargs.get("tags") is None
    )

    assert _count_lines(trash_via_tool) == _count_lines(trash_via_api) == 3


# -------- registration --------


def test_create_server_registers_delete_status_cleanup() -> None:
    store = MagicMock()
    app = create_server(store)
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert {"delete", "status", "cleanup"}.issubset(names)


# -------- export tool --------


def test_export_tool_forwards_args_default_zip() -> None:
    store = MagicMock()
    store.export.return_value = 7
    app = create_server(store)

    tool = asyncio.run(app.get_tool("export"))
    result = tool.fn(path="/tmp/out.zip")

    assert store.export.call_count == 1
    call_args, call_kwargs = store.export.call_args
    # First positional arg is a Path
    from pathlib import Path

    assert call_args[0] == Path("/tmp/out.zip")
    assert call_kwargs.get("format") == "zip"
    assert result == {"exported": 7, "path": "/tmp/out.zip"}


def test_export_tool_forwards_jsonl_format() -> None:
    store = MagicMock()
    store.export.return_value = 3
    app = create_server(store)

    tool = asyncio.run(app.get_tool("export"))
    result = tool.fn(path="/tmp/out.jsonl", format="jsonl")

    _, call_kwargs = store.export.call_args
    assert call_kwargs.get("format") == "jsonl"
    assert result == {"exported": 3, "path": "/tmp/out.jsonl"}


# -------- import_data tool --------


def test_import_data_tool_forwards_path() -> None:
    store = MagicMock()
    store.import_data.return_value = 5
    app = create_server(store)

    tool = asyncio.run(app.get_tool("import_data"))
    result = tool.fn(path="/tmp/in.zip")

    from pathlib import Path

    store.import_data.assert_called_once_with(Path("/tmp/in.zip"))
    assert result == {"imported": 5}


def test_create_server_registers_export_and_import() -> None:
    store = MagicMock()
    app = create_server(store)
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert {"export", "import_data"}.issubset(names)


# -------- ValueError -> structured error payload --------


def test_query_tool_returns_structured_error_on_value_error() -> None:
    store = MagicMock()
    store.query.side_effect = ValueError("bad query")
    app = create_server(store)

    tool = asyncio.run(app.get_tool("query"))
    result = tool.fn(query="x")

    assert result == {"error": "bad query", "code": "value_error"}


def test_delete_tool_returns_structured_error_on_value_error() -> None:
    store = MagicMock()
    store.delete.side_effect = ValueError(
        "delete() requires exactly one of doc_id or source_url"
    )
    app = create_server(store)

    tool = asyncio.run(app.get_tool("delete"))
    result = tool.fn()

    assert result == {
        "error": "delete() requires exactly one of doc_id or source_url",
        "code": "value_error",
    }


def test_import_data_tool_returns_structured_error_on_value_error() -> None:
    store = MagicMock()
    store.import_data.side_effect = ValueError("missing EOF marker")
    app = create_server(store)

    tool = asyncio.run(app.get_tool("import_data"))
    result = tool.fn(path="/tmp/in.jsonl")

    assert result == {"error": "missing EOF marker", "code": "value_error"}


def test_cleanup_tool_returns_structured_error_on_value_error(monkeypatch) -> None:
    from crossmem import server as server_module

    store = MagicMock()

    def boom(*_a, **_kw):
        raise ValueError("unknown cleanup mode: 'bogus'")

    monkeypatch.setattr(server_module, "cleanup_op", boom)
    app = create_server(store)

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="x", mode="bogus")

    assert result == {
        "error": "unknown cleanup mode: 'bogus'",
        "code": "value_error",
    }


# -------- empty_trash tool --------


def test_empty_trash_tool_registered() -> None:
    """``empty_trash`` shows up in the tool list."""
    app = create_server(MagicMock())
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "empty_trash" in names


def test_empty_trash_tool_default_ttl(monkeypatch) -> None:
    """Default invocation forwards ttl_days=30 and returns the removed count.

    Destructive path requires ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``.
    """
    from crossmem import server as server_module

    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, "1")
    captured: dict[str, object] = {}

    def fake_empty(trash_path=None, *, ttl_days=30):
        captured["trash_path"] = trash_path
        captured["ttl_days"] = ttl_days
        return 5

    monkeypatch.setattr(server_module, "empty_trash_op", fake_empty)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("empty_trash"))
    result = tool.fn()
    assert captured["ttl_days"] == 30
    assert result == {"removed": 5}


def test_empty_trash_tool_forwards_ttl_zero(monkeypatch) -> None:
    """``ttl_days=0`` is forwarded verbatim (DSGVO purge through MCP).

    Destructive path requires ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``.
    """
    from crossmem import server as server_module

    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, "1")
    captured: dict[str, int] = {}

    def fake_empty(trash_path=None, *, ttl_days=30):
        captured["ttl_days"] = ttl_days
        return 9

    monkeypatch.setattr(server_module, "empty_trash_op", fake_empty)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("empty_trash"))
    result = tool.fn(ttl_days=0)
    assert captured["ttl_days"] == 0
    assert result == {"removed": 9}


# -------- restore_from_trash tool --------


def test_restore_from_trash_tool_registered() -> None:
    app = create_server(MagicMock())
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "restore_from_trash" in names


def test_restore_from_trash_tool_returns_serialized_doc(monkeypatch) -> None:
    """Successful restore returns the doc payload (no embedding)."""
    from crossmem import server as server_module

    store = MagicMock()
    doc = _make_doc("abc123", content="restored content")

    captured: dict[str, object] = {}

    def fake_restore(s, doc_id, trash_path=None):
        captured["store"] = s
        captured["doc_id"] = doc_id
        return doc

    monkeypatch.setattr(server_module, "restore_from_trash_op", fake_restore)
    app = create_server(store)

    tool = asyncio.run(app.get_tool("restore_from_trash"))
    result = tool.fn(doc_id="abc123")
    assert captured["store"] is store
    assert captured["doc_id"] == "abc123"
    assert result["id"] == "abc123"
    assert result["content"] == "restored content"
    assert "embedding" not in result


def test_restore_from_trash_tool_unknown_id_returns_value_error(monkeypatch) -> None:
    """Unknown id -> structured ``value_error`` payload, never a traceback."""
    from crossmem import server as server_module

    def boom(_store, doc_id, trash_path=None):
        raise ValueError(f"doc_id not found in trash: {doc_id!r}")

    monkeypatch.setattr(server_module, "restore_from_trash_op", boom)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("restore_from_trash"))
    result = tool.fn(doc_id="missing")
    assert result["code"] == "value_error"
    assert "missing" in result["error"]


# -------- destructive-MCP gate (task 26.5) --------
#
# Default posture: a compromised or prompt-injected LLM must NOT be able to
# permanently delete documents or empty the soft-delete trash through MCP.
# Both destructive paths require the operator to opt in explicitly via
# ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1`` in the environment that launches the
# MCP server. The CLI (``crossmem trash empty``) is unaffected — humans keep
# full control without the gate.


def test_cleanup_tool_dry_run_false_blocked_without_env_switch(monkeypatch) -> None:
    """Without the env-switch, ``dry_run=False`` is forced to ``True``.

    The underlying ``cleanup_op`` must still be called (preview is useful) but
    with ``dry_run=True`` and the response surfaces ``forced_dry_run=True`` so
    the caller can detect the downgrade. No deletion side-effect occurs.
    """
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    monkeypatch.delenv(DESTRUCTIVE_MCP_ENV, raising=False)
    captured: dict[str, object] = {}

    def fake_cleanup(s, q, *, dry_run, mode):
        captured["dry_run"] = dry_run
        captured["mode"] = mode
        return CleanupResult(mode=mode, previewed_ids=["x"])

    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    store = MagicMock()
    app = create_server(store)

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="python", dry_run=False, mode="tag")

    # cleanup_op was called, but with dry_run forced to True.
    assert captured == {"dry_run": True, "mode": "tag"}
    # Response signals the downgrade.
    assert result["dry_run"] is True
    assert result["forced_dry_run"] is True
    assert result["matched"] == ["x"]
    assert result["deleted"] == 0


def test_cleanup_tool_dry_run_true_works_without_env_switch(monkeypatch) -> None:
    """Preview (``dry_run=True``) remains freely available — no gate needed."""
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    monkeypatch.delenv(DESTRUCTIVE_MCP_ENV, raising=False)
    fake_cleanup = MagicMock(
        return_value=CleanupResult(mode="tag", previewed_ids=["a", "b"])
    )
    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="python", dry_run=True, mode="tag")

    fake_cleanup.assert_called_once()
    assert fake_cleanup.call_args.kwargs["dry_run"] is True
    assert result["matched"] == ["a", "b"]
    assert result["deleted"] == 0
    # The downgrade marker only appears when a destructive call was downgraded.
    assert "forced_dry_run" not in result


@pytest.mark.parametrize("env_value", ["0", "false", "", "no"])
def test_cleanup_tool_falsey_env_does_not_unlock_destructive(
    env_value, monkeypatch
) -> None:
    """Only ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1`` opens the gate."""
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, env_value)
    captured: dict[str, object] = {}

    def fake_cleanup(s, q, *, dry_run, mode):
        captured["dry_run"] = dry_run
        return CleanupResult(mode=mode, previewed_ids=[])

    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="python", dry_run=False, mode="tag")

    assert captured["dry_run"] is True
    assert result["forced_dry_run"] is True


def test_cleanup_tool_dry_run_false_with_env_switch_executes(monkeypatch) -> None:
    """With ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1`` the destructive call goes through."""
    from crossmem import server as server_module
    from crossmem.cleanup import CleanupResult

    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, "1")
    fake_cleanup = MagicMock(
        return_value=CleanupResult(mode="tag", previewed_ids=["x"], deleted_ids=["x"])
    )
    monkeypatch.setattr(server_module, "cleanup_op", fake_cleanup)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("cleanup"))
    result = tool.fn(query="python", dry_run=False, mode="tag")

    assert fake_cleanup.call_args.kwargs["dry_run"] is False
    assert result["dry_run"] is False
    assert "forced_dry_run" not in result
    assert result["deleted"] == 1


def test_empty_trash_tool_no_op_without_env_switch(monkeypatch) -> None:
    """Without the env-switch ``empty_trash`` is a no-op with a clear marker.

    The underlying ``empty_trash_op`` must NOT be called — a no-op response is
    returned with ``removed=0`` and ``blocked=True`` plus a hint string.
    """
    from crossmem import server as server_module

    monkeypatch.delenv(DESTRUCTIVE_MCP_ENV, raising=False)
    sentinel = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(server_module, "empty_trash_op", sentinel)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("empty_trash"))
    result = tool.fn(ttl_days=0)

    sentinel.assert_not_called()
    assert result["removed"] == 0
    assert result["blocked"] is True
    assert DESTRUCTIVE_MCP_ENV in result.get("hint", "")


@pytest.mark.parametrize("env_value", ["0", "false", "", "no"])
def test_empty_trash_tool_falsey_env_stays_blocked(env_value, monkeypatch) -> None:
    """Only ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1`` opens the gate."""
    from crossmem import server as server_module

    monkeypatch.setenv(DESTRUCTIVE_MCP_ENV, env_value)
    sentinel = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(server_module, "empty_trash_op", sentinel)
    app = create_server(MagicMock())

    tool = asyncio.run(app.get_tool("empty_trash"))
    result = tool.fn(ttl_days=0)

    sentinel.assert_not_called()
    assert result["blocked"] is True


def test_destructive_mcp_env_constant_name() -> None:
    """The env-var name is stable: changing it would break operator configs."""
    assert DESTRUCTIVE_MCP_ENV == "CROSSMEM_ALLOW_DESTRUCTIVE_MCP"
