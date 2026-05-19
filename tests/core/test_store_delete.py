"""Tests for KnowledgeStore.delete — soft-delete to trash + permanent delete."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crossmem.core.models import Document, Metadata
from crossmem.core.store import KnowledgeStore


def _make_doc(doc_id: str, source_url: str = "https://example.com/x") -> Document:
    """Build a minimal Document for delete tests."""
    return Document(
        id=doc_id,
        content=f"content of {doc_id}",
        embedding=[0.1, 0.2, 0.3],
        metadata=Metadata(
            source_url=source_url,
            title=f"title {doc_id}",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=3,
            namespace="default",
            tags=["python"],
            content_hash=f"hash-{doc_id}",
        ),
    )


def _make_store(
    trash_path: Path | None = None,
) -> tuple[KnowledgeStore, MagicMock, MagicMock]:
    """Build a KnowledgeStore with mocked backend and embedder."""
    backend = MagicMock()
    # Default: get_by_id returns None so unrelated tests get a clean miss
    # instead of a MagicMock proxy (which would look like a hit).
    backend.get_by_id.return_value = None
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    embedder.embed_passage.return_value = [0.0] * 384
    store = KnowledgeStore(backend=backend, embedder=embedder, trash_path=trash_path)
    return store, backend, embedder


def test_delete_by_source_url_writes_each_doc_to_trash(tmp_path: Path) -> None:
    """Soft-delete by source_url writes one JSONL line per matched doc."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    docs = [_make_doc("a"), _make_doc("b")]
    backend.get_by_url.return_value = docs

    count = store.delete(source_url="https://example.com/x")

    assert count == 2
    assert backend.delete.call_count == 2
    backend.delete.assert_any_call("a")
    backend.delete.assert_any_call("b")
    assert trash.exists()

    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    ids_in_trash = {r["doc"]["id"] for r in records}
    assert ids_in_trash == {"a", "b"}
    for r in records:
        assert "deleted_at" in r
        assert isinstance(r["deleted_at"], str) and r["deleted_at"]
        assert r["doc"]["content"].startswith("content of ")
        assert r["doc"]["embedding"] == [0.1, 0.2, 0.3]
        assert r["doc"]["metadata"]["source_url"] == "https://example.com/x"


def test_delete_permanent_does_not_write_to_trash(tmp_path: Path) -> None:
    """permanent=True bypasses the trash entirely."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = [_make_doc("x")]

    count = store.delete(source_url="https://example.com/x", permanent=True)

    assert count == 1
    backend.delete.assert_called_once_with("x")
    assert not trash.exists()


def test_delete_permanent_by_doc_id_calls_backend_delete(tmp_path: Path) -> None:
    """permanent=True with doc_id checks existence then deletes, no trash write."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_id.return_value = _make_doc("abc123")

    count = store.delete(doc_id="abc123", permanent=True)

    assert count == 1
    backend.get_by_id.assert_called_once_with("abc123")
    backend.delete.assert_called_once_with("abc123")
    backend.get_by_url.assert_not_called()
    assert not trash.exists()


def test_delete_permanent_by_unknown_doc_id_is_noop(tmp_path: Path) -> None:
    """permanent=True with an unknown doc_id returns 0 and does not call delete."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_id.return_value = None

    count = store.delete(doc_id="missing", permanent=True)

    assert count == 0
    backend.get_by_id.assert_called_once_with("missing")
    backend.delete.assert_not_called()
    assert not trash.exists()


def test_delete_returns_zero_when_no_docs_match(tmp_path: Path) -> None:
    """Soft-delete with no matching docs writes nothing and returns 0."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = []

    count = store.delete(source_url="https://example.com/missing")

    assert count == 0
    backend.delete.assert_not_called()
    assert not trash.exists()


def test_delete_requires_exactly_one_of_doc_id_or_source_url(tmp_path: Path) -> None:
    """Neither or both raises ValueError."""
    store, _backend, _ = _make_store(trash_path=tmp_path / "trash.jsonl")
    with pytest.raises(ValueError):
        store.delete()
    with pytest.raises(ValueError):
        store.delete(doc_id="x", source_url="https://example.com/x")


def test_delete_soft_delete_by_doc_id_writes_to_trash(tmp_path: Path) -> None:
    """Soft-delete by doc_id resolves via get_by_id and trashes the record."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_id.return_value = _make_doc("abc123")

    count = store.delete(doc_id="abc123")

    assert count == 1
    backend.get_by_id.assert_called_once_with("abc123")
    backend.delete.assert_called_once_with("abc123")
    backend.get_by_url.assert_not_called()
    assert trash.exists()
    record = json.loads(trash.read_text(encoding="utf-8").strip())
    assert record["doc"]["id"] == "abc123"
    assert "deleted_at" in record


def test_delete_soft_delete_by_unknown_doc_id_is_noop(tmp_path: Path) -> None:
    """Soft-delete with an unknown doc_id returns 0 and writes nothing."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_id.return_value = None

    count = store.delete(doc_id="missing")

    assert count == 0
    backend.delete.assert_not_called()
    assert not trash.exists()


def test_delete_appends_to_existing_trash_file(tmp_path: Path) -> None:
    """Trash is append-only — pre-existing lines must remain."""
    trash = tmp_path / "trash.jsonl"
    trash.write_text(
        json.dumps({"deleted_at": "2025-01-01T00:00:00+00:00", "doc": {"id": "old"}})
        + "\n",
        encoding="utf-8",
    )
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = [_make_doc("new")]

    count = store.delete(source_url="https://example.com/x")

    assert count == 1
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["doc"]["id"] == "old"
    assert second["doc"]["id"] == "new"


def test_delete_creates_trash_parent_directory(tmp_path: Path) -> None:
    """Missing parent directory of trash_path is created on first soft-delete."""
    trash = tmp_path / "nested" / "deeper" / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = [_make_doc("a")]

    count = store.delete(source_url="https://example.com/x")

    assert count == 1
    assert trash.exists()


def test_delete_by_source_url_logs_hit_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Soft-delete by source_url must log how many docs matched the URL."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = [_make_doc("a"), _make_doc("b")]

    with caplog.at_level(logging.INFO, logger="crossmem.core.store"):
        store.delete(source_url="https://example.com/x")

    messages = [r.getMessage() for r in caplog.records]
    assert any("2" in m and "https://example.com/x" in m for m in messages)


def test_delete_by_source_url_logs_zero_when_no_match(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A no-match URL delete must still emit a hit-count log with 0."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = []

    with caplog.at_level(logging.INFO, logger="crossmem.core.store"):
        store.delete(source_url="https://example.com/missing")

    messages = [r.getMessage() for r in caplog.records]
    assert any("0" in m and "https://example.com/missing" in m for m in messages)


def test_delete_permanent_by_source_url_logs_hit_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """permanent=True by source_url logs the hit count just like soft-delete."""
    trash = tmp_path / "trash.jsonl"
    store, backend, _ = _make_store(trash_path=trash)
    backend.get_by_url.return_value = [_make_doc("a"), _make_doc("b"), _make_doc("c")]

    with caplog.at_level(logging.INFO, logger="crossmem.core.store"):
        store.delete(source_url="https://example.com/x", permanent=True)

    messages = [r.getMessage() for r in caplog.records]
    assert any("3" in m and "https://example.com/x" in m for m in messages)


def test_delete_default_trash_path_uses_home_crossmem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without trash_path, soft-delete falls back to ~/.crossmem trash."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    store, backend, _ = _make_store(trash_path=None)
    backend.get_by_url.return_value = [_make_doc("a")]

    count = store.delete(source_url="https://example.com/x")

    assert count == 1
    expected = fake_home / ".crossmem" / ".crossmem-trash.jsonl"
    assert expected.exists()
    record = json.loads(expected.read_text(encoding="utf-8").strip())
    assert record["doc"]["id"] == "a"
