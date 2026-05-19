"""Tests for the cleanup, trash, and restore module."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from crossmem.cleanup import (
    CleanupResult,
    TrashEntry,
    cleanup,
    empty_trash,
    list_trash,
    restore_from_trash,
)
from crossmem.core.models import Document, Metadata
from crossmem.core.store import KnowledgeStore


def _make_doc(doc_id: str, tags: list[str] | None = None) -> Document:
    """Build a minimal Document for cleanup tests."""
    return Document(
        id=doc_id,
        content=f"content of {doc_id}",
        embedding=[0.1, 0.2, 0.3],
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=f"title {doc_id}",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=3,
            namespace="default",
            tags=list(tags) if tags else [],
            content_hash=f"hash-{doc_id}",
        ),
    )


def _make_store(query_result: list[Document]) -> MagicMock:
    """Build a MagicMock store.

    ``query_result`` drives both the semantic-mode :meth:`KnowledgeStore.query`
    path and the tag-mode :meth:`KnowledgeStore.find_by_tag` path (which
    returns an iterator per the native-index contract added by TODO 26.1).
    Tag-mode cleanup no longer calls ``store.query`` at all.
    """
    store = MagicMock()
    store.query.return_value = query_result
    store.find_by_tag.side_effect = lambda _tag: iter(query_result)
    return store


def _write_trash_lines(trash_path: Path, records: list[dict]) -> None:
    trash_path.parent.mkdir(parents=True, exist_ok=True)
    trash_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _record_for(doc: Document, deleted_at: str) -> dict:
    return {"deleted_at": deleted_at, "doc": asdict(doc)}


# ---------------------------------------------------------------------------
# cleanup() — tag mode
# ---------------------------------------------------------------------------


def test_cleanup_tag_dry_run_previews_but_does_not_delete(tmp_path: Path) -> None:
    """dry_run=True with mode='tag' returns matches, deletes nothing."""
    trash = tmp_path / "trash.jsonl"
    a = _make_doc("a", tags=["python"])
    b = _make_doc("b", tags=["python", "asyncio"])
    store = _make_store([a, b])

    result = cleanup(store, "python", dry_run=True, mode="tag", trash_path=trash)

    assert isinstance(result, CleanupResult)
    assert result.mode == "tag"
    assert set(result.previewed_ids) == {"a", "b"}
    assert result.deleted_ids == []
    store.delete.assert_not_called()
    assert not trash.exists()


def test_cleanup_tag_passes_query_through_to_find_by_tag(tmp_path: Path) -> None:
    """cleanup tag-mode forwards the tag verbatim to ``find_by_tag``.

    Backends are responsible for native exact-tag filtering (TODO 26.1);
    cleanup no longer post-filters by ``tag in doc.metadata.tags``.
    """
    trash = tmp_path / "trash.jsonl"
    a = _make_doc("a", tags=["python"])
    store = _make_store([a])

    result = cleanup(store, "python", dry_run=True, mode="tag", trash_path=trash)

    store.find_by_tag.assert_called_once_with("python")
    store.query.assert_not_called()
    assert result.previewed_ids == ["a"]


def test_cleanup_tag_writes_each_match_to_trash_and_deletes(tmp_path: Path) -> None:
    """dry_run=False writes JSONL lines and calls store.delete for each match."""
    trash = tmp_path / "trash.jsonl"
    a = _make_doc("a", tags=["python"])
    b = _make_doc("b", tags=["python", "asyncio"])
    store = _make_store([a, b])

    result = cleanup(store, "python", dry_run=False, mode="tag", trash_path=trash)

    assert set(result.deleted_ids) == {"a", "b"}
    assert set(result.previewed_ids) == {"a", "b"}

    # Both docs are permanently deleted from the store (no double-trash).
    assert store.delete.call_count == 2
    for call in store.delete.call_args_list:
        assert call.kwargs.get("permanent") is True
        assert call.kwargs.get("doc_id") in {"a", "b"}

    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["doc"]["id"] for r in records} == {"a", "b"}
    for r in records:
        assert "deleted_at" in r and r["deleted_at"]
        assert r["doc"]["embedding"] == [0.1, 0.2, 0.3]


def test_cleanup_tag_no_matches_does_not_create_trash(tmp_path: Path) -> None:
    """No matches: trash file is not created and delete is not called."""
    trash = tmp_path / "trash.jsonl"
    store = _make_store([])  # query returns nothing

    result = cleanup(store, "rust", dry_run=False, mode="tag", trash_path=trash)

    assert result.previewed_ids == []
    assert result.deleted_ids == []
    store.delete.assert_not_called()
    assert not trash.exists()


# ---------------------------------------------------------------------------
# cleanup() — semantic mode
# ---------------------------------------------------------------------------


def test_cleanup_semantic_uses_query_top_k_and_deletes(tmp_path: Path) -> None:
    """mode='semantic' returns top-k by relevance and deletes them."""
    trash = tmp_path / "trash.jsonl"
    docs = [_make_doc(f"d{i}", tags=["misc"]) for i in range(3)]
    store = _make_store(docs)

    result = cleanup(
        store,
        "outdated docs",
        dry_run=False,
        mode="semantic",
        top_k=3,
        trash_path=trash,
    )

    # store.query was invoked with top_k=3 (semantic mode)
    store.query.assert_called_once()
    call_kwargs = store.query.call_args.kwargs
    assert call_kwargs.get("query") == "outdated docs"
    assert call_kwargs.get("top_k") == 3
    # No tag pre-filter for semantic mode
    assert "tags" not in call_kwargs or call_kwargs.get("tags") is None

    assert set(result.deleted_ids) == {"d0", "d1", "d2"}
    assert store.delete.call_count == 3
    assert trash.exists()
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_cleanup_semantic_dry_run_does_not_delete(tmp_path: Path) -> None:
    trash = tmp_path / "trash.jsonl"
    docs = [_make_doc("d1"), _make_doc("d2")]
    store = _make_store(docs)

    result = cleanup(
        store, "anything", dry_run=True, mode="semantic", top_k=2, trash_path=trash
    )

    assert result.deleted_ids == []
    assert set(result.previewed_ids) == {"d1", "d2"}
    store.delete.assert_not_called()
    assert not trash.exists()


def test_cleanup_unknown_mode_raises(tmp_path: Path) -> None:
    store = _make_store([])
    with pytest.raises(ValueError):
        cleanup(store, "x", mode="banana", trash_path=tmp_path / "t.jsonl")


def test_cleanup_appends_to_existing_trash(tmp_path: Path) -> None:
    """Pre-existing trash lines must remain after cleanup writes new ones."""
    trash = tmp_path / "trash.jsonl"
    trash.write_text(
        json.dumps({"deleted_at": "2024-01-01T00:00:00+00:00", "doc": {"id": "old"}})
        + "\n",
        encoding="utf-8",
    )
    store = _make_store([_make_doc("new", tags=["python"])])

    cleanup(store, "python", dry_run=False, mode="tag", trash_path=trash)

    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["doc"]["id"] == "old"
    assert json.loads(lines[1])["doc"]["id"] == "new"


def test_cleanup_default_trash_path_uses_home_crossmem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    store = _make_store([_make_doc("a", tags=["python"])])
    cleanup(store, "python", dry_run=False, mode="tag")

    expected = fake_home / ".crossmem" / ".crossmem-trash.jsonl"
    assert expected.exists()


# ---------------------------------------------------------------------------
# empty_trash()
# ---------------------------------------------------------------------------


def test_empty_trash_missing_file_returns_zero(tmp_path: Path) -> None:
    trash = tmp_path / "absent.jsonl"
    assert empty_trash(trash) == 0
    assert not trash.exists()


def test_empty_trash_keeps_recent_drops_old_with_freezegun(tmp_path: Path) -> None:
    """A doc deleted 31 days ago is removed; one deleted 29 days ago is kept."""
    trash = tmp_path / "trash.jsonl"
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    old_dt = (now - timedelta(days=31)).isoformat()
    fresh_dt = (now - timedelta(days=29)).isoformat()
    _write_trash_lines(
        trash,
        [
            _record_for(_make_doc("old"), old_dt),
            _record_for(_make_doc("fresh"), fresh_dt),
        ],
    )

    with freeze_time(now):
        removed = empty_trash(trash, ttl_days=30)

    assert removed == 1
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["doc"]["id"] == "fresh"


def test_empty_trash_accepts_explicit_now(tmp_path: Path) -> None:
    """The ``now`` kwarg overrides the wall clock without freezegun."""
    trash = tmp_path / "trash.jsonl"
    pinned = datetime(2026, 5, 10, tzinfo=timezone.utc)
    old_dt = (pinned - timedelta(days=40)).isoformat()
    fresh_dt = (pinned - timedelta(days=10)).isoformat()
    _write_trash_lines(
        trash,
        [
            _record_for(_make_doc("old"), old_dt),
            _record_for(_make_doc("fresh"), fresh_dt),
        ],
    )

    removed = empty_trash(trash, ttl_days=30, now=pinned)

    assert removed == 1
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert {json.loads(line)["doc"]["id"] for line in lines} == {"fresh"}


def test_empty_trash_ttl_zero_removes_all(tmp_path: Path) -> None:
    """ttl_days=0 wipes every trash entry (DSGVO)."""
    trash = tmp_path / "trash.jsonl"
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    _write_trash_lines(
        trash,
        [
            _record_for(_make_doc("a"), now.isoformat()),
            _record_for(_make_doc("b"), (now - timedelta(days=1)).isoformat()),
        ],
    )

    removed = empty_trash(trash, ttl_days=0, now=now)

    assert removed == 2
    assert trash.read_text(encoding="utf-8") == ""


def test_empty_trash_keeps_corrupt_line_unchanged(tmp_path: Path) -> None:
    """Lines that fail to parse are kept so the user can inspect them."""
    trash = tmp_path / "trash.jsonl"
    trash.write_text("not-json\n", encoding="utf-8")
    removed = empty_trash(trash, ttl_days=30, now=datetime.now(timezone.utc))
    assert removed == 0
    assert trash.read_text(encoding="utf-8").strip() == "not-json"


# ---------------------------------------------------------------------------
# restore_from_trash()
# ---------------------------------------------------------------------------


def test_restore_from_trash_re_inserts_doc_and_drops_line(tmp_path: Path) -> None:
    trash = tmp_path / "trash.jsonl"
    doc_a = _make_doc("a", tags=["python"])
    doc_b = _make_doc("b", tags=["rust"])
    _write_trash_lines(
        trash,
        [
            _record_for(doc_a, "2026-01-01T00:00:00+00:00"),
            _record_for(doc_b, "2026-01-02T00:00:00+00:00"),
        ],
    )
    store = MagicMock()

    restored = restore_from_trash(store, "a", trash_path=trash)

    assert restored.id == "a"
    assert restored.content == doc_a.content
    assert restored.embedding == doc_a.embedding
    assert restored.metadata.tags == ("python",)

    # Restore went through the public KnowledgeStore.restore API.
    store.restore.assert_called_once()
    (passed,) = store.restore.call_args.args
    assert passed.id == "a"

    # Trash file now has only the unrestored doc.
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["doc"]["id"] == "b"


def test_restore_from_trash_unknown_id_raises(tmp_path: Path) -> None:
    trash = tmp_path / "trash.jsonl"
    _write_trash_lines(
        trash,
        [_record_for(_make_doc("a"), "2026-01-01T00:00:00+00:00")],
    )
    store = MagicMock()

    with pytest.raises(ValueError):
        restore_from_trash(store, "missing", trash_path=trash)

    # Trash is unchanged when restore fails.
    lines = trash.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["doc"]["id"] == "a"
    store.restore.assert_not_called()


def test_restore_from_trash_missing_file_raises(tmp_path: Path) -> None:
    """If the trash file does not exist, restoring any id raises ValueError."""
    trash = tmp_path / "absent.jsonl"
    store = MagicMock()
    with pytest.raises(ValueError):
        restore_from_trash(store, "x", trash_path=trash)


def test_restore_from_trash_skips_corrupt_lines(tmp_path: Path) -> None:
    """Corrupt JSON or missing 'doc' is preserved verbatim, search continues."""
    trash = tmp_path / "trash.jsonl"
    target = _make_doc("a", tags=["python"])
    valid_record = json.dumps(_record_for(target, "2026-01-01T00:00:00+00:00"))
    trash.write_text(
        "not-json\n"
        + json.dumps({"deleted_at": "2026-01-01T00:00:00+00:00"})  # no 'doc'
        + "\n"
        + valid_record
        + "\n",
        encoding="utf-8",
    )
    store = MagicMock()

    restored = restore_from_trash(store, "a", trash_path=trash)

    assert restored.id == "a"
    remaining = trash.read_text(encoding="utf-8").splitlines()
    assert remaining == [
        "not-json",
        json.dumps({"deleted_at": "2026-01-01T00:00:00+00:00"}),
    ]


def test_restore_from_trash_missing_metadata_raises(tmp_path: Path) -> None:
    """Trash record with the right id but missing required Document fields."""
    trash = tmp_path / "trash.jsonl"
    bad = {
        "deleted_at": "2026-01-01T00:00:00+00:00",
        "doc": {"id": "a"},  # no content/embedding/metadata
    }
    trash.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    store = MagicMock()

    with pytest.raises(ValueError):
        restore_from_trash(store, "a", trash_path=trash)


def test_restore_from_trash_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    trash = fake_home / ".crossmem" / ".crossmem-trash.jsonl"
    _write_trash_lines(
        trash,
        [_record_for(_make_doc("a", tags=["python"]), "2026-01-01T00:00:00+00:00")],
    )
    store = MagicMock()

    restored = restore_from_trash(store, "a")
    assert restored.id == "a"
    assert trash.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# list_trash()
# ---------------------------------------------------------------------------


def test_list_trash_missing_file_returns_empty(tmp_path: Path) -> None:
    """No trash file at all -> empty list, no errors."""
    trash = tmp_path / "absent.jsonl"
    assert list_trash(trash) == []
    assert not trash.exists()


def test_list_trash_returns_entries_in_file_order(tmp_path: Path) -> None:
    """Each line yields one ``TrashEntry`` with id, deleted_at, source_url, title."""
    trash = tmp_path / "trash.jsonl"
    a = _make_doc("a", tags=["python"])
    b = _make_doc("b", tags=["rust"])
    _write_trash_lines(
        trash,
        [
            _record_for(a, "2026-01-01T00:00:00+00:00"),
            _record_for(b, "2026-01-02T00:00:00+00:00"),
        ],
    )

    entries = list_trash(trash)

    assert [e.doc_id for e in entries] == ["a", "b"]
    assert isinstance(entries[0], TrashEntry)
    assert entries[0].deleted_at == "2026-01-01T00:00:00+00:00"
    assert entries[0].source_url == "https://example.com/a"
    assert entries[0].title == "title a"
    assert entries[1].deleted_at == "2026-01-02T00:00:00+00:00"


def test_list_trash_skips_corrupt_lines(tmp_path: Path) -> None:
    """Unparseable JSON or records without a doc payload are silently dropped."""
    trash = tmp_path / "trash.jsonl"
    target = _make_doc("a", tags=["python"])
    valid = json.dumps(_record_for(target, "2026-01-01T00:00:00+00:00"))
    trash.write_text(
        "not-json\n"
        + json.dumps({"deleted_at": "2026-01-01T00:00:00+00:00"})  # no 'doc'
        + "\n"
        + valid
        + "\n",
        encoding="utf-8",
    )

    entries = list_trash(trash)
    assert [e.doc_id for e in entries] == ["a"]


def test_list_trash_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    trash = fake_home / ".crossmem" / ".crossmem-trash.jsonl"
    _write_trash_lines(
        trash,
        [_record_for(_make_doc("a", tags=["x"]), "2026-01-01T00:00:00+00:00")],
    )
    assert [e.doc_id for e in list_trash()] == ["a"]


# ---------------------------------------------------------------------------
# Boundary: cleanup must not reach into KnowledgeStore._backend
# ---------------------------------------------------------------------------


def test_cleanup_module_does_not_reference_private_backend() -> None:
    """cleanup.py must go through the public store API, never store._backend."""
    cleanup_src = (
        Path(__file__).resolve().parent.parent / "src" / "crossmem" / "cleanup.py"
    )
    text = cleanup_src.read_text(encoding="utf-8")
    assert "_backend" not in text


def test_cleanup_then_restore_roundtrip_uses_public_restore(tmp_path: Path) -> None:
    """End-to-end: cleanup deletes via store.delete, restore via store.restore."""
    trash = tmp_path / "trash.jsonl"
    doc = _make_doc("rt", tags=["python"])

    backend = MagicMock()
    backend.get_by_url.return_value = [doc]
    embedder = MagicMock()
    embedder.model_name = "mock"
    embedder.embed_query.return_value = [0.0] * 384
    embedder.embed_passage.return_value = [0.0] * 384
    real_store = KnowledgeStore(backend=backend, embedder=embedder)

    # Drive cleanup through the real KnowledgeStore.find_by_tag: tag mode
    # delegates straight to the backend's native tag index (TODO 26.1) and
    # deletes each match without RRF/over-fetch.
    backend.find_by_tag.return_value = iter([doc])

    result = cleanup(real_store, "python", dry_run=False, mode="tag", trash_path=trash)
    assert result.deleted_ids == ["rt"]
    assert trash.exists()
    backend.find_by_tag.assert_called_once_with("python")
    backend.query_fts.assert_not_called()

    # Spy on the public restore: it must be invoked exactly once and forward
    # the original Document (same id, same embedding) to the backend.
    backend.store.reset_mock()
    embedder.embed_query.reset_mock()
    embedder.embed_passage.reset_mock()
    restored = restore_from_trash(real_store, "rt", trash_path=trash)

    assert restored.id == "rt"
    backend.store.assert_called_once()
    (passed,) = backend.store.call_args.args
    assert passed.id == "rt"
    assert passed.embedding == doc.embedding
    # No re-embedding happened during restore.
    embedder.embed_query.assert_not_called()
    embedder.embed_passage.assert_not_called()
