"""Atomic multi-chunk store: all chunks of a document land in one transaction.

A crash between chunks (e.g. backend failure on chunk N) must leave the DB
in a defined state: either every chunk of the document is persisted, or
none of them are. Half-stored documents are forbidden because callers have
no way to discover that a document is incomplete.

The contract is expressed via :meth:`VectorStoreBase.upsert_many`:
:meth:`KnowledgeStore.store` invokes it exactly once per document, and
backends MUST treat the supplied list as a single atomic unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from pathlib import Path


def _multi_chunk_prose(min_bytes: int = 6 * 1024) -> str:
    """Build prose content guaranteed to yield several chunks."""
    parts: list[str] = []
    i = 0
    while sum(len(p.encode("utf-8")) for p in parts) < min_bytes:
        parts.append(
            f"Sentence number {i} discusses topic {i} with detail {i} and "
            f"context word_{i} for chunk diversity in this prose paragraph. "
        )
        i += 1
    return "".join(parts)


def test_store_invokes_upsert_many_once_per_document(tmp_path: Path) -> None:
    """``KnowledgeStore.store`` collapses all chunks into one batch call."""
    backend = SQLiteBackend(tmp_path / "atomic.db")
    calls: list[list[Document]] = []
    original_upsert = backend.upsert_many

    def spy(docs: list[Document]) -> None:
        calls.append(list(docs))
        original_upsert(docs)

    backend.upsert_many = spy  # type: ignore[method-assign]

    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-atomic"))
    content = _multi_chunk_prose()
    ids = store.store(
        content=content,
        source_url="https://example.com/atomic",
        title="Atomic",
        source_type="web",
    )

    assert len(ids) > 1, "test requires multi-chunk content to be meaningful"
    assert len(calls) == 1
    assert [d.id for d in calls[0]] == ids
    backend.close()


def test_store_multi_chunk_roundtrip_all_chunks_visible(tmp_path: Path) -> None:
    """Every chunk of a multi-chunk document is retrievable after store()."""
    backend = SQLiteBackend(tmp_path / "atomic.db")
    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-atomic"))
    url = "https://example.com/full"
    ids = store.store(
        content=_multi_chunk_prose(),
        source_url=url,
        title="Full",
        source_type="web",
    )
    assert len(ids) > 1

    persisted = backend.get_by_url(url)
    assert {d.id for d in persisted} == set(ids)
    backend.close()


def test_store_crash_mid_batch_leaves_db_empty(tmp_path: Path) -> None:
    """Failure mid-batch rolls back every chunk persisted so far.

    We intercept ``_write_doc`` so the third write inside ``upsert_many``
    raises. The first two ``_write_doc`` invocations already issued
    ``INSERT`` statements on the open transaction — atomicity guarantees
    that the surrounding ``with conn:`` rolls them back, so ``get_by_url``
    MUST report zero chunks for the affected source after the failure.
    """
    backend = SQLiteBackend(tmp_path / "crash.db")
    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-crash"))
    url = "https://example.com/crash"

    real_write = backend._write_doc
    calls = {"n": 0}

    def flaky(conn: object, doc: Document) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated mid-batch crash")
        real_write(conn, doc)  # type: ignore[arg-type]

    backend._write_doc = flaky  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated mid-batch crash"):
        store.store(
            content=_multi_chunk_prose(),
            source_url=url,
            title="Crash",
            source_type="web",
        )

    assert calls["n"] >= 3, "test setup expects a multi-chunk store"

    backend._write_doc = real_write  # type: ignore[method-assign]
    persisted = backend.get_by_url(url)
    assert persisted == [], (
        "crash mid-upsert_many must leave the DB empty for this URL "
        f"(got {len(persisted)} chunks)"
    )
    backend.close()


def test_upsert_many_atomic_on_sqlite_backend(tmp_path: Path) -> None:
    """Calling ``upsert_many`` with a poisoned entry rolls back the whole batch.

    Drives the SQLite backend directly to prove the transactional guarantee
    independent of ``KnowledgeStore``.
    """
    backend = SQLiteBackend(tmp_path / "direct.db")
    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-direct"))

    # Build two real chunks via the public API.
    url = "https://example.com/direct"
    ids = store.store(
        content=_multi_chunk_prose(),
        source_url=url,
        title="Direct",
        source_type="web",
    )
    assert len(ids) > 1

    # Wipe the URL and confirm an empty starting point.
    for doc_id in ids:
        backend.delete(doc_id)
    assert backend.get_by_url(url) == []

    # Build a batch where the second document is malformed (None) so the
    # backend rejects it mid-transaction; the first document must NOT
    # survive.
    good_doc = Document(
        id="good-doc-id",
        content="good content",
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=url,
            title="Direct",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=384,
            namespace="default",
            tags=["python"],
            content_hash="hash-good",
        ),
    )

    with pytest.raises((AttributeError, TypeError)):
        backend.upsert_many([good_doc, None])  # type: ignore[list-item]

    assert backend.get_by_url(url) == []
    assert backend.get_by_id("good-doc-id") is None
    backend.close()


def test_upsert_many_empty_list_is_noop(tmp_path: Path) -> None:
    """Empty input must not touch the DB."""
    backend = SQLiteBackend(tmp_path / "empty.db")
    backend.upsert_many([])
    stats = backend.stats()
    assert stats["document_count"] == 0
    backend.close()


def test_upsert_many_persists_every_doc_in_batch(tmp_path: Path) -> None:
    """Positive path: all docs in a successful batch land in the DB."""
    backend = SQLiteBackend(tmp_path / "ok.db")
    store = KnowledgeStore(backend, FixedEmbedder(model_name="mock-batch"))
    url = "https://example.com/ok"
    ids = store.store(
        content=_multi_chunk_prose(),
        source_url=url,
        title="OK",
        source_type="web",
    )
    persisted = backend.get_by_url(url)
    assert {d.id for d in persisted} == set(ids)
    # Every chunk must be query-able by id.
    for doc_id in ids:
        assert backend.get_by_id(doc_id) is not None
    backend.close()
