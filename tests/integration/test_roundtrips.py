"""End-to-end roundtrip integration tests across backends.

Each test exercises a full pipeline (store, query, delete OR
export+import) and asserts that the surviving payload
(id, content, tags, source_url, embedding) matches the original.

Coverage matrix:
- store -> query -> delete (soft delete to trash, then permanent delete)
- export (JSONL + ZIP) -> import -> query in a fresh DB
- full pipeline: store -> export -> import -> query

ChromaDB / Qdrant variants use ``@pytest.mark.skipif`` because the
optional client is typically not installed in the dev environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import Document, Metadata, generate_content_hash
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from pathlib import Path

    from crossmem.backends.base import VectorStoreBase

try:  # pragma: no cover - import guard exercised by skipif below
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_store(db_path: Path, trash_path: Path | None = None) -> KnowledgeStore:
    """Build a KnowledgeStore on a real SQLite backend."""
    backend = SQLiteBackend(db_path)
    return KnowledgeStore(
        backend, FixedEmbedder(model_name="mock-roundtrip"), trash_path=trash_path
    )


def _seed_three(store: KnowledgeStore) -> list[str]:
    ids: list[str] = []
    ids.extend(
        store.store(
            content="Alpha content about Python decorators",
            source_url="https://example.com/a",
            title="Alpha",
            source_type="web",
            tags=["python"],
        )
    )
    ids.extend(
        store.store(
            content="Beta content about Rust ownership",
            source_url="https://example.com/b",
            title="Beta",
            source_type="web",
            tags=["rust"],
        )
    )
    ids.extend(
        store.store(
            content="Gamma content about Go channels",
            source_url="https://example.com/c",
            title="Gamma",
            source_type="web",
            tags=["go", "concurrency"],
        )
    )
    return ids


def _seed_five(store: KnowledgeStore) -> list[str]:
    ids: list[str] = []
    for i in range(5):
        ids.extend(
            store.store(
                content=f"document number {i} with unique payload {i * 31}",
                source_url=f"https://example.com/doc{i}",
                title=f"Doc {i}",
                source_type="web",
                tags=[f"tag{i}"],
            )
        )
    return ids


def _docs_by_id(backend: VectorStoreBase) -> dict[str, Document]:
    return {d.id: d for d in backend.iter_all()}


def _embeddings_close(a: list[float], b: list[float]) -> bool:
    return a == pytest.approx(b, rel=1e-6, abs=1e-6)


# ======================================================================
# 1) store -> query -> delete roundtrip (SQLite)
# ======================================================================


class TestStoreQueryDeleteRoundtripSQLite:
    def test_full_cycle(self, tmp_path: Path) -> None:
        trash = tmp_path / "trash.jsonl"
        store = _make_store(tmp_path / "store.db", trash_path=trash)
        ids = _seed_three(store)
        assert len(ids) == 3

        # Each doc is queryable by its content via the hybrid search.
        for term in ("Alpha", "Beta", "Gamma"):
            hits = store.query(term, top_k=5)
            assert any(term.lower() in d.content.lower() for d in hits), (
                f"{term} not found in {[d.content for d in hits]}"
            )

        # Soft-delete the alpha doc -> goes to trash, removed from backend.
        deleted = store.delete(source_url="https://example.com/a")
        assert deleted == 1
        assert trash.exists()
        trash_lines = trash.read_text(encoding="utf-8").strip().splitlines()
        assert len(trash_lines) == 1
        assert "Alpha" in trash_lines[0]

        # The deleted doc no longer appears in queries.
        hits = store.query("Alpha", top_k=5)
        remaining_ids = {d.id for d in hits}
        assert ids[0] not in remaining_ids

        # The other two docs are still there.
        backend_ids = {d.id for d in store._backend.iter_all()}
        assert ids[1] in backend_ids
        assert ids[2] in backend_ids
        assert ids[0] not in backend_ids

        # Permanent delete bypasses the trash.
        permanent = store.delete(source_url="https://example.com/b", permanent=True)
        assert permanent == 1
        # Trash file size unchanged (still only the alpha entry).
        post_lines = trash.read_text(encoding="utf-8").strip().splitlines()
        assert len(post_lines) == 1

        store._backend.close()  # type: ignore[attr-defined]

    def test_restore_from_trash_and_query_again(self, tmp_path: Path) -> None:
        """Soft-deleted docs can be re-inserted via :meth:`KnowledgeStore.restore`."""
        trash = tmp_path / "trash.jsonl"
        store = _make_store(tmp_path / "restore.db", trash_path=trash)
        store.store(
            content="Restorable doc about Python",
            source_url="https://example.com/restorable",
            title="Restorable",
            source_type="web",
            tags=["python"],
        )

        snapshot = next(iter(store._backend.iter_all()))
        assert store.delete(source_url="https://example.com/restorable") == 1
        assert list(store._backend.iter_all()) == []

        # Restore the snapshot directly — should be queryable again.
        store.restore(snapshot)
        hits = store.query("Restorable", top_k=5)
        assert any(d.id == snapshot.id for d in hits)

        store._backend.close()  # type: ignore[attr-defined]


# ======================================================================
# 2) export -> import -> query roundtrip (SQLite)
# ======================================================================


class TestExportImportQueryRoundtripSQLite:
    @pytest.mark.parametrize("fmt", ["jsonl", "zip"])
    def test_export_then_import_then_query(self, tmp_path: Path, fmt: str) -> None:
        src = _make_store(tmp_path / "src.db")
        ids = _seed_five(src)
        assert len(ids) == 5

        out = tmp_path / f"dump.{fmt}"
        n_written = src.export(out, format=fmt)
        assert n_written == 5
        assert out.exists()

        # Import into a fresh DB.
        dst = _make_store(tmp_path / "dst.db")
        n_read = dst.import_data(out)
        assert n_read == 5

        src_docs = _docs_by_id(src._backend)  # type: ignore[arg-type]
        dst_docs = _docs_by_id(dst._backend)  # type: ignore[arg-type]
        assert set(dst_docs) == set(src_docs)

        for doc_id, src_doc in src_docs.items():
            dst_doc = dst_docs[doc_id]
            assert dst_doc.content == src_doc.content
            assert dst_doc.metadata.source_url == src_doc.metadata.source_url
            assert dst_doc.metadata.tags == src_doc.metadata.tags
            assert dst_doc.metadata.title == src_doc.metadata.title
            # sqlite-vec stores fp32 — float identity holds across export/import.
            assert _embeddings_close(dst_doc.embedding, src_doc.embedding)

        # The imported docs are queryable.
        for i in range(5):
            hits = dst.query(f"document number {i}", top_k=10)
            assert any(f"document number {i}" in d.content for d in hits)

        src._backend.close()  # type: ignore[attr-defined]
        dst._backend.close()  # type: ignore[attr-defined]


# ======================================================================
# 3) Full roundtrip: store -> export -> import -> query
# ======================================================================


class TestFullRoundtripDataIdentity:
    def test_unicode_content_survives_full_pipeline(self, tmp_path: Path) -> None:
        """10 Unicode docs survive the full pipeline with id/content/tags intact."""
        src = _make_store(tmp_path / "src.db")

        unicode_payloads = [
            ("English ASCII baseline", ["english", "baseline"]),
            ("CJK 中文测试 文本 内容", ["chinese", "cjk"]),
            ("Japanese こんにちは 世界 テスト", ["japanese", "cjk"]),
            ("Korean 안녕하세요 세계 시험", ["korean", "cjk"]),
            ("Emoji content 🚀🎉 with 👋 markers", ["emoji"]),
            ("Arabic RTL مرحبا بالعالم اختبار", ["arabic", "rtl"]),
            ("Hebrew RTL שלום עולם בדיקה", ["hebrew", "rtl"]),
            ("Mixed scripts: hello 你好 こんにちは 안녕 🌍", ["mixed", "cjk", "emoji"]),
            ("German Umlaute: Schöne Grüße über Köln", ["german"]),
            ("Math symbols: ∑ √ ∞ ≈ π × ÷", ["math"]),
        ]

        original_ids: list[str] = []
        for i, (content, tags) in enumerate(unicode_payloads):
            ids = src.store(
                content=content,
                source_url=f"https://example.com/u{i}",
                title=f"Unicode {i}",
                source_type="web",
                tags=tags,
            )
            assert len(ids) == 1
            original_ids.extend(ids)

        original = _docs_by_id(src._backend)  # type: ignore[arg-type]
        assert len(original) == 10

        # Export and import into a fresh DB; the imported docs must round-trip.
        out = tmp_path / "unicode.zip"
        src.export(out, format="zip")

        final_store = _make_store(tmp_path / "imported.db")
        final_store.import_data(out)
        final_backend = final_store._backend  # type: ignore[assignment]
        final = _docs_by_id(final_backend)
        assert set(final) == set(original)

        # Identity holds for content, source_url, tags, embedding.
        for doc_id, src_doc in original.items():
            dst_doc = final[doc_id]
            assert dst_doc.content == src_doc.content
            assert dst_doc.metadata.source_url == src_doc.metadata.source_url
            assert dst_doc.metadata.tags == src_doc.metadata.tags
            assert _embeddings_close(dst_doc.embedding, src_doc.embedding)

        # Each Unicode payload is independently queryable in the final DB.
        for content, _tags in unicode_payloads:
            # Pick a representative substring (first 5 chars or so) — the
            # FTS5 trigram tokenizer matches CJK substrings reliably.
            probe = content.split()[0]
            hits = final_store.query(probe, top_k=10)
            assert hits, f"no hits for {probe!r}"

        src._backend.close()  # type: ignore[attr-defined]
        final_backend.close()


# ======================================================================
# Optional backends — placeholder skipif tests so the matrix is explicit.
# ======================================================================


@pytest.mark.skipif(chromadb is None, reason="chromadb not installed")
def test_store_query_delete_roundtrip_chroma(
    tmp_path: Path,
) -> None:  # pragma: no cover
    """Store/query/delete roundtrip on ChromaBackend (optional)."""
    from crossmem.backends.chroma_backend import ChromaBackend

    backend = ChromaBackend(
        client=chromadb.EphemeralClient(), collection_name="roundtrip"
    )
    store = KnowledgeStore(
        backend,
        FixedEmbedder(model_name="mock-roundtrip"),
        trash_path=tmp_path / "trash.jsonl",
    )
    ids = store.store(
        content="Chroma roundtrip content",
        source_url="https://example.com/chroma-rt",
        title="Chroma RT",
        source_type="web",
        tags=["chroma"],
    )
    assert ids
    docs = backend.get_by_url("https://example.com/chroma-rt")
    assert len(docs) == 1
    backend.delete(docs[0].id)
    assert backend.get_by_url("https://example.com/chroma-rt") == []


@pytest.mark.skipif(True, reason="qdrant backend not implemented yet (Task 9.2)")
def test_store_query_delete_roundtrip_qdrant() -> None:  # pragma: no cover
    """Placeholder until Task 9.2 lands the Qdrant backend."""


def _make_doc_for_inline_check(doc_id: str, content: str) -> Document:
    """Tiny helper proving Document construction stays compatible."""
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * EMBEDDING_DIM,
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=doc_id,
            source_type="web",
            stored_at="2024-01-01T00:00:00Z",
            embedding_model="mock",
            embedding_dim=EMBEDDING_DIM,
            namespace="default",
            tags=["x"],
            content_hash=generate_content_hash(content),
        ),
    )


def test_make_doc_helper_is_constructable() -> None:
    """Sanity: the inline Document helper builds without errors."""
    doc = _make_doc_for_inline_check("d0", "hello")
    assert doc.id == "d0"
    assert len(doc.embedding) == EMBEDDING_DIM
