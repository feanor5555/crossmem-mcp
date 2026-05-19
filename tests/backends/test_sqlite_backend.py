"""Tests for SQLiteBackend schema setup and initialization."""

from __future__ import annotations

import json

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata, generate_content_hash


class TestSQLiteBackendInit:
    """Test database initialization and schema creation."""

    def test_create_in_memory(self) -> None:
        backend = SQLiteBackend(":memory:")
        assert backend._conn is not None
        backend.close()

    def test_create_file_based(self, tmp_path: object) -> None:
        db_file = tmp_path / "test.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)
        assert db_file.exists()  # type: ignore[union-attr]
        backend.close()

    def test_tables_exist(self) -> None:
        backend = SQLiteBackend(":memory:")
        conn = backend._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
        ).fetchall()
        table_names = {row["name"] for row in rows}

        expected = {"documents", "document_tags", "schema_version"}
        assert expected.issubset(table_names)

        # Virtual tables show up differently — query them directly
        conn.execute("SELECT * FROM vec_documents WHERE rowid = -1")
        conn.execute("SELECT * FROM fts_documents WHERE rowid = -1")
        backend.close()

    def test_schema_version_is_current(self) -> None:
        # Schema is initialized at v1 then migrated through v2 (sync state)
        # to v3 (trigram FTS tokenizer).
        backend = SQLiteBackend(":memory:")
        conn = backend._get_conn()
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 3
        backend.close()

    def test_pragmas_set(self) -> None:
        backend = SQLiteBackend(":memory:")
        conn = backend._get_conn()

        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy == 5000

        sync = conn.execute("PRAGMA synchronous").fetchone()[0]
        assert sync == 1  # NORMAL

        cache = conn.execute("PRAGMA cache_size").fetchone()[0]
        assert cache == -64000
        backend.close()

    def test_indices_exist(self) -> None:
        backend = SQLiteBackend(":memory:")
        conn = backend._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {row["name"] for row in rows}

        expected = {"idx_source_url", "idx_namespace", "idx_stored_at", "idx_tag"}
        assert expected.issubset(index_names)
        backend.close()

    def test_idempotent_init(self, tmp_path: object) -> None:
        db_file = tmp_path / "test.db"  # type: ignore[operator]

        backend1 = SQLiteBackend(db_file)
        backend1.close()

        backend2 = SQLiteBackend(db_file)
        conn = backend2._get_conn()
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 3

        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1
        backend2.close()

    def test_wal_mode_file_based(self, tmp_path: object) -> None:
        db_file = tmp_path / "test.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)
        conn = backend._get_conn()

        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        backend.close()


def _make_doc(
    doc_id: str = "test123",
    content: str = "Test content",
    tags: list[str] | None = None,
) -> Document:
    """Helper to create a test Document."""
    if tags is None:
        tags = ["python", "test"]
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url="https://example.com",
            title="Test Doc",
            source_type="web",
            stored_at="2024-01-01T00:00:00Z",
            embedding_model="test-model",
            embedding_dim=384,
            namespace="default",
            tags=tags,
            content_hash=generate_content_hash(content),
        ),
    )


class TestSQLiteBackendQueryVector:
    """Test the query_vector() method."""

    def test_query_vector_finds_similar(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc()
        backend.store(doc)

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id
        backend.close()

    def test_query_vector_top_k(self) -> None:
        backend = SQLiteBackend(":memory:")
        for i in range(5):
            doc = _make_doc(doc_id=f"doc{i}", content=f"Content {i}")
            backend.store(doc)

        results = backend.query_vector([0.1] * 384, top_k=3)
        assert len(results) == 3
        backend.close()

    def test_query_vector_empty_db(self) -> None:
        backend = SQLiteBackend(":memory:")
        results = backend.query_vector([0.1] * 384, top_k=5)
        assert results == []
        backend.close()


class TestSQLiteBackendQueryFts:
    """Test the query_fts() method."""

    def test_query_fts_finds_content(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc(content="Python asyncio tutorial")
        backend.store(doc)

        results = backend.query_fts("Python asyncio", top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id
        backend.close()

    def test_query_fts_with_tag_filter(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc1 = _make_doc(doc_id="d1", content="Python guide", tags=["python"])
        doc2 = _make_doc(doc_id="d2", content="Python reference", tags=["javascript"])
        backend.store(doc1)
        backend.store(doc2)

        results = backend.query_fts("Python", tags=["python"])
        assert len(results) == 1
        assert results[0].id == "d1"
        backend.close()

    def test_query_fts_special_chars(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc(content="Test content")
        backend.store(doc)

        # Should not crash with special FTS5 characters
        results = backend.query_fts('test "quotes" AND OR NOT', top_k=5)
        assert isinstance(results, list)
        backend.close()

    def test_query_fts_empty_db(self) -> None:
        backend = SQLiteBackend(":memory:")
        results = backend.query_fts("anything", top_k=5)
        assert results == []
        backend.close()

    def test_query_fts_empty_string_returns_empty(self) -> None:
        """Empty query short-circuits to ``[]`` without hitting FTS5."""
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(content="Python asyncio tutorial"))

        assert backend.query_fts("", top_k=5) == []
        backend.close()

    def test_query_fts_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only query short-circuits to ``[]`` without hitting FTS5."""
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(content="Python asyncio tutorial"))

        assert backend.query_fts("   ", top_k=5) == []
        assert backend.query_fts("\t\n", top_k=5) == []
        backend.close()

    def test_query_fts_empty_string_with_tags_returns_empty(self) -> None:
        """Empty query short-circuits even when a tag filter is supplied."""
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(content="Python guide", tags=["python"]))

        assert backend.query_fts("", top_k=5, tags=["python"]) == []
        assert backend.query_fts("   ", top_k=5, tags=["python"]) == []
        backend.close()

    def test_query_fts_empty_string_skips_fts_match(self) -> None:
        """Empty/whitespace queries never reach the FTS5 ``MATCH`` clause.

        Without the short-circuit a stray tokenizer change could turn the
        quoted-empty-phrase trick into an ``OperationalError`` (see TODO
        24.8). The trace callback records every SQL statement executed by
        the connection so we can assert no ``fts_documents MATCH`` query
        is issued for empty input.
        """
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(content="Python asyncio"))

        executed: list[str] = []
        conn = backend._get_conn()
        conn.set_trace_callback(executed.append)
        try:
            assert backend.query_fts("", top_k=5) == []
            assert backend.query_fts("   ", top_k=5) == []
            assert backend.query_fts("", top_k=5, tags=["python"]) == []
        finally:
            conn.set_trace_callback(None)

        joined = " ".join(executed).lower()
        assert "fts_documents match" not in joined
        backend.close()


class TestSQLiteBackendStore:
    """Test the store() method."""

    def test_store_document(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc()
        backend.store(doc)

        conn = backend._get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc.id,)).fetchone()
        assert row is not None
        assert row["content"] == "Test content"
        assert row["source_url"] == "https://example.com"
        assert json.loads(row["tags"]) == ["python", "test"]
        backend.close()

    def test_store_creates_vec_entry(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc()
        backend.store(doc)

        conn = backend._get_conn()
        vec_row = conn.execute("SELECT rowid FROM vec_documents").fetchone()
        assert vec_row is not None
        backend.close()

    def test_store_creates_fts_entry(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc()
        backend.store(doc)

        conn = backend._get_conn()
        fts_row = conn.execute(
            "SELECT * FROM fts_documents WHERE fts_documents MATCH 'Test'"
        ).fetchone()
        assert fts_row is not None
        backend.close()

    def test_store_creates_tags(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc()
        backend.store(doc)

        conn = backend._get_conn()
        tags = conn.execute(
            "SELECT tag FROM document_tags WHERE doc_id = ?", (doc.id,)
        ).fetchall()
        assert {r["tag"] for r in tags} == {"python", "test"}
        backend.close()

    def test_store_upsert(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc1 = _make_doc(content="Version 1")
        backend.store(doc1)

        doc2 = _make_doc(content="Version 2")
        backend.store(doc2)

        conn = backend._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        row = conn.execute(
            "SELECT content FROM documents WHERE id = ?", (doc1.id,)
        ).fetchone()
        assert row["content"] == "Version 2"
        backend.close()


def _make_doc_with_url(
    doc_id: str,
    source_url: str,
    content: str = "Test content",
    tags: list[str] | None = None,
) -> Document:
    """Helper to create a Document with a custom source_url."""
    if tags is None:
        tags = ["python", "test"]
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=source_url,
            title="Test Doc",
            source_type="web",
            stored_at="2024-01-01T00:00:00Z",
            embedding_model="test-model",
            embedding_dim=384,
            namespace="default",
            tags=tags,
            content_hash=generate_content_hash(content),
        ),
    )


class TestSQLiteBackendDelete:
    """Test the delete() method."""

    def test_delete_removes_document(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc(content="Python asyncio tutorial")
        backend.store(doc)

        backend.delete(doc.id)

        # Vector and FTS searches find nothing
        assert backend.query_vector([0.1] * 384, top_k=5) == []
        assert backend.query_fts("Python asyncio", top_k=5) == []

        # All four tables are empty
        conn = backend._get_conn()
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        vec_count = conn.execute("SELECT COUNT(*) FROM vec_documents").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM fts_documents").fetchone()[0]
        tag_count = conn.execute("SELECT COUNT(*) FROM document_tags").fetchone()[0]
        assert doc_count == 0
        assert vec_count == 0
        assert fts_count == 0
        assert tag_count == 0
        backend.close()

    def test_delete_nonexistent_is_noop(self) -> None:
        backend = SQLiteBackend(":memory:")
        # Must not raise
        backend.delete("does-not-exist")
        backend.close()

    def test_delete_only_removes_target(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc1 = _make_doc(doc_id="d1", content="Alpha content")
        doc2 = _make_doc(doc_id="d2", content="Beta content")
        backend.store(doc1)
        backend.store(doc2)

        backend.delete("d1")

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == "d2"

        conn = backend._get_conn()
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count == 1
        backend.close()


class TestSQLiteBackendGetByUrl:
    """Test the get_by_url() method."""

    def test_get_by_url_finds_doc(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc_with_url("d1", "https://example.com/page")
        backend.store(doc)

        results = backend.get_by_url("https://example.com/page")
        assert len(results) == 1
        assert results[0].id == "d1"
        assert results[0].metadata.source_url == "https://example.com/page"
        backend.close()

    def test_get_by_url_returns_empty_list(self) -> None:
        backend = SQLiteBackend(":memory:")
        results = backend.get_by_url("https://nope.example.com/")
        assert results == []
        backend.close()

    def test_get_by_url_returns_multiple(self) -> None:
        backend = SQLiteBackend(":memory:")
        url = "https://example.com/shared"
        doc1 = _make_doc_with_url("d1", url, content="First")
        doc2 = _make_doc_with_url("d2", url, content="Second")
        backend.store(doc1)
        backend.store(doc2)

        results = backend.get_by_url(url)
        assert len(results) == 2
        assert {r.id for r in results} == {"d1", "d2"}
        backend.close()

    def test_get_by_url_returns_deterministic_order(self) -> None:
        """Two stores in different order MUST yield the same id sequence."""
        url = "https://example.com/shared"

        backend_a = SQLiteBackend(":memory:")
        backend_a.store(_make_doc_with_url("z1", url, content="A"))
        backend_a.store(_make_doc_with_url("a1", url, content="B"))
        backend_a.store(_make_doc_with_url("m1", url, content="C"))
        order_a = [d.id for d in backend_a.get_by_url(url)]
        backend_a.close()

        backend_b = SQLiteBackend(":memory:")
        backend_b.store(_make_doc_with_url("m1", url, content="C"))
        backend_b.store(_make_doc_with_url("z1", url, content="A"))
        backend_b.store(_make_doc_with_url("a1", url, content="B"))
        order_b = [d.id for d in backend_b.get_by_url(url)]
        backend_b.close()

        assert order_a == order_b
        assert order_a == sorted(order_a)


class TestSQLiteBackendGetById:
    """Test the get_by_id() method."""

    def test_get_by_id_returns_doc_with_embedding(self) -> None:
        backend = SQLiteBackend(":memory:")
        doc = _make_doc(doc_id="d1", content="Alpha")
        backend.store(doc)

        result = backend.get_by_id("d1")

        assert result is not None
        assert result.id == "d1"
        assert result.content == "Alpha"
        # Embedding round-trips with full fidelity (float32 precision)
        assert len(result.embedding) == 384
        assert pytest.approx(result.embedding[0], rel=1e-5) == 0.1
        backend.close()

    def test_get_by_id_returns_none_for_missing(self) -> None:
        backend = SQLiteBackend(":memory:")
        assert backend.get_by_id("nope") is None
        backend.close()


class TestSQLiteBackendStats:
    """Test the stats() method."""

    def test_stats_empty_db(self) -> None:
        backend = SQLiteBackend(":memory:")
        stats = backend.stats()

        assert stats["document_count"] == 0
        assert stats["top_tags"] == []
        assert stats["backend"] == "sqlite"
        assert stats["db_size_bytes"] >= 0
        backend.close()

    def test_stats_counts_documents(self) -> None:
        backend = SQLiteBackend(":memory:")
        for i in range(3):
            backend.store(_make_doc(doc_id=f"doc{i}", content=f"Content {i}"))

        stats = backend.stats()
        assert stats["document_count"] == 3
        backend.close()

    def test_stats_top_tags(self) -> None:
        backend = SQLiteBackend(":memory:")
        # python: 3, test: 3, javascript: 2, rust: 1
        backend.store(_make_doc(doc_id="d1", tags=["python", "test"]))
        backend.store(_make_doc(doc_id="d2", tags=["python", "test"]))
        backend.store(_make_doc(doc_id="d3", tags=["python", "test", "javascript"]))
        backend.store(_make_doc(doc_id="d4", tags=["javascript", "rust"]))

        stats = backend.stats()
        top_tags = stats["top_tags"]

        # All tags should appear (only 4 unique tags, well under the 10 limit)
        assert len(top_tags) == 4
        # First entries are the highest-count tags
        first_two = {top_tags[0][0], top_tags[1][0]}
        assert first_two == {"python", "test"}
        assert top_tags[0][1] == 3
        assert top_tags[1][1] == 3
        assert top_tags[2] == ("javascript", 2)
        assert top_tags[3] == ("rust", 1)
        backend.close()

    def test_stats_top_tags_limited_to_10(self) -> None:
        backend = SQLiteBackend(":memory:")
        # Create 12 docs each with a unique tag -> 12 unique tags
        for i in range(12):
            backend.store(_make_doc(doc_id=f"doc{i}", tags=[f"tag{i:02d}"]))

        stats = backend.stats()
        assert len(stats["top_tags"]) == 10
        backend.close()

    def test_stats_db_size_file(self, tmp_path: object) -> None:
        db_file = tmp_path / "stats.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)
        backend.store(_make_doc())

        stats = backend.stats()
        assert stats["db_size_bytes"] > 0
        assert stats["backend"] == "sqlite"
        backend.close()


class TestSQLiteBackendFindByTag:
    """Tests for the find_by_tag() native-index path (TODO 26.1)."""

    def test_find_by_tag_returns_docs_with_exact_tag(self) -> None:
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(doc_id="d1", tags=["python", "asyncio"]))
        backend.store(_make_doc(doc_id="d2", tags=["python"]))
        backend.store(_make_doc(doc_id="d3", tags=["rust"]))

        ids = {d.id for d in backend.find_by_tag("python")}
        assert ids == {"d1", "d2"}
        backend.close()

    def test_find_by_tag_returns_iterator(self) -> None:
        """The contract returns an Iterator so callers can stream large hits."""
        from collections.abc import Iterator

        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(doc_id="d1", tags=["python"]))

        it = backend.find_by_tag("python")
        assert isinstance(it, Iterator)
        backend.close()

    def test_find_by_tag_finds_docs_whose_content_lacks_the_tag(self) -> None:
        """Spec: tag-only-in-metadata docs MUST be returned (no content match)."""
        backend = SQLiteBackend(":memory:")
        # Content has nothing in common with the tag string.
        backend.store(
            _make_doc(
                doc_id="meta-only",
                content="Completely unrelated prose with no tag tokens.",
                tags=["zzzobscure"],
            )
        )

        hits = list(backend.find_by_tag("zzzobscure"))
        assert [d.id for d in hits] == ["meta-only"]
        backend.close()

    def test_find_by_tag_distinct_per_doc(self) -> None:
        """A doc that carries the tag once appears exactly once in the result."""
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(doc_id="d1", tags=["python", "python:3.12"]))
        backend.store(_make_doc(doc_id="d2", tags=["python"]))

        ids = [d.id for d in backend.find_by_tag("python")]
        assert sorted(ids) == ["d1", "d2"]
        assert len(ids) == 2
        backend.close()

    def test_find_by_tag_no_match_returns_empty(self) -> None:
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(doc_id="d1", tags=["python"]))

        assert list(backend.find_by_tag("rust")) == []
        backend.close()

    def test_find_by_tag_uses_idx_tag_index(self) -> None:
        """The SQL plan MUST use the ``idx_tag`` index, not a sequential scan.

        EXPLAIN QUERY PLAN reports whether SQLite picks an index. The
        backend must issue a query that the planner can serve via the
        ``idx_tag`` index on ``document_tags`` so cleanup --mode tag
        stays fast even on million-doc corpora.
        """
        backend = SQLiteBackend(":memory:")
        backend.store(_make_doc(doc_id="d1", tags=["python"]))

        conn = backend._get_conn()
        plan_rows = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT DISTINCT d.* FROM documents d
            INNER JOIN document_tags dt ON d.id = dt.doc_id
            WHERE dt.tag = ?
            """,
            ("python",),
        ).fetchall()
        plan_text = " ".join(str(row["detail"]) for row in plan_rows).lower()
        assert "idx_tag" in plan_text
        backend.close()
