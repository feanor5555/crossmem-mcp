"""Tests for the v2 -> v3 schema migration (Task 21.6).

Existing DBs created before the trigram FTS tokenizer landed retain their
original (default ``unicode61``) tokenizer, which misses CJK substring queries.
The v3 migration drops ``fts_documents`` and re-creates it with
``tokenize='trigram remove_diacritics 1'``, then re-indexes from ``documents``.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import sqlite_vec

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM

if TYPE_CHECKING:
    from pathlib import Path


def _build_v2_db(db_path: Path, *, include_sync_residue: bool = False) -> None:
    """Create a v2 DB with the *old* (non-trigram) FTS tokenizer.

    Mirrors the historic v1 schema (default ``unicode61``). The trigram
    migration must upgrade this DB in-place.

    The ``include_sync_residue`` flag also writes the now-removed
    v1->v2 sync columns (``dirty``, ``sync_state`` table and
    ``idx_documents_dirty``). It is used by the forward-compat test that
    asserts pre-existing DBs which still carry these artefacts continue
    to open via the current code path.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA page_size = 8192")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    if include_sync_residue:
        conn.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT,
                source_type TEXT,
                stored_at TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                namespace TEXT DEFAULT 'default',
                tags TEXT,
                content_hash TEXT,
                dirty INTEGER NOT NULL DEFAULT 1
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT,
                source_type TEXT,
                stored_at TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                namespace TEXT DEFAULT 'default',
                tags TEXT,
                content_hash TEXT
            )
            """
        )
    conn.execute(
        "CREATE VIRTUAL TABLE vec_documents "
        f"USING vec0(embedding float[{EMBEDDING_DIM}])"
    )
    # OLD tokenizer — no explicit tokenize clause => unicode61 default.
    conn.execute(
        """
        CREATE VIRTUAL TABLE fts_documents USING fts5(
            content, title, tags,
            content='documents', content_rowid='rowid'
        )
        """
    )
    conn.execute("CREATE TABLE document_tags (doc_id TEXT NOT NULL, tag TEXT NOT NULL)")
    conn.execute("CREATE INDEX idx_source_url ON documents(source_url)")
    conn.execute("CREATE INDEX idx_namespace ON documents(namespace)")
    conn.execute("CREATE INDEX idx_stored_at ON documents(stored_at)")
    conn.execute("CREATE INDEX idx_tag ON document_tags(tag)")
    if include_sync_residue:
        conn.execute(
            "CREATE INDEX idx_documents_dirty ON documents(dirty) WHERE dirty = 1"
        )
        conn.execute(
            """
            CREATE TABLE sync_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_seq INTEGER NOT NULL DEFAULT 0,
                last_session TEXT
            )
            """
        )
        conn.execute("INSERT INTO sync_state (id, current_seq) VALUES (1, 0)")
    conn.execute("INSERT INTO schema_version VALUES (2)")

    # Seed one ASCII and one CJK doc so the migration has rows to rebuild.
    import struct

    ascii_blob = struct.pack(f"{EMBEDDING_DIM}f", *([0.1] * EMBEDDING_DIM))
    cjk_blob = struct.pack(f"{EMBEDDING_DIM}f", *([0.2] * EMBEDDING_DIM))

    if include_sync_residue:
        conn.execute(
            """
            INSERT INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash, dirty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                "ascii1",
                "hello world from before migration",
                "https://example.com/a",
                "ASCII Doc",
                "web",
                "2024-01-01T00:00:00Z",
                "test-model",
                EMBEDDING_DIM,
                "default",
                "[]",
                "hash-a",
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ascii1",
                "hello world from before migration",
                "https://example.com/a",
                "ASCII Doc",
                "web",
                "2024-01-01T00:00:00Z",
                "test-model",
                EMBEDDING_DIM,
                "default",
                "[]",
                "hash-a",
            ),
        )
    a_rowid = conn.execute(
        "SELECT rowid FROM documents WHERE id = ?", ("ascii1",)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
        (a_rowid, ascii_blob),
    )
    conn.execute(
        "INSERT INTO fts_documents (rowid, content, title, tags) VALUES (?, ?, ?, ?)",
        (a_rowid, "hello world from before migration", "ASCII Doc", "[]"),
    )

    if include_sync_residue:
        conn.execute(
            """
            INSERT INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash, dirty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                "cjk1",
                "今天天气很好，我喜欢编程",
                "https://example.com/c",
                "中文测试",
                "web",
                "2024-01-01T00:00:00Z",
                "test-model",
                EMBEDDING_DIM,
                "default",
                "[]",
                "hash-c",
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cjk1",
                "今天天气很好，我喜欢编程",
                "https://example.com/c",
                "中文测试",
                "web",
                "2024-01-01T00:00:00Z",
                "test-model",
                EMBEDDING_DIM,
                "default",
                "[]",
                "hash-c",
            ),
        )
    c_rowid = conn.execute(
        "SELECT rowid FROM documents WHERE id = ?", ("cjk1",)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
        (c_rowid, cjk_blob),
    )
    conn.execute(
        "INSERT INTO fts_documents (rowid, content, title, tags) VALUES (?, ?, ?, ?)",
        (c_rowid, "今天天气很好，我喜欢编程", "中文测试", "[]"),
    )

    conn.commit()
    conn.close()


def test_migration_bumps_schema_version_to_3(tmp_path: Path) -> None:
    db_file = tmp_path / "v2.db"
    _build_v2_db(db_file)

    backend = SQLiteBackend(db_file)
    try:
        conn = backend._get_conn()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[
            "version"
        ]
        assert version == 3
    finally:
        backend.close()


def test_migration_swaps_fts_tokenizer_to_trigram(tmp_path: Path) -> None:
    db_file = tmp_path / "v2.db"
    _build_v2_db(db_file)

    backend = SQLiteBackend(db_file)
    try:
        conn = backend._get_conn()
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='fts_documents'"
        ).fetchone()["sql"]
        assert "trigram" in sql.lower()
        assert "remove_diacritics" in sql.lower()
    finally:
        backend.close()


def test_migration_preserves_existing_docs(tmp_path: Path) -> None:
    db_file = tmp_path / "v2.db"
    _build_v2_db(db_file)

    backend = SQLiteBackend(db_file)
    try:
        results = backend.query_fts("hello", top_k=10)
        assert any(d.id == "ascii1" for d in results)
    finally:
        backend.close()


def test_migration_enables_cjk_substring_search(tmp_path: Path) -> None:
    """The point of the migration: post-upgrade CJK substrings must hit."""
    db_file = tmp_path / "v2.db"
    _build_v2_db(db_file)

    backend = SQLiteBackend(db_file)
    try:
        # 3-char substring (trigram minimum) of the CJK doc content.
        results = backend.query_fts("喜欢编", top_k=10)
        assert any(d.id == "cjk1" for d in results)
    finally:
        backend.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-opening a migrated DB does not bump the version or break FTS."""
    db_file = tmp_path / "v2.db"
    _build_v2_db(db_file)

    backend1 = SQLiteBackend(db_file)
    backend1.close()

    backend2 = SQLiteBackend(db_file)
    try:
        conn = backend2._get_conn()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[
            "version"
        ]
        assert version == 3

        # FTS still works after second open.
        results = backend2.query_fts("hello", top_k=10)
        assert any(d.id == "ascii1" for d in results)
    finally:
        backend2.close()


def test_fresh_db_lands_at_v3(tmp_path: Path) -> None:
    """A brand-new DB created by SQLiteBackend starts at the current version."""
    db_file = tmp_path / "fresh.db"
    backend = SQLiteBackend(db_file)
    try:
        conn = backend._get_conn()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[
            "version"
        ]
        assert version == 3
    finally:
        backend.close()


def test_v2_db_with_sync_residue_still_opens(tmp_path: Path) -> None:
    """Forward-compat: legacy DBs that still carry the removed sync columns open.

    Pre-Task-23.1 v2 DBs may have a ``dirty`` column, an ``idx_documents_dirty``
    index and a ``sync_state`` table. The current migration path ignores
    those artefacts and must still upgrade the DB to v3 and serve queries.
    """
    db_file = tmp_path / "v2_legacy.db"
    _build_v2_db(db_file, include_sync_residue=True)

    backend = SQLiteBackend(db_file)
    try:
        conn = backend._get_conn()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[
            "version"
        ]
        assert version == 3

        # FTS still works after upgrading a DB with sync residue.
        results = backend.query_fts("hello", top_k=10)
        assert any(d.id == "ascii1" for d in results)
        cjk_hits = backend.query_fts("喜欢编", top_k=10)
        assert any(d.id == "cjk1" for d in cjk_hits)
    finally:
        backend.close()
