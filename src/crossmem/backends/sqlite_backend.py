"""SQLite backend using sqlite-vec for vectors and FTS5 for full-text search."""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite_vec

from crossmem.backends.base import BackendStats, VectorStoreBase
from crossmem.core.embedding import EMBEDDING_DIM

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crossmem.core.models import Document


class SQLiteBackend(VectorStoreBase):
    """SQLite + sqlite-vec + FTS5 backend (default, zero-config).

    Threading model: a single shared ``sqlite3.Connection`` is opened with
    ``check_same_thread=False`` and serialised by ``self._lock`` (an
    ``RLock``). This lets FastMCP dispatch tool calls onto worker threads
    without tripping sqlite3's per-thread-affinity ``ProgrammingError``. WAL
    journalling keeps concurrent readers fast on disk; the RLock keeps
    writers consistent and protects multi-statement transactions
    (``store``/``delete``).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        # RLock so a method that already holds the lock can re-enter without
        # deadlocking (defensive: keeps refactors safe even if ``_get_conn``
        # is later called from inside a locked region).
        self._lock = threading.RLock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the shared connection, lazily creating it on first use.

        Contract (TODO 26.18, variant a): this method acquires
        ``self._lock`` itself. The lock is an ``RLock``, so callers that
        already hold it (``store``, ``query_vector``, ...) re-enter
        without deadlocking. The acquisition serialises the
        first-time connection creation against any other thread that
        races on the first call, and keeps the public contract simple:
        "``_get_conn`` is always safe to call".
        """
        with self._lock:
            if self._conn is None:
                # ``check_same_thread=False`` allows the connection to be used
                # from any thread; ``self._lock`` provides the serialisation
                # sqlite3 would otherwise enforce per-thread.
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()

        # PRAGMAs — WAL and page_size only for file-based DBs
        if self._db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA page_size = 8192")
            # Restrict the on-disk DB file to owner read/write only on
            # Linux/Mac (cross-platform rule #4). Windows POSIX modes are
            # a no-op so we skip the chmod entirely.
            if sys.platform != "win32":
                os.chmod(self._db_path, 0o600)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Schema versioning
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)

        version_row = conn.execute("SELECT version FROM schema_version").fetchone()
        if version_row is None:
            self._create_schema_v1(conn)
            conn.execute("INSERT INTO schema_version VALUES (1)")
            conn.commit()
            current_version = 1
        else:
            current_version = version_row["version"]

        # Idempotent migrations — each step bumps schema_version on success.
        # The historic v1->v2 migration added sync-state columns that the
        # current schema no longer touches; we skip straight to v3 and let
        # any leftover ``dirty``/``sync_state`` artefacts in pre-existing
        # DBs sit idle. Tokenizer-rebuild (v2->v3) is still required.
        if current_version < 3:
            self._migrate_v2_to_v3(conn)
            conn.execute("UPDATE schema_version SET version = 3")
            conn.commit()

    def _create_schema_v1(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
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
        """)

        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)

        # Tokenizer: 'trigram' enables CJK substring search (each token is a
        # 3-char window) and diacritics-insensitive matching. 'unicode61' treats
        # consecutive CJK chars as a single token, so substring queries miss.
        # Existing v2 DBs are upgraded by ``_migrate_v2_to_v3``.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents USING fts5(
                content, title, tags,
                content='documents', content_rowid='rowid',
                tokenize='trigram remove_diacritics 1'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_tags (
                doc_id TEXT NOT NULL,
                tag TEXT NOT NULL
            )
        """)

        # Indices
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_url ON documents(source_url)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON documents(namespace)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stored_at ON documents(stored_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag ON document_tags(tag)")

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        """Rebuild ``fts_documents`` with the trigram tokenizer.

        v1 DBs created the FTS5 table without an explicit ``tokenize=`` clause,
        so they default to ``unicode61`` and miss CJK substring queries. v3
        drops and recreates the table with ``trigram remove_diacritics 1`` and
        re-indexes from ``documents``.

        Idempotent: introspects the existing table's SQL and skips the rebuild
        if the trigram tokenizer is already in place.
        """
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='fts_documents'"
        ).fetchone()
        if existing is not None and "trigram" in (existing["sql"] or "").lower():
            return

        conn.execute("DROP TABLE IF EXISTS fts_documents")
        conn.execute("""
            CREATE VIRTUAL TABLE fts_documents USING fts5(
                content, title, tags,
                content='documents', content_rowid='rowid',
                tokenize='trigram remove_diacritics 1'
            )
        """)
        # External-content FTS5 tables are empty after creation; repopulate
        # from ``documents`` so historic rows become searchable again.
        conn.execute(
            "INSERT INTO fts_documents (rowid, content, title, tags) "
            "SELECT rowid, content, title, tags FROM documents"
        )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def store(self, doc: Document) -> None:
        """Store a single document atomically (thread-safe)."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                self._write_doc(conn, doc)

    def upsert_many(self, docs: list[Document]) -> None:
        """Persist every document in ``docs`` inside one transaction.

        If any write fails, the connection's context manager rolls back
        the entire batch — callers observe an empty ``get_by_url`` result
        rather than a partially-stored document.
        """
        if not docs:
            return
        with self._lock:
            conn = self._get_conn()
            with conn:
                for doc in docs:
                    self._write_doc(conn, doc)

    def _write_doc(self, conn: sqlite3.Connection, doc: Document) -> None:
        """Write one document inside an already-open transaction.

        Callers wrap the invocation in ``with conn:`` so a raised exception
        rolls back every statement issued in the same block — this is what
        makes :meth:`upsert_many` atomic for multi-chunk stores.
        """
        tags_json = json.dumps(doc.metadata.tags)
        embedding_bytes = struct.pack(f"{len(doc.embedding)}f", *doc.embedding)

        # Check if exists (for upsert)
        existing = conn.execute(
            "SELECT rowid FROM documents WHERE id = ?", (doc.id,)
        ).fetchone()

        if existing:
            old_rowid = existing[0]
            # Delete from virtual tables first
            conn.execute("DELETE FROM vec_documents WHERE rowid = ?", (old_rowid,))
            conn.execute("DELETE FROM fts_documents WHERE rowid = ?", (old_rowid,))
            conn.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc.id,))

        # Insert/replace document
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.id,
                doc.content,
                doc.metadata.source_url,
                doc.metadata.title,
                doc.metadata.source_type,
                doc.metadata.stored_at,
                doc.metadata.embedding_model,
                doc.metadata.embedding_dim,
                doc.metadata.namespace,
                tags_json,
                doc.metadata.content_hash,
            ),
        )

        # Get the rowid
        rowid = conn.execute(
            "SELECT rowid FROM documents WHERE id = ?", (doc.id,)
        ).fetchone()[0]

        # Insert embedding
        conn.execute(
            "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
            (rowid, embedding_bytes),
        )

        # Insert into FTS
        conn.execute(
            "INSERT INTO fts_documents (rowid, content, title, tags)"
            " VALUES (?, ?, ?, ?)",
            (rowid, doc.content, doc.metadata.title, tags_json),
        )

        # Insert tags
        for tag in doc.metadata.tags:
            conn.execute(
                "INSERT INTO document_tags (doc_id, tag) VALUES (?, ?)",
                (doc.id, tag),
            )

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a sqlite3.Row to a Document object."""
        from crossmem.core.models import Document, Metadata

        return Document(
            id=row["id"],
            content=row["content"],
            embedding=[],  # Embedding nicht bei jedem Query laden
            metadata=Metadata(
                source_url=row["source_url"],
                title=row["title"] or "",
                source_type=row["source_type"] or "",
                stored_at=row["stored_at"] or "",
                embedding_model=row["embedding_model"] or "",
                embedding_dim=row["embedding_dim"] or 0,
                namespace=row["namespace"] or "default",
                tags=json.loads(row["tags"]) if row["tags"] else [],
                content_hash=row["content_hash"] or "",
            ),
        )

    def query_vector(self, embedding: list[float], top_k: int = 10) -> list[Document]:
        """Query documents by vector similarity (thread-safe)."""
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT d.* FROM documents d
                INNER JOIN (
                    SELECT rowid, distance FROM vec_documents
                    WHERE embedding MATCH ? AND k = ?
                ) v ON d.rowid = v.rowid
                ORDER BY v.distance ASC
                """,
                (embedding_bytes, top_k),
            ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def query_fts(
        self, text: str, top_k: int = 10, tags: list[str] | None = None
    ) -> list[Document]:
        """Query documents by full-text search (thread-safe).

        Empty or whitespace-only ``text`` short-circuits to ``[]`` without
        hitting FTS5. The quoted-empty-phrase fallback (``'""'``) happens to
        return zero rows today, but is tokenizer-dependent — a future
        tokenizer change could turn it into ``sqlite3.OperationalError``.
        Callers that pass empty input clearly mean "no text predicate", so
        we honour that explicitly.
        """
        if not text or not text.strip():
            return []

        # Escape FTS5 query — wrap in double quotes to treat as phrase
        escaped = '"' + text.replace('"', '""') + '"'

        with self._lock:
            conn = self._get_conn()
            if tags:
                # Pre-filter by tags via document_tags join
                placeholders = ",".join("?" * len(tags))
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT d.* FROM documents d
                    INNER JOIN fts_documents f ON d.rowid = f.rowid
                    INNER JOIN document_tags dt ON d.id = dt.doc_id
                    WHERE fts_documents MATCH ? AND dt.tag IN ({placeholders})
                    LIMIT ?
                    """,
                    (escaped, *tags, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT d.* FROM documents d
                    INNER JOIN fts_documents f ON d.rowid = f.rowid
                    WHERE fts_documents MATCH ?
                    LIMIT ?
                    """,
                    (escaped, top_k),
                ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID (thread-safe). No-op if id is unknown."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                existing = conn.execute(
                    "SELECT rowid FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()

                if existing is None:
                    return

                rowid = existing[0]
                conn.execute("DELETE FROM vec_documents WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM fts_documents WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def get_by_url(self, source_url: str) -> list[Document]:
        """Retrieve documents by source URL (exact match, thread-safe).

        Results are returned in ascending ``id`` order so callers see a
        stable sequence across stores: doc ids are deterministic content
        hashes, so sorting by id gives a backend-independent total order
        that matches the contract enforced by the Chroma and Qdrant
        backends.
        """
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM documents WHERE source_url = ? ORDER BY id ASC",
                (source_url,),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document by id with its embedding restored.

        Returns ``None`` if the id is unknown. The embedding is hydrated
        from ``vec_documents`` so soft-delete callers can serialise the
        full record to the trash JSONL without a second round-trip.
        Thread-safe via ``self._lock``.
        """
        from crossmem.core.models import Document, Metadata

        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                SELECT d.*, v.embedding AS embedding_blob
                FROM documents d
                INNER JOIN vec_documents v ON d.rowid = v.rowid
                WHERE d.id = ?
                """,
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        blob = row["embedding_blob"]
        dim = len(blob) // 4
        embedding = list(struct.unpack(f"{dim}f", blob))
        return Document(
            id=row["id"],
            content=row["content"],
            embedding=embedding,
            metadata=Metadata(
                source_url=row["source_url"],
                title=row["title"] or "",
                source_type=row["source_type"] or "",
                stored_at=row["stored_at"] or "",
                embedding_model=row["embedding_model"] or "",
                embedding_dim=row["embedding_dim"] or 0,
                namespace=row["namespace"] or "default",
                tags=json.loads(row["tags"]) if row["tags"] else [],
                content_hash=row["content_hash"] or "",
            ),
        )

    def iter_all(self) -> Iterator[Document]:
        """Yield every stored document with its embedding restored.

        The DB rows are materialised under ``self._lock`` so concurrent
        writers cannot perturb the snapshot; the generator then yields
        lock-free so callers can run arbitrary code per row.
        """
        from crossmem.core.models import Document, Metadata

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT d.*, v.embedding AS embedding_blob, d.rowid AS doc_rowid
                FROM documents d
                INNER JOIN vec_documents v ON d.rowid = v.rowid
                ORDER BY d.rowid ASC
                """
            ).fetchall()

        for row in rows:
            blob = row["embedding_blob"]
            dim = len(blob) // 4
            embedding = list(struct.unpack(f"{dim}f", blob))
            yield Document(
                id=row["id"],
                content=row["content"],
                embedding=embedding,
                metadata=Metadata(
                    source_url=row["source_url"],
                    title=row["title"] or "",
                    source_type=row["source_type"] or "",
                    stored_at=row["stored_at"] or "",
                    embedding_model=row["embedding_model"] or "",
                    embedding_dim=row["embedding_dim"] or 0,
                    namespace=row["namespace"] or "default",
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    content_hash=row["content_hash"] or "",
                ),
            )

    def find_by_tag(self, tag: str) -> Iterator[Document]:
        """Yield every document whose tag list contains ``tag`` exactly.

        Served by an ``INNER JOIN document_tags`` on the ``idx_tag``
        index; rows are materialised under ``self._lock`` so concurrent
        writers cannot perturb the snapshot, then yielded lock-free.
        """
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT DISTINCT d.* FROM documents d
                INNER JOIN document_tags dt ON d.id = dt.doc_id
                WHERE dt.tag = ?
                ORDER BY d.rowid ASC
                """,
                (tag,),
            ).fetchall()

        for row in rows:
            yield self._row_to_document(row)

    def stats(self) -> BackendStats:
        """Return store statistics (thread-safe)."""
        with self._lock:
            conn = self._get_conn()

            document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[
                0
            ]

            if self._db_path == ":memory:":
                page_count = conn.execute("PRAGMA page_count").fetchone()[0]
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                db_size_bytes = page_count * page_size
            else:
                db_path = Path(self._db_path)
                db_size_bytes = db_path.stat().st_size if db_path.exists() else 0

            tag_rows = conn.execute(
                """
                SELECT tag, COUNT(*) AS cnt FROM document_tags
                GROUP BY tag
                ORDER BY cnt DESC, tag ASC
                LIMIT 10
                """
            ).fetchall()

        top_tags = [(row["tag"], row["cnt"]) for row in tag_rows]

        return {
            "document_count": document_count,
            "db_size_bytes": db_size_bytes,
            "top_tags": top_tags,
            "backend": "sqlite",
        }
