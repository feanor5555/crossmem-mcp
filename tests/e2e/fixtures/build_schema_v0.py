"""Generate ``tests/e2e/fixtures/schema_v0.db`` (task 27.18).

The committed binary lives next to this script so the
``schema_migration`` E2E scenario can copy a known pre-``schema_version``
SQLite layout into its fake-home and observe ``crossmem``'s in-place
migration on first backend open. We keep the builder around (rather
than checking in a one-off blob) so:

* the layout is reproducible from source on any platform when the
  pinned sqlite-vec / SQLite versions change;
* reviewers can read what "old layout" means without having to crack
  the binary open with ``sqlite3``;
* a future v3 -> v4 migration test can fork this builder for its own
  pre-v3 fixture without re-deriving the schema from prose.

The fixture intentionally:

* omits the ``schema_version`` table entirely (the heart of the
  "pre-versioned" world);
* uses the historic default FTS5 tokenizer (no ``tokenize=`` clause),
  which the v2 -> v3 migration must rebuild;
* seeds two documents with deterministic embeddings so the scenario
  can assert post-migration readability without depending on the
  fastembed model.

Running ``python tests/e2e/fixtures/build_schema_v0.py`` rewrites the
``schema_v0.db`` sibling in place. The output is deterministic for a
given Python/SQLite/sqlite-vec stack; small page-layout differences
across major SQLite versions are expected and acceptable — the
scenario asserts *behaviour*, not byte identity.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Make the builder runnable from a source checkout without installing
# crossmem (mirrors the PYTHONPATH guidance in CLAUDE.md so contributors
# can regenerate the fixture in a clean clone).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import sqlite_vec  # noqa: E402 — path adjusted above

from crossmem.core.embedding import EMBEDDING_DIM  # noqa: E402

FIXTURE_PATH = Path(__file__).with_name("schema_v0.db")


def build(target: Path = FIXTURE_PATH) -> None:
    target.unlink(missing_ok=True)
    # Use small page_size + DELETE journal mode for a compact, single-file
    # fixture. ``SQLiteBackend._init_db`` re-applies its own PRAGMAs
    # (WAL + page_size=8192) on open, but page_size is only honoured for
    # a brand-new DB — the existing pages keep their original size,
    # which is fine: SQLite handles mixed-page-size workloads via
    # ``VACUUM`` if a future migration ever needs uniform pages.
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA page_size = 4096")
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # NO ``schema_version`` table — that is the whole point of the
    # fixture. ``SQLiteBackend._init_db`` must observe its absence and
    # bootstrap the row to ``1`` before the v2 -> v3 step runs.
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
    # Historic tokenizer: no explicit ``tokenize=`` clause => unicode61
    # default. The v3 migration drops this and rebuilds with trigram.
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

    # Two seed docs. Content carries the literal token ``test`` so the
    # scenario's ``test``-FTS probe finds it after the trigram rebuild.
    rows = [
        (
            "legacy-doc-1",
            "This is a legacy test document about python asyncio.",
            "https://example.com/legacy-1",
            "Legacy Doc One",
            "web",
            "2024-01-01T00:00:00Z",
            "legacy-model",
            EMBEDDING_DIM,
            "default",
            '["python", "asyncio"]',
            "hash-legacy-1",
        ),
        (
            "legacy-doc-2",
            "Another legacy test entry mentioning sqlite and migrations.",
            "https://example.com/legacy-2",
            "Legacy Doc Two",
            "web",
            "2024-01-02T00:00:00Z",
            "legacy-model",
            EMBEDDING_DIM,
            "default",
            '["sqlite", "migration"]',
            "hash-legacy-2",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO documents
        (id, content, source_url, title, source_type, stored_at,
         embedding_model, embedding_dim, namespace, tags, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    # Mirror the FTS5 ``external-content`` contract: writers must
    # explicitly insert into the FTS table for it to see the row.
    # The v3 migration rebuilds this index, so the values written here
    # are only used to prove pre-migration content survives the rebuild.
    conn.executemany(
        "INSERT INTO fts_documents (rowid, content, title, tags) "
        "SELECT rowid, content, title, tags FROM documents WHERE id = ?",
        [(rows[0][0],), (rows[1][0],)],
    )
    # The ``vec_documents`` virtual table is intentionally left empty:
    # the v3 migration only rebuilds ``fts_documents`` and never touches
    # vec rows, so seeding embeddings here would only inflate the
    # committed fixture (vec0 reserves a fixed BLOB slot per row).
    # The scenario asserts FTS readability — that is the migration's
    # actual concern.

    conn.executemany(
        "INSERT INTO document_tags (doc_id, tag) VALUES (?, ?)",
        [
            ("legacy-doc-1", "python"),
            ("legacy-doc-1", "asyncio"),
            ("legacy-doc-2", "sqlite"),
            ("legacy-doc-2", "migration"),
        ],
    )

    conn.commit()
    conn.close()
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
