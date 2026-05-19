"""Edge-case integration tests for the SQLite backend + KnowledgeStore.

Covers behaviours that are easy to overlook in unit tests:

* fresh / empty database (no crash, no spurious results)
* Unicode payloads across CJK, emoji + combining marks, and RTL scripts
* large content (1 MB) round-trip with multi-chunk storage
* concurrent writers against a WAL-mode file-based DB
* embedding dimension mismatch is rejected with a clear exception

The tests use a shared mock embedder per test case (no fastembed download)
and exercise the real :class:`SQLiteBackend` through the
:class:`KnowledgeStore` facade so the FTS5 + sqlite-vec layer is part of
the assertion surface.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata, generate_content_hash
from crossmem.core.store import KnowledgeStore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(vec: list[float] | None = None) -> MagicMock:
    """Return a mock embedder that always emits a 384-dim vector."""
    embedder = MagicMock()
    embedder.model_name = "mock-edge-cases"
    fixed = vec if vec is not None else [0.1] * 384
    embedder.embed_passage.return_value = fixed
    embedder.embed_query.return_value = fixed
    embedder.embed_passage_batch.side_effect = lambda texts, batch_size=32: (
        [fixed] * len(texts)
    )
    return embedder


def _make_store(backend: SQLiteBackend) -> KnowledgeStore:
    return KnowledgeStore(backend=backend, embedder=_make_embedder())


# ---------------------------------------------------------------------------
# 1. Empty DB
# ---------------------------------------------------------------------------


def test_empty_db_query_returns_empty() -> None:
    """A fresh DB returns ``[]`` for any query and reports zero docs."""
    backend = SQLiteBackend(":memory:")
    try:
        store = _make_store(backend)

        assert store.query("foo") == []
        # An empty query string must not crash either — FTS5 escaping has to
        # cope with the degenerate input.
        assert store.query("") == []

        stats = store.stats()
        assert stats["document_count"] == 0
        assert stats["top_tags"] == []
        assert stats["backend"] == "sqlite"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 2. Unicode — CJK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "fts_query"),
    [
        # Chinese (Simplified). 'trigram' tokenizer needs >= 3-char queries
        # for CJK, so we feed a 3-char substring of the stored content.
        ("你好世界 hybrid search 中文测试内容", "中文测"),
        # Japanese.
        ("こんにちは世界 これは日本語のテストです", "日本語"),
        # Korean.
        ("안녕하세요 세계 한국어 검색 테스트입니다", "한국어"),
    ],
)
def test_unicode_cjk_chinese_japanese_korean(content: str, fts_query: str) -> None:
    """Store + FTS-recall CJK content; bytes survive the UTF-8 round-trip."""
    backend = SQLiteBackend(":memory:")
    try:
        store = _make_store(backend)
        url = f"https://example.com/{abs(hash(content))}"
        ids = store.store(
            content=content,
            source_url=url,
            title="cjk doc",
            source_type="web",
        )
        assert ids, "store() must return at least one id for non-empty content"

        # FTS recall via a 3-char CJK substring (trigram requirement).
        results = backend.query_fts(fts_query, top_k=5)
        assert any(r.id in ids for r in results), (
            f"FTS5 did not recall the {fts_query!r} substring"
        )

        # UTF-8 round-trip: the stored content must equal the input verbatim.
        recovered = backend.get_by_url(url)
        assert recovered, "get_by_url must find the freshly stored doc"
        assert content in recovered[0].content
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 3. Unicode — emoji + combining marks
# ---------------------------------------------------------------------------


def test_unicode_emoji_and_combining_marks() -> None:
    """Emoji and decomposed combining marks survive store + retrieve."""
    backend = SQLiteBackend(":memory:")
    try:
        store = _make_store(backend)

        # Emoji surround an ASCII word.
        emoji_content = "\U0001f525 search \U0001f4af relevance ranking"
        emoji_url = "https://example.com/emoji"
        store.store(
            content=emoji_content,
            source_url=emoji_url,
            title="emoji",
            source_type="web",
        )
        recovered = backend.get_by_url(emoji_url)
        assert recovered
        assert "\U0001f525" in recovered[0].content
        assert "\U0001f4af" in recovered[0].content

        # FTS5 still finds the ASCII word neighbouring the emoji.
        results = backend.query_fts("search", top_k=5)
        assert any(r.metadata.source_url == emoji_url for r in results)

        # Decomposed 'café' (e + U+0301 combining acute) — the bytes must
        # survive untouched. The trigram tokenizer with remove_diacritics
        # also matches the precomposed form.
        decomposed = "café au lait"  # 'café' decomposed
        precomposed = "café"  # 'café' precomposed
        deco_url = "https://example.com/cafe-decomposed"
        store.store(
            content=decomposed,
            source_url=deco_url,
            title="cafe",
            source_type="web",
        )
        recovered = backend.get_by_url(deco_url)
        assert recovered
        assert decomposed in recovered[0].content
        # The combining-mark form is not byte-equal to the precomposed form.
        assert precomposed not in decomposed
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 4. Unicode — RTL (Arabic + Hebrew)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "fts_query"),
    [
        ("مرحبا بالعالم هذا اختبار البحث العربي", "مرحبا"),
        ("שלום עולם זוהי בדיקת חיפוש בעברית", "שלום"),
    ],
)
def test_unicode_rtl_arabic_hebrew(content: str, fts_query: str) -> None:
    """Arabic and Hebrew round-trip through SQLite + FTS5."""
    backend = SQLiteBackend(":memory:")
    try:
        store = _make_store(backend)
        url = f"https://example.com/rtl/{abs(hash(content))}"
        ids = store.store(
            content=content,
            source_url=url,
            title="rtl doc",
            source_type="web",
        )
        assert ids

        recovered = backend.get_by_url(url)
        assert recovered
        assert content in recovered[0].content

        results = backend.query_fts(fts_query, top_k=5)
        assert any(r.id in ids for r in results), (
            f"FTS5 did not recall the RTL substring {fts_query!r}"
        )
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 5. 1 MB content -> multi-chunk
# ---------------------------------------------------------------------------


def _lorem_ipsum_bytes(target_bytes: int) -> str:
    """Generate distinct prose of roughly ``target_bytes`` bytes (UTF-8)."""
    sentences: list[str] = []
    written = 0
    i = 0
    while written < target_bytes:
        s = (
            f"Sentence {i} discusses topic {i} with detail {i} and context "
            f"word_{i} for chunk diversity in this prose paragraph. "
        )
        sentences.append(s)
        written += len(s.encode("utf-8"))
        i += 1
    return "".join(sentences)[:target_bytes]


def test_1mb_content_chunked_correctly() -> None:
    """1 MB of prose splits into many chunks and remains queryable."""
    backend = SQLiteBackend(":memory:")
    try:
        store = _make_store(backend)
        content = _lorem_ipsum_bytes(1024 * 1024)
        assert len(content.encode("utf-8")) >= 1024 * 1024 - 256  # ~1MB

        import time

        t0 = time.monotonic()
        ids = store.store(
            content=content,
            source_url="https://example.com/large",
            title="Large",
            source_type="web",
        )
        elapsed = time.monotonic() - t0

        assert len(ids) > 1, "1 MB prose must produce more than one chunk"
        # Sanity: the count is reflected in stats.
        assert store.stats()["document_count"] == len(ids)
        # Soft performance budget — way above the realistic 1-2s, but well
        # below the 60s pytest timeout. Mock embedder so this is pure I/O.
        assert elapsed < 30.0, f"1 MB store took {elapsed:.2f}s — too slow"

        # FTS recall: a token guaranteed to appear in the generated text.
        results = backend.query_fts("Sentence", top_k=5)
        assert results
        assert any(r.id in ids for r in results)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 6. Concurrent writes (WAL)
# ---------------------------------------------------------------------------


def test_concurrent_writes_wal_mode(tmp_path: Path) -> None:
    """Three threads each write 50 docs to one WAL DB without locking errors."""
    db_file = tmp_path / "concurrent.db"

    # Pre-create the schema so worker threads only do INSERTs (otherwise the
    # idempotent CREATE TABLE statements race for the writer slot during
    # init and one thread can hit ``database is locked``).
    SQLiteBackend(db_file).close()

    threads_count = 3
    docs_per_thread = 50
    barrier = threading.Barrier(threads_count)
    errors: list[tuple[int, str]] = []

    def worker(tid: int) -> None:
        try:
            backend = SQLiteBackend(db_file)
            barrier.wait()  # all threads start writing simultaneously
            for i in range(docs_per_thread):
                content = f"thread {tid} doc {i} unique payload"
                doc = Document(
                    id=f"t{tid}_d{i}",
                    content=content,
                    embedding=[0.1 + tid * 0.01 + i * 0.001] * 384,
                    metadata=Metadata(
                        source_url=f"https://example.com/t{tid}/{i}",
                        title=f"Doc {i}",
                        source_type="web",
                        stored_at="2024-01-01T00:00:00Z",
                        embedding_model="test",
                        embedding_dim=384,
                        namespace="default",
                        tags=["concurrent", f"t{tid}"],
                        content_hash=generate_content_hash(content),
                    ),
                )
                backend.store(doc)
            backend.close()
        except Exception as exc:  # noqa: BLE001 - reported back via list
            errors.append((tid, f"{type(exc).__name__}: {exc}"))

    threads = [
        threading.Thread(target=worker, args=(tid,)) for tid in range(threads_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writers raised: {errors}"

    # All docs landed.
    backend = SQLiteBackend(db_file)
    try:
        stats = backend.stats()
        assert stats["document_count"] == threads_count * docs_per_thread

        # WAL really is enabled on the file-based DB.
        mode = backend._get_conn().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 7. Dimension mismatch
# ---------------------------------------------------------------------------


def test_dimension_mismatch_rejected() -> None:
    """A 256-dim embedding into a 384-dim vec0 column raises a clear error.

    sqlite-vec enforces the column dimension declared in
    ``vec0(embedding float[384])`` and surfaces the mismatch as a
    ``sqlite3.OperationalError`` with a human-readable message that names
    both dimensions. The test pins the exception type and the message
    shape so a future silent acceptance regression would fail loudly.
    """
    import sqlite3

    backend = SQLiteBackend(":memory:")
    try:
        bad_content = "wrong dim doc"
        wrong = Document(
            id="wrongdim",
            content=bad_content,
            embedding=[0.1] * 256,  # default dim is 384
            metadata=Metadata(
                source_url="https://example.com/wrong-dim",
                title="Bad",
                source_type="web",
                stored_at="2024-01-01T00:00:00Z",
                embedding_model="bad-model",
                embedding_dim=256,
                namespace="default",
                tags=[],
                content_hash=generate_content_hash(bad_content),
            ),
        )

        with pytest.raises(sqlite3.OperationalError) as exc_info:
            backend.store(wrong)
        msg = str(exc_info.value).lower()
        assert "dimension" in msg
        assert "384" in msg and "256" in msg

        # Backend stays usable: the failed insert did not corrupt state and
        # subsequent stores with the correct dimension still work.
        good = Document(
            id="gooddim",
            content="good dim doc",
            embedding=[0.1] * 384,
            metadata=Metadata(
                source_url="https://example.com/good-dim",
                title="Good",
                source_type="web",
                stored_at="2024-01-01T00:00:00Z",
                embedding_model="good-model",
                embedding_dim=384,
                namespace="default",
                tags=[],
                content_hash=generate_content_hash("good dim doc"),
            ),
        )
        backend.store(good)
        assert backend.stats()["document_count"] == 1
    finally:
        backend.close()
