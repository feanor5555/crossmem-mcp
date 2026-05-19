"""Smoke tests for the multilingual FTS5 tokenizer (Task 0.6.9).

These verify that the SQLite FTS5 tokenizer handles non-ASCII scripts:
- CJK substring search (Chinese / Japanese / Korean)
- Diacritics normalisation (café <-> cafe, naïve <-> naive)
- Emoji-adjacent words (emoji shouldn't break tokenisation)

The `trigram` tokenizer is used because `unicode61` treats consecutive CJK
characters as a single token, so substring queries miss. `trigram` requires
queries of >= 3 characters for MATCH (its tokens are 3-char windows).
"""

from __future__ import annotations

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata, generate_content_hash


def _make_doc(
    doc_id: str,
    content: str,
    title: str = "",
    tags: list[str] | None = None,
) -> Document:
    """Build a Document with deterministic embedding for FTS tests."""
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=title,
            source_type="web",
            stored_at="2024-01-01T00:00:00Z",
            embedding_model="test-model",
            embedding_dim=384,
            namespace="default",
            tags=tags or [],
            content_hash=generate_content_hash(content),
        ),
    )


def test_fts_finds_chinese_text() -> None:
    """CJK substring search returns the stored doc.

    Note: `trigram` requires queries of >= 3 characters. A 2-char query like
    "编程" cannot match via MATCH (no trigram exists for a 2-char string).
    Using a 3-char substring "喜欢编" (slice of the content) covers the
    multilingual goal: arbitrary CJK substrings must be findable.
    """
    backend = SQLiteBackend(":memory:")
    try:
        doc = _make_doc(
            doc_id="cjk1",
            content="今天天气很好，我喜欢编程",
            title="中文测试",
        )
        backend.store(doc)

        results = backend.query_fts("喜欢编", top_k=5)
        assert len(results) >= 1
        assert any(r.id == doc.id for r in results)

        # Also exercise a title-side match (>= 3 chars).
        results_title = backend.query_fts("中文测", top_k=5)
        assert any(r.id == doc.id for r in results_title)
    finally:
        backend.close()


def test_fts_finds_diacritics_normalized() -> None:
    """Diacritics are stripped: 'cafe' matches 'café', 'naive' matches 'naïve'."""
    backend = SQLiteBackend(":memory:")
    try:
        doc_cafe = _make_doc(doc_id="cafe1", content="café au lait")
        backend.store(doc_cafe)

        results = backend.query_fts("cafe", top_k=5)
        assert len(results) >= 1
        assert any(r.id == doc_cafe.id for r in results)

        doc_naive = _make_doc(doc_id="naive1", content="naïve thought")
        backend.store(doc_naive)

        results = backend.query_fts("naive", top_k=5)
        assert any(r.id == doc_naive.id for r in results)
    finally:
        backend.close()


def test_fts_finds_emoji_adjacent_word() -> None:
    """An emoji between words must not break tokenisation of the words."""
    backend = SQLiteBackend(":memory:")
    try:
        doc = _make_doc(doc_id="emoji1", content="hello \U0001f30d world")
        backend.store(doc)

        results = backend.query_fts("hello", top_k=5)
        assert len(results) >= 1
        assert any(r.id == doc.id for r in results)
    finally:
        backend.close()
