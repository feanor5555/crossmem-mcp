"""FTS5-injection / special-character hardening for the SQLite backend.

The FTS5 query parser treats characters like ``"``, ``(``, ``)``, ``*``,
``:``, ``-``, ``OR``, ``NOT`` and ``AND`` as syntax. An unescaped user
query can therefore raise ``sqlite3.OperationalError`` ("fts5: syntax
error near ...") or, in the worst case, alter the search semantics.

``SQLiteBackend.query_fts`` MUST defend against this by wrapping the
input in double quotes and escaping any embedded ``"`` as ``""`` so the
query is treated as a literal phrase. These tests exercise a battery of
hostile payloads and assert:

  1. No ``sqlite3.OperationalError`` (or any other) is raised.
  2. The return value is always a ``list`` (possibly empty).
  3. Bombs that look like FTS5 wildcards (``*``, ``OR``, ``NOT``) do not
     accidentally match unrelated documents.
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import Document, Metadata


def _doc(doc_id: str, content: str, tags: list[str] | None = None) -> Document:
    embedding = [float(b) / 255.0 for b in hashlib.sha256(content.encode()).digest()]
    embedding = (embedding * ((EMBEDDING_DIM // len(embedding)) + 1))[:EMBEDDING_DIM]
    return Document(
        id=doc_id,
        content=content,
        embedding=embedding,
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=f"Title {doc_id}",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=EMBEDDING_DIM,
            namespace="default",
            tags=tags or [],
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        ),
    )


# Payloads cover: SQL-style injection, FTS5 syntax tokens, raw quotes,
# wildcards, column qualifiers, parens, NEAR operator, NOT/OR/AND.
INJECTION_PAYLOADS: list[str] = [
    'foo" OR 1=1 --',
    'foo" OR "1"="1',
    '"',
    '""',
    '"""',
    "'",
    "(",
    ")",
    "()",
    "(((",
    "*",
    "**",
    "* *",
    ":",
    "content:foo",
    "tags:python",
    "foo OR bar",
    "foo AND bar",
    "foo NOT bar",
    "NEAR(foo bar)",
    "foo NEAR/3 bar",
    "-foo",
    "+foo",
    "^foo",
    "foo^",
    "foo bar baz",
    "  ",
    "",
    "\\",
    '\\"',
    "; DROP TABLE documents; --",
    "' OR '' = '",
    "中文测试",
    "русский",
    "🚀 emoji 💀",
    "AND OR NOT NEAR",
    'col:"value"',
    "MATCH 'foo'",
]


@pytest.fixture
def backend() -> SQLiteBackend:
    """Backend pre-seeded with a few harmless docs."""
    b = SQLiteBackend(":memory:")
    b.store(_doc("d1", "Python guide for beginners", tags=["python"]))
    b.store(_doc("d2", "Rust async programming", tags=["rust"]))
    b.store(_doc("d3", "Go concurrency patterns", tags=["go"]))
    yield b
    b.close()


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_query_fts_handles_injection_without_crash(
    backend: SQLiteBackend, payload: str
) -> None:
    """Every hostile payload returns a list and never raises."""
    try:
        results = backend.query_fts(payload, top_k=10)
    except sqlite3.OperationalError as exc:  # pragma: no cover — explicit fail
        pytest.fail(f"FTS5 raised OperationalError on payload {payload!r}: {exc}")
    assert isinstance(results, list)


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_query_fts_with_tags_handles_injection_without_crash(
    backend: SQLiteBackend, payload: str
) -> None:
    """Same battery, but with a tag pre-filter (different code path)."""
    try:
        results = backend.query_fts(payload, top_k=10, tags=["python"])
    except sqlite3.OperationalError as exc:  # pragma: no cover — explicit fail
        pytest.fail(
            f"FTS5 raised OperationalError on payload {payload!r} with tags: {exc}"
        )
    assert isinstance(results, list)


def test_query_fts_wildcard_does_not_match_everything(backend: SQLiteBackend) -> None:
    """A bare ``*`` must NOT degrade into "match every doc" semantics.

    With proper phrase quoting, ``"*"`` is searched as the literal
    character ``*`` — it should not match documents whose content is
    purely natural language.
    """
    results = backend.query_fts("*", top_k=10)
    assert isinstance(results, list)
    # None of the seeded docs contain a literal '*' character.
    assert results == []


def test_query_fts_or_does_not_become_logical_or(backend: SQLiteBackend) -> None:
    """``foo OR bar`` is treated as a literal phrase, not as Boolean OR."""
    results = backend.query_fts("python OR rust", top_k=10)
    assert isinstance(results, list)
    # None of the seeded docs literally contain the phrase "python OR rust".
    assert results == []


def test_query_fts_does_not_drop_table(backend: SQLiteBackend) -> None:
    """Classic SQL injection is harmless because params are bound, not
    interpolated; the documents table must still exist after the query."""
    backend.query_fts("'; DROP TABLE documents; --", top_k=5)
    # The store still works after the attempted injection.
    backend.store(_doc("d4", "still alive"))
    assert backend.query_fts("alive", top_k=5)
