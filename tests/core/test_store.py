"""Tests for KnowledgeStore facade (DI over backend + embedder)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from crossmem.core.chunking import Chunk
from crossmem.core.models import Document, Metadata, generate_content_hash, generate_id
from crossmem.core.store import KnowledgeStore

if TYPE_CHECKING:
    import pytest


def _make_store(
    embedding: list[float] | None = None,
    model_name: str = "intfloat/multilingual-e5-small",
) -> tuple[KnowledgeStore, MagicMock, MagicMock]:
    """Build a KnowledgeStore with mocked backend and embedder."""
    backend = MagicMock()
    embedder = MagicMock()
    embedder.model_name = model_name
    vec = embedding if embedding is not None else [0.1] * 384
    embedder.embed_passage.return_value = vec
    embedder.embed_query.return_value = vec
    # ``KnowledgeStore.store`` calls ``embed_passage_batch`` once per
    # document. The mock returns ``vec`` per input text so existing
    # assertions about per-chunk embeddings still hold without each test
    # having to wire up batch return values.
    embedder.embed_passage_batch.side_effect = lambda texts, batch_size=32: (
        [vec] * len(texts)
    )
    return KnowledgeStore(backend=backend, embedder=embedder), backend, embedder


def _stored_docs(backend: MagicMock) -> list[Document]:
    """Return every Document handed to ``backend.upsert_many`` in batch order.

    :meth:`KnowledgeStore.store` collapses one call's chunks into a single
    ``upsert_many`` invocation so the whole document lands atomically; this
    helper flattens those batches back into the order they were persisted.
    """
    out: list[Document] = []
    for call in backend.upsert_many.call_args_list:
        out.extend(call.args[0])
    return out


def test_store_empty_content_returns_empty_and_skips_embedding() -> None:
    """Empty content -> no chunks -> no embedder call and no backend write."""
    store, backend, embedder = _make_store()
    ids = store.store(
        content="",
        source_url="https://example.com/empty",
        title="Empty",
        source_type="web",
    )
    assert ids == []
    backend.upsert_many.assert_not_called()
    backend.store.assert_not_called()
    embedder.embed_passage_batch.assert_not_called()
    embedder.embed_passage.assert_not_called()


def test_store_returns_list_of_doc_ids() -> None:
    """store() returns a list of doc_id strings (one per emitted chunk)."""
    store, backend, _embedder = _make_store()
    ids = store.store(
        content="hello world",
        source_url="https://example.com",
        title="Example",
        source_type="web",
    )
    assert isinstance(ids, list)
    assert ids
    assert all(isinstance(i, str) for i in ids)
    stored = _stored_docs(backend)
    assert [d.id for d in stored] == ids


def test_store_calls_embed_passage_batch_once_with_all_chunks() -> None:
    """``embed_passage_batch`` must be invoked exactly once with every chunk."""
    store, backend, embedder = _make_store()
    store.store(
        content="hello world",
        source_url="https://example.com",
        title="Example",
        source_type="web",
    )
    stored = _stored_docs(backend)
    # One model invocation per document — not one per chunk.
    assert embedder.embed_passage_batch.call_count == 1
    batched_texts = embedder.embed_passage_batch.call_args.args[0]
    assert batched_texts == [d.content for d in stored]
    # The per-chunk ``embed_passage`` entry-point is not used by ``store()``.
    embedder.embed_passage.assert_not_called()
    embedder.embed_query.assert_not_called()


def test_store_calls_backend_upsert_many_once_per_document() -> None:
    """``backend.upsert_many`` runs exactly once per logical document."""
    store, backend, _embedder = _make_store()
    ids = store.store(
        content="hello world",
        source_url="https://example.com",
        title="Example",
        source_type="web",
    )
    stored = _stored_docs(backend)
    # One batch call per logical document, regardless of chunk count.
    assert backend.upsert_many.call_count == 1
    assert len(stored) == len(ids)
    for d in stored:
        assert isinstance(d, Document)


def test_store_single_chunk_id_is_deterministic() -> None:
    """For short content -> 1 chunk; doc.id mixes the chunk_index into the hash."""
    store, backend, _embedder = _make_store()
    content = "deterministic content"
    source_url = "https://example.com/page"
    ids = store.store(
        content=content,
        source_url=source_url,
        title="Page",
        source_type="web",
    )
    assert len(ids) == 1
    stored = _stored_docs(backend)
    assert len(stored) == 1
    doc = stored[0]
    expected_hash = generate_content_hash(doc.content)
    # Chunk index 0 is mixed into the ID so identical chunks at different
    # positions never collide; a single-chunk doc still derives deterministically.
    expected_id = generate_id("default", source_url, expected_hash, chunk_index=0)
    assert doc.id == expected_id == ids[0]
    assert doc.metadata.content_hash == expected_hash


def test_store_populates_metadata_fields() -> None:
    """Metadata must reflect arguments and embedding info."""
    embedding = [0.5] * 384
    store, backend, _embedder = _make_store(
        embedding=embedding, model_name="custom-model"
    )
    store.store(
        content="some content",
        source_url="https://docs.example.com/x",
        title="Doc Title",
        source_type="web",
    )
    stored = _stored_docs(backend)
    assert stored
    for doc in stored:
        meta = doc.metadata
        assert meta.source_url == "https://docs.example.com/x"
        assert meta.title == "Doc Title"
        assert meta.source_type == "web"
        assert meta.embedding_model == "custom-model"
        assert meta.embedding_dim == 384
        assert meta.namespace == "default"
        # Auto-tagging may add entries but never strips them.
        assert isinstance(meta.tags, tuple)
        assert isinstance(meta.stored_at, str) and meta.stored_at
        assert doc.embedding == tuple(embedding)


def test_store_default_namespace_when_not_specified() -> None:
    """Without explicit namespace argument, every chunk lands in 'default'."""
    store, backend, _embedder = _make_store()
    store.store(
        content="c",
        source_url="https://x.example",
        title="t",
        source_type="web",
    )
    for doc in _stored_docs(backend):
        assert doc.metadata.namespace == "default"


def test_store_custom_namespace_propagates_to_chunks() -> None:
    """Custom namespace propagates into every chunk Document and its ID."""
    store, backend, _embedder = _make_store()
    tags = ["mylabel"]
    store.store(
        content="c",
        source_url="https://x.example",
        title="t",
        source_type="web",
        namespace="alice",
        tags=tags,
    )
    stored = _stored_docs(backend)
    assert stored
    for i, doc in enumerate(stored):
        assert doc.metadata.namespace == "alice"
        assert "mylabel" in doc.metadata.tags
        # IDs are deterministic over (namespace, source_url, content_hash,
        # chunk_index) — the chunk index disambiguates duplicate-content chunks.
        expected_id = generate_id(
            "alice",
            "https://x.example",
            generate_content_hash(doc.content),
            chunk_index=i,
        )
        assert doc.id == expected_id


# ---------------------------------------------------------------------------
# 0.6.6 — chunking + auto-tagging integration
# ---------------------------------------------------------------------------


def _large_content(min_bytes: int = 5 * 1024) -> str:
    """Build prose content guaranteed to exceed the prose chunk window.

    Each sentence is unique (numbered) so resulting chunks have distinct
    content and therefore distinct deterministic IDs.
    """
    parts: list[str] = []
    i = 0
    text_len = 0
    while text_len < min_bytes:
        parts.append(
            f"Sentence number {i} discusses topic {i} with detail {i} and "
            f"context word_{i} for chunk diversity in this prose paragraph. "
        )
        text_len = sum(len(p.encode("utf-8")) for p in parts)
        i += 1
    return "".join(parts)


def test_store_5kb_content_yields_more_than_one_chunk() -> None:
    """5KB+ prose content must split into more than one chunk-document."""
    store, backend, _embedder = _make_store()
    big_content = _large_content()
    assert len(big_content.encode("utf-8")) >= 5 * 1024

    ids = store.store(
        content=big_content,
        source_url="https://example.com/big",
        title="Big",
        source_type="web",
    )
    assert len(ids) > 1
    stored = _stored_docs(backend)
    assert len(stored) == len(ids)
    # Every emitted chunk-document carries unique content & id.
    assert len({d.id for d in stored}) == len(stored)


def test_store_auto_tags_from_url_appear_in_every_chunk() -> None:
    """URL-derived auto-tags appear in every chunk in addition to caller-tags."""
    store, backend, _embedder = _make_store()
    caller_tags = ["mycustomtag"]
    big_content = _large_content()

    store.store(
        content=big_content,
        # MDN host -> "mdn"; "/python/" path mention -> "python"; "/v17/" -> ":17"
        source_url="https://developer.mozilla.org/python/v17/asyncio.html",
        title="Tutorial",
        source_type="web",
        tags=caller_tags,
    )
    stored = _stored_docs(backend)
    assert len(stored) > 1, "test requires multi-chunk content"

    for doc in stored:
        # caller-supplied tag preserved on every chunk
        assert "mycustomtag" in doc.metadata.tags
        # Auto-tags from URL host / path / version present on every chunk
        assert "mdn" in doc.metadata.tags
        assert "python" in doc.metadata.tags
        # URL ".../python/v17/..." yields a "python:17" pair tag
        assert "python:17" in doc.metadata.tags


def test_store_re_store_identical_content_is_idempotent() -> None:
    """Re-storing identical content yields the same IDs in the same order."""
    store, backend, _embedder = _make_store()
    args = {
        "content": _large_content(),
        "source_url": "https://example.com/idempotent",
        "title": "Idem",
        "source_type": "web",
        "tags": ["t1"],
    }
    ids_first = store.store(**args)
    backend.reset_mock()
    ids_second = store.store(**args)

    assert ids_first == ids_second
    # The deterministic ID derivation means two identical stores produce
    # identical Document IDs (re-stores overwrite at the backend layer).
    stored_second = _stored_docs(backend)
    assert [d.id for d in stored_second] == ids_second


def test_store_duplicate_chunk_content_does_not_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two chunks with identical content must yield distinct IDs.

    Pre-fix, ``Document.from_payload`` derived the id from
    ``(namespace, source_url, content_hash)``. Two chunks with the same
    content (boilerplate, high-overlap short prose) produced the same id
    and ``INSERT OR REPLACE`` overwrote the earlier chunk in the backend —
    a silent data loss bug. Mixing ``chunk_index`` into the id keeps every
    chunk uniquely addressable even when their content hashes collide.
    """
    store, backend, _embedder = _make_store()

    duplicate = "shared boilerplate sentence."

    def fake_chunk(content: str, source_type: str, *, title: str = "") -> list[Chunk]:
        return [
            Chunk(content=duplicate, chunk_index=0),
            Chunk(content=duplicate, chunk_index=1),
        ]

    monkeypatch.setattr("crossmem.core.store.chunk_content", fake_chunk)

    ids = store.store(
        content="ignored — patched chunker emits duplicates",
        source_url="https://example.com/dup",
        title="Dup",
        source_type="web",
    )

    assert len(ids) == 2, "two chunks must produce two ids"
    assert len(set(ids)) == 2, f"duplicate-content chunks must have distinct ids: {ids}"

    stored = _stored_docs(backend)
    assert len(stored) == 2
    assert {d.id for d in stored} == set(ids)
    # Both chunks share content + content_hash; only chunk_index disambiguates.
    assert stored[0].content == stored[1].content
    assert stored[0].metadata.content_hash == stored[1].metadata.content_hash


def test_store_duplicate_chunk_content_ids_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat store() with duplicate-content chunks yields the same ids."""
    store, backend, _embedder = _make_store()
    duplicate = "identical chunk body"

    def fake_chunk(content: str, source_type: str, *, title: str = "") -> list[Chunk]:
        return [
            Chunk(content=duplicate, chunk_index=0),
            Chunk(content=duplicate, chunk_index=1),
            Chunk(content=duplicate, chunk_index=2),
        ]

    monkeypatch.setattr("crossmem.core.store.chunk_content", fake_chunk)

    first = store.store(
        content="x",
        source_url="https://example.com/dup",
        title="Dup",
        source_type="web",
    )
    backend.reset_mock()
    second = store.store(
        content="x",
        source_url="https://example.com/dup",
        title="Dup",
        source_type="web",
    )
    assert first == second
    assert len(set(first)) == 3


def test_restore_forwards_doc_to_backend_without_re_embedding() -> None:
    """restore() must hand the Document to backend.store unchanged, no embed call."""
    store, backend, embedder = _make_store()
    fixed_embedding = [0.42] * 384
    doc = Document(
        id="restored-id",
        content="restored content",
        embedding=fixed_embedding,
        metadata=Metadata(
            source_url="https://example.com/restored",
            title="Restored",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=384,
            namespace="default",
            tags=["python"],
            content_hash="hash-restored",
        ),
    )

    store.restore(doc)

    backend.store.assert_called_once_with(doc)
    assert embedder.embed_passage.call_count == 0
    assert embedder.embed_query.call_count == 0
