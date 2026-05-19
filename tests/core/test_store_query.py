"""Tests for KnowledgeStore.query — hybrid search via RRF."""

from __future__ import annotations

from unittest.mock import MagicMock

from crossmem.core.models import Document, Metadata
from crossmem.core.store import KnowledgeStore


def _make_doc(doc_id: str, tags: list[str] | None = None) -> Document:
    """Build a minimal Document for ranking tests."""
    return Document(
        id=doc_id,
        content=f"content of {doc_id}",
        embedding=[0.0] * 384,
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=doc_id,
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=384,
            namespace="default",
            tags=list(tags) if tags else [],
            content_hash=doc_id,
        ),
    )


def _make_store(
    fts_results: list[Document],
    vec_results: list[Document],
    embedding: list[float] | None = None,
) -> tuple[KnowledgeStore, MagicMock, MagicMock]:
    """Build a KnowledgeStore with mocked backend and embedder."""
    backend = MagicMock()
    backend.query_fts.return_value = fts_results
    backend.query_vector.return_value = vec_results
    embedder = MagicMock()
    vec = embedding if embedding is not None else [0.1] * 384
    embedder.embed_query.return_value = vec
    embedder.embed_passage.return_value = vec
    return KnowledgeStore(backend=backend, embedder=embedder), backend, embedder


def test_query_embeds_query_text_for_vector_search() -> None:
    """The query string must be embedded once and passed to query_vector."""
    embedding = [0.42] * 384
    store, backend, embedder = _make_store(
        fts_results=[], vec_results=[], embedding=embedding
    )
    store.query("hello world", top_k=5)
    embedder.embed_query.assert_called_once_with("hello world")
    embedder.embed_passage.assert_not_called()
    backend.query_vector.assert_called_once()
    passed_embedding, passed_top_k = (
        backend.query_vector.call_args.args[0],
        backend.query_vector.call_args.args[1]
        if len(backend.query_vector.call_args.args) > 1
        else backend.query_vector.call_args.kwargs.get("top_k"),
    )
    assert passed_embedding == embedding
    assert passed_top_k == 15  # top_k * 3


def test_query_calls_fts_with_tag_pre_filter() -> None:
    """tags must be forwarded to backend.query_fts as pre-filter."""
    store, backend, _embedder = _make_store(fts_results=[], vec_results=[])
    store.query("python asyncio", top_k=10, tags=["python", "python:3.12"])
    backend.query_fts.assert_called_once()
    kwargs = backend.query_fts.call_args.kwargs
    args = backend.query_fts.call_args.args
    # text is first positional
    assert args[0] == "python asyncio"
    # top_k = top_k * 3 = 30
    top_k_arg = args[1] if len(args) > 1 else kwargs.get("top_k")
    assert top_k_arg == 30
    # tags forwarded
    tags_arg = (
        kwargs.get("tags") if "tags" in kwargs else (args[2] if len(args) > 2 else None)
    )
    assert tags_arg == ["python", "python:3.12"]


def test_query_doc_in_both_lists_ranks_higher_than_single_list_doc() -> None:
    """RRF: a doc appearing in BOTH FTS and Vector outranks a single-list doc."""
    both = _make_doc("both")
    only_fts = _make_doc("only_fts")
    only_vec = _make_doc("only_vec")
    # Each list has 'both' at rank 1, and a unique doc at rank 0.
    fts_results = [only_fts, both]
    vec_results = [only_vec, both]
    store, _backend, _embedder = _make_store(
        fts_results=fts_results, vec_results=vec_results
    )
    results = store.query("q", top_k=3)
    assert [d.id for d in results][0] == "both"
    assert set(d.id for d in results) == {"both", "only_fts", "only_vec"}


def test_query_top_k_truncates_results() -> None:
    """Result list must be truncated to top_k."""
    docs = [_make_doc(f"d{i}") for i in range(5)]
    store, _backend, _embedder = _make_store(fts_results=docs, vec_results=docs)
    results = store.query("q", top_k=2)
    assert len(results) == 2
    # All docs share identical RRF score; truncation keeps insertion order.
    assert {d.id for d in results} <= {d.id for d in docs}


def test_query_post_filters_vector_results_by_tags() -> None:
    """Vector results without any of the requested tags are filtered out."""
    matching = _make_doc("vec_match", tags=["python"])
    non_matching = _make_doc("vec_skip", tags=["rust"])
    store, _backend, _embedder = _make_store(
        fts_results=[],
        vec_results=[non_matching, matching],
    )
    results = store.query("q", top_k=10, tags=["python"])
    ids = [d.id for d in results]
    assert "vec_match" in ids
    assert "vec_skip" not in ids


def test_query_returns_empty_when_both_backends_empty() -> None:
    """Empty FTS and Vector results yield an empty list."""
    store, _backend, _embedder = _make_store(fts_results=[], vec_results=[])
    assert store.query("anything", top_k=10) == []


def test_query_dedupes_doc_appearing_in_both_lists() -> None:
    """A doc in both lists appears only once in the final result."""
    both = _make_doc("both")
    store, _backend, _embedder = _make_store(fts_results=[both], vec_results=[both])
    results = store.query("q", top_k=10)
    assert len(results) == 1
    assert results[0].id == "both"


def test_query_default_top_k_is_ten() -> None:
    """Default top_k is 10 (per spec signature)."""
    docs = [_make_doc(f"d{i}") for i in range(20)]
    store, backend, _embedder = _make_store(fts_results=docs, vec_results=docs)
    results = store.query("q")
    assert len(results) == 10
    # top_k * 3 = 30 was requested from each backend
    fts_top_k = backend.query_fts.call_args.args[1]
    vec_top_k = backend.query_vector.call_args.args[1]
    assert fts_top_k == 30
    assert vec_top_k == 30


def test_query_no_tags_does_not_post_filter_vector() -> None:
    """Without tags, vector results are not post-filtered."""
    a = _make_doc("a", tags=["x"])
    b = _make_doc("b", tags=[])
    store, _backend, _embedder = _make_store(fts_results=[], vec_results=[a, b])
    results = store.query("q", top_k=10)
    ids = {d.id for d in results}
    assert ids == {"a", "b"}


def test_query_widens_vector_window_to_surface_on_tag_hits() -> None:
    """Tag-filtered query must surface on-tag hits beyond the first top_k*3 vec hits.

    Regression for 24.9: the previous implementation requested exactly
    ``top_k * 3`` candidates from the vector backend and post-filtered by
    tags. If those candidates were dominated by off-tag docs, on-tag hits
    that ranked just past the window were silently dropped — even when the
    backend held plenty of matches further down the rank. The store must
    therefore widen the vector fetch until enough on-tag hits surface or
    the backend is exhausted.
    """
    top_k = 5
    initial_fetch = top_k * 3  # 15
    # Backend "data": first 15 vector hits are off-tag rust docs; the next
    # 5 are on-tag python docs. A naive top_k*3 fetch would surface zero
    # matches; the widened fetch must reach the python docs.
    off_tag = [_make_doc(f"rust_{i}", tags=["rust"]) for i in range(initial_fetch)]
    on_tag = [_make_doc(f"py_{i}", tags=["python"]) for i in range(5)]
    full = off_tag + on_tag

    def fake_query_vector(_embedding: list[float], k: int) -> list[Document]:
        # Mimic a real backend: returns up to k rows in distance order;
        # if k exceeds dataset size, returns the whole dataset.
        return full[:k]

    backend = MagicMock()
    backend.query_fts.return_value = []
    backend.query_vector.side_effect = fake_query_vector
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    store = KnowledgeStore(backend=backend, embedder=embedder)

    results = store.query("q", top_k=top_k, tags=["python"])

    ids = {d.id for d in results}
    assert ids == {f"py_{i}" for i in range(5)}, (
        "Expected all on-tag python docs to surface even though they sit "
        "outside the initial top_k*3 vector window."
    )
    # Sanity: the store must have asked for more than the initial window.
    requested_top_ks = [c.args[1] for c in backend.query_vector.call_args_list]
    assert max(requested_top_ks) > initial_fetch, (
        f"Expected store to widen the vector fetch beyond {initial_fetch}; "
        f"actual requests: {requested_top_ks}"
    )


def test_query_does_not_widen_when_backend_exhausted() -> None:
    """Stop widening once the backend returns fewer rows than requested.

    If the vector backend returns fewer docs than ``k`` it is signalling
    the dataset is exhausted; the store must not loop forever re-asking.
    """
    on_tag = [_make_doc(f"py_{i}", tags=["python"]) for i in range(2)]

    def fake_query_vector(_embedding: list[float], _k: int) -> list[Document]:
        # Backend holds exactly 2 docs regardless of k.
        return list(on_tag)

    backend = MagicMock()
    backend.query_fts.return_value = []
    backend.query_vector.side_effect = fake_query_vector
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    store = KnowledgeStore(backend=backend, embedder=embedder)

    results = store.query("q", top_k=10, tags=["python"])

    assert {d.id for d in results} == {"py_0", "py_1"}
    # One vector call is enough; the second call (if any) sees the same
    # short return and must terminate the loop.
    assert backend.query_vector.call_count <= 2
