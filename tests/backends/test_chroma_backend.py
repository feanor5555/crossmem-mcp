"""Tests for ChromaBackend (skipped when chromadb is not installed)."""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import chromadb
except ImportError:
    chromadb = None

pytestmark = pytest.mark.skipif(chromadb is None, reason="chromadb not installed")

# The import below relies on chromadb being available at import-time of the
# backend module (it imports chromadb lazily inside __init__).  Importing the
# module itself should be safe even without chromadb.
from crossmem.backends.chroma_backend import ChromaBackend  # noqa: E402
from crossmem.core.models import Document, Metadata, generate_content_hash  # noqa: E402


def _make_doc(
    doc_id: str = "test123",
    content: str = "Test content",
    source_url: str = "https://example.com",
    tags: list[str] | None = None,
) -> Document:
    """Create a Document for testing."""
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


@pytest.fixture
def backend() -> Iterator[ChromaBackend]:
    """Build an isolated ChromaBackend per test.

    chromadb 1.5.9's ``EphemeralClient`` shares the default tenant/database
    across instances within the same process, so a collection named
    ``test_crossmem`` created in one test leaks into the next. Use a unique
    collection name per test and call ``delete_collection`` in teardown so
    tests cannot observe each other's state.
    """
    client = chromadb.EphemeralClient()
    collection_name = f"test_crossmem_{uuid.uuid4().hex}"
    backend_obj = ChromaBackend(client=client, collection_name=collection_name)
    try:
        yield backend_obj
    finally:
        # Best-effort cleanup; failures here must not mask test results.
        with contextlib.suppress(Exception):
            client.delete_collection(name=collection_name)


class TestChromaBackendStore:
    def test_store_and_query_vector(self, backend: ChromaBackend) -> None:
        doc = _make_doc()
        backend.store(doc)

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id
        assert results[0].content == "Test content"
        assert results[0].metadata.source_url == "https://example.com"
        assert results[0].metadata.tags == ("python", "test")

    def test_store_upsert(self, backend: ChromaBackend) -> None:
        doc1 = _make_doc(content="Version 1")
        backend.store(doc1)

        doc2 = _make_doc(content="Version 2")
        backend.store(doc2)

        # Same id -> single doc, latest content wins
        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].content == "Version 2"


class TestChromaBackendDelete:
    def test_delete_removes_doc(self, backend: ChromaBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Alpha")
        doc2 = _make_doc(doc_id="d2", content="Beta")
        backend.store(doc1)
        backend.store(doc2)

        backend.delete("d1")

        results = backend.query_vector([0.1] * 384, top_k=5)
        ids = {r.id for r in results}
        assert "d1" not in ids
        assert "d2" in ids

    def test_delete_nonexistent_is_noop(self, backend: ChromaBackend) -> None:
        backend.delete("does-not-exist")  # Should not raise


class TestChromaBackendGetByUrl:
    def test_get_by_url_finds_docs(self, backend: ChromaBackend) -> None:
        url = "https://example.com/page"
        doc1 = _make_doc(doc_id="d1", source_url=url, content="First")
        doc2 = _make_doc(doc_id="d2", source_url=url, content="Second")
        doc3 = _make_doc(doc_id="d3", source_url="https://other.example/", content="X")
        backend.store(doc1)
        backend.store(doc2)
        backend.store(doc3)

        results = backend.get_by_url(url)
        assert {r.id for r in results} == {"d1", "d2"}

    def test_get_by_url_empty(self, backend: ChromaBackend) -> None:
        assert backend.get_by_url("https://nope.example.com/") == []

    def test_get_by_url_returns_deterministic_order(
        self, backend: ChromaBackend
    ) -> None:
        """Order must be stable (sorted by id) regardless of store sequence."""
        url = "https://example.com/shared"
        backend.store(_make_doc(doc_id="z1", source_url=url, content="A"))
        backend.store(_make_doc(doc_id="a1", source_url=url, content="B"))
        backend.store(_make_doc(doc_id="m1", source_url=url, content="C"))

        ids = [d.id for d in backend.get_by_url(url)]
        assert ids == sorted(ids)


class TestChromaBackendGetById:
    def test_get_by_id_finds_doc(self, backend: ChromaBackend) -> None:
        doc = _make_doc(doc_id="d1", content="Alpha")
        backend.store(doc)

        result = backend.get_by_id("d1")

        assert result is not None
        assert result.id == "d1"
        assert result.content == "Alpha"
        assert len(result.embedding) == 384

    def test_get_by_id_returns_none_for_missing(self, backend: ChromaBackend) -> None:
        assert backend.get_by_id("nope") is None


class TestChromaBackendQueryFts:
    def test_query_fts_substring_match(self, backend: ChromaBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Python asyncio tutorial")
        doc2 = _make_doc(doc_id="d2", content="Rust ownership guide")
        backend.store(doc1)
        backend.store(doc2)

        results = backend.query_fts("asyncio", top_k=10)
        ids = {r.id for r in results}
        assert "d1" in ids
        assert "d2" not in ids

    def test_query_fts_with_tag_filter(self, backend: ChromaBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Python guide", tags=["python"])
        doc2 = _make_doc(doc_id="d2", content="Python ref", tags=["javascript"])
        backend.store(doc1)
        backend.store(doc2)

        results = backend.query_fts("Python", top_k=10, tags=["python"])
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_query_fts_empty_db(self, backend: ChromaBackend) -> None:
        assert backend.query_fts("anything", top_k=5) == []

    def test_query_fts_is_case_insensitive(self, backend: ChromaBackend) -> None:
        """Querying with mixed case must match docs stored with any case.

        SQLite's FTS5 trigram tokenizer is case-insensitive; the Chroma
        backend must behave the same way so that the same `query("Python")`
        finds a doc stored as "python ..." regardless of backend choice.
        """
        # Use a marker token absent from other tests to avoid cross-test
        # contamination from chromadb's shared ephemeral default tenant.
        backend.store(_make_doc(doc_id="ci_lower", content="zylophone asyncio note"))
        backend.store(_make_doc(doc_id="ci_mixed", content="Zylophone decorators"))
        backend.store(_make_doc(doc_id="ci_upper", content="ZYLOPHONE typing ref"))
        backend.store(_make_doc(doc_id="ci_other", content="rust ownership guide"))

        ids = {r.id for r in backend.query_fts("Zylophone", top_k=10)}
        assert ids == {"ci_lower", "ci_mixed", "ci_upper"}

        ids_lower = {r.id for r in backend.query_fts("zylophone", top_k=10)}
        assert ids_lower == {"ci_lower", "ci_mixed", "ci_upper"}


class TestChromaBackendStats:
    def test_stats_empty_db(self, backend: ChromaBackend) -> None:
        stats = backend.stats()
        assert stats["backend"] == "chroma"
        assert stats["document_count"] == 0
        assert stats["top_tags"] == []
        # TODO 26.9 (variant a): db_size_bytes is part of the unified
        # contract. Ephemeral clients have no persist directory so the
        # value is 0, but the key MUST always be present.
        assert stats["db_size_bytes"] == 0

    def test_stats_counts_documents(self, backend: ChromaBackend) -> None:
        for i in range(3):
            backend.store(_make_doc(doc_id=f"doc{i}", content=f"Content {i}"))

        stats = backend.stats()
        assert stats["document_count"] == 3
        assert stats["backend"] == "chroma"

    def test_stats_db_size_persistent_dir(self, tmp_path: object) -> None:
        """A persistent Chroma backend reports the on-disk footprint.

        Chroma's persistent client materialises a SQLite file plus index
        directories under ``path``; the unified ``db_size_bytes`` is the
        sum of every file's size, so a populated collection MUST report
        a strictly positive value.
        """
        path = tmp_path / "chroma-persist"  # type: ignore[operator]
        path.mkdir()
        b = ChromaBackend(path=path, collection_name="size_probe")
        b.store(_make_doc(doc_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

        stats = b.stats()
        assert stats["db_size_bytes"] > 0
        assert stats["backend"] == "chroma"


class TestChromaBackendRoundtrip:
    def test_full_roundtrip(self, backend: ChromaBackend) -> None:
        doc = _make_doc(content="Roundtrip content")
        backend.store(doc)

        # Found via vector search
        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id

        # Found via URL
        assert len(backend.get_by_url("https://example.com")) == 1

        # Delete
        backend.delete(doc.id)

        # Gone everywhere
        assert backend.query_vector([0.1] * 384, top_k=5) == []
        assert backend.get_by_url("https://example.com") == []


class TestChromaBackendInit:
    def test_default_ephemeral_when_no_args(self) -> None:
        b = ChromaBackend(collection_name="test_default")
        assert b.collection.count() == 0

    def test_persistent_path(self, tmp_path: object) -> None:
        db_dir = tmp_path / "chroma_db"  # type: ignore[operator]
        b = ChromaBackend(path=db_dir, collection_name="test_persist")
        b.store(_make_doc())
        assert b.collection.count() == 1


class TestChromaBackendFindByTag:
    """Tests for the find_by_tag() native-path (TODO 26.1)."""

    def test_find_by_tag_returns_docs_with_exact_tag(
        self, backend: ChromaBackend
    ) -> None:
        backend.store(_make_doc(doc_id="d1", tags=["python", "asyncio"]))
        backend.store(_make_doc(doc_id="d2", tags=["python"]))
        backend.store(_make_doc(doc_id="d3", tags=["rust"]))

        ids = {d.id for d in backend.find_by_tag("python")}
        assert ids == {"d1", "d2"}

    def test_find_by_tag_returns_iterator(self, backend: ChromaBackend) -> None:
        from collections.abc import Iterator

        backend.store(_make_doc(doc_id="d1", tags=["python"]))
        assert isinstance(backend.find_by_tag("python"), Iterator)

    def test_find_by_tag_finds_docs_whose_content_lacks_the_tag(
        self, backend: ChromaBackend
    ) -> None:
        """Spec: tag-only-in-metadata docs MUST be returned (no content match)."""
        backend.store(
            _make_doc(
                doc_id="meta-only",
                content="Completely unrelated prose with no tag tokens.",
                tags=["zzzobscure"],
            )
        )

        hits = list(backend.find_by_tag("zzzobscure"))
        assert [d.id for d in hits] == ["meta-only"]

    def test_find_by_tag_does_not_match_csv_substring(
        self, backend: ChromaBackend
    ) -> None:
        """Tag ``py`` MUST NOT match a doc tagged only with ``python``.

        Chroma stores ``tags`` as a CSV string; a naive ``$contains`` filter
        would mis-match prefixes. The implementation must use a tokenized
        membership check.
        """
        backend.store(_make_doc(doc_id="d1", tags=["python"]))
        backend.store(_make_doc(doc_id="d2", tags=["py"]))

        ids = {d.id for d in backend.find_by_tag("py")}
        assert ids == {"d2"}

    def test_find_by_tag_no_match_returns_empty(self, backend: ChromaBackend) -> None:
        backend.store(_make_doc(doc_id="d1", tags=["python"]))
        assert list(backend.find_by_tag("rust")) == []

    def test_find_by_tag_paginates(
        self, backend: ChromaBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Result set must span multiple pages without truncation."""
        from crossmem.backends import chroma_backend as cb

        monkeypatch.setattr(cb, "_ITER_PAGE_SIZE", 4)
        n = 10
        for i in range(n):
            backend.store(_make_doc(doc_id=f"p{i}", tags=["bulk"]))

        ids = {d.id for d in backend.find_by_tag("bulk")}
        assert ids == {f"p{i}" for i in range(n)}

    def test_find_by_tag_drops_tags_removed_on_reupsert(
        self, backend: ChromaBackend
    ) -> None:
        """Re-storing a doc with a different tag set must evict old tag keys.

        Chroma's upsert merges metadata, so a naive re-store would leave
        stale ``tag:<old>: True`` keys behind and cause phantom hits on the
        native-index path. The backend must clear removed tags in the same
        upsert so ``find_by_tag`` reflects the latest tag set only.
        """
        backend.store(_make_doc(doc_id="d1", tags=["python", "asyncio"]))
        assert {d.id for d in backend.find_by_tag("python")} == {"d1"}
        assert {d.id for d in backend.find_by_tag("asyncio")} == {"d1"}

        # Re-store under the same id with a disjoint tag set.
        backend.store(_make_doc(doc_id="d1", tags=["rust"]))

        assert list(backend.find_by_tag("python")) == []
        assert list(backend.find_by_tag("asyncio")) == []
        assert {d.id for d in backend.find_by_tag("rust")} == {"d1"}


class TestChromaBackendIterAll:
    def test_iter_all_returns_all_stored_docs(self, backend: ChromaBackend) -> None:
        ids = {f"doc{i}" for i in range(5)}
        for i in range(5):
            backend.store(_make_doc(doc_id=f"doc{i}", content=f"Content {i}"))

        items = list(backend.iter_all())
        assert {d.id for d in items} == ids
        for d in items:
            assert isinstance(d, Document)
            assert d.content.startswith("Content ")
            assert d.metadata.source_url == "https://example.com"
            assert d.metadata.tags == ("python", "test")
            assert len(d.embedding) == 384

    def test_iter_all_pagination(
        self,
        backend: ChromaBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink page size so fewer docs trigger multi-page paths.
        from crossmem.backends import chroma_backend as cb

        monkeypatch.setattr(cb, "_ITER_PAGE_SIZE", 4)

        n = 10  # > 2x page size
        for i in range(n):
            backend.store(_make_doc(doc_id=f"p{i}", content=f"Page {i}"))

        items = list(backend.iter_all())
        ids = [d.id for d in items]
        assert len(ids) == n
        assert len(set(ids)) == n  # no duplicates
        assert set(ids) == {f"p{i}" for i in range(n)}

    def test_iter_all_empty(self, backend: ChromaBackend) -> None:
        assert list(backend.iter_all()) == []


class TestChromaBackendQueryFtsServerSide:
    """Server-side filter path for ``query_fts`` (TODO 26.3).

    The backend stores a lowercased ``documents`` column and routes
    ``query_fts`` through Chroma's ``where_document={"$contains": ...}``
    so substring filtering happens inside Chroma instead of the
    client-side page walk that preceded 26.3. The case-insensitive
    contract is preserved by lowercasing both sides at write/query.
    """

    def test_uses_where_document_contains_on_fast_path(
        self, backend: ChromaBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fast path must call ``collection.get`` with the
        ``where_document={"$contains": needle}`` clause and the lowercased
        needle so the case-insensitive contract holds without a client-side
        rescan."""
        backend.store(_make_doc(doc_id="d1", content="Python asyncio tutorial"))
        backend.store(_make_doc(doc_id="d2", content="Rust ownership guide"))

        calls: list[dict[str, object]] = []
        real_get = backend.collection.get

        def spy_get(*args: object, **kwargs: object) -> object:
            calls.append(kwargs)
            return real_get(*args, **kwargs)

        monkeypatch.setattr(backend.collection, "get", spy_get)

        results = backend.query_fts("ASYNCIO", top_k=10)

        assert {r.id for r in results} == {"d1"}
        fts_calls = [c for c in calls if "where_document" in c]
        assert fts_calls, "server-side fast path was not exercised"
        assert fts_calls[0]["where_document"] == {"$contains": "asyncio"}

    def test_writes_content_lower_metadata_key(self, backend: ChromaBackend) -> None:
        """Write-time denormalisation: each row carries a ``content_lower``
        metadata key that mirrors the lowercased ``documents`` field. The
        key is the contract surface for the server-side fast path.
        """
        backend.store(_make_doc(doc_id="d1", content="HelloCaseSensitive WORLD"))
        raw = backend.collection.get(ids=["d1"], include=["documents", "metadatas"])
        # Metadata mirrors the lowercased content and exposes the original
        # under ``content`` so read paths can reconstruct the doc.
        assert raw["metadatas"][0]["content_lower"] == ("hellocasesensitive world")
        assert raw["metadatas"][0]["content"] == "HelloCaseSensitive WORLD"
        # The ``documents`` column itself is lowercased so Chroma's
        # ``$contains`` can match without round-tripping casing.
        assert raw["documents"][0] == "hellocasesensitive world"

    def test_round_trips_original_case_via_metadata(
        self, backend: ChromaBackend
    ) -> None:
        """Read paths must surface the original-cased content stored at
        write time even though ``documents`` is lowercased on the wire."""
        backend.store(_make_doc(doc_id="d1", content="MixedCase Content"))

        v = backend.query_vector([0.1] * 384, top_k=5)
        assert v[0].content == "MixedCase Content"

        fts = backend.query_fts("mixedcase", top_k=5)
        assert fts[0].content == "MixedCase Content"

        single = backend.get_by_id("d1")
        assert single is not None
        assert single.content == "MixedCase Content"

    def test_fast_path_uses_single_call_not_full_scan(
        self, backend: ChromaBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server-side fast path issues exactly one ``collection.get``
        call per query, regardless of collection size. The pre-26.3 client
        scan paged in ``ceil(N / _ITER_PAGE_SIZE)`` calls — counting the
        calls keeps the spec's "≥halved page-fetches" promise testable
        without spinning up 1k docs in the default suite.
        """
        from crossmem.backends import chroma_backend as cb

        # Shrink the page size so a tiny collection would normally need
        # multiple fallback pages — the assertion proves we skipped them.
        monkeypatch.setattr(cb, "_ITER_PAGE_SIZE", 2)
        for i in range(6):
            backend.store(_make_doc(doc_id=f"p{i}", content=f"Python item {i}"))

        calls = 0
        real_get = backend.collection.get

        def spy_get(*args: object, **kwargs: object) -> object:
            nonlocal calls
            if "where_document" in kwargs:
                calls += 1
            return real_get(*args, **kwargs)

        monkeypatch.setattr(backend.collection, "get", spy_get)

        results = backend.query_fts("python", top_k=10)
        assert len(results) == 6
        assert calls == 1  # one server-side call replaces 3 fallback pages

    def test_falls_back_to_client_scan_when_filter_rejected(
        self, backend: ChromaBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client that rejects the ``where_document`` filter shape (older
        version, mocked client, future schema change) must transparently
        fall back to the client-side substring scan."""
        backend.store(_make_doc(doc_id="d1", content="Python asyncio tutorial"))
        backend.store(_make_doc(doc_id="d2", content="Rust ownership guide"))

        real_get = backend.collection.get

        def get_that_rejects_where_document(*args: object, **kwargs: object) -> object:
            if "where_document" in kwargs:
                raise ValueError("simulated: where_document not supported")
            return real_get(*args, **kwargs)

        monkeypatch.setattr(backend.collection, "get", get_that_rejects_where_document)

        results = backend.query_fts("asyncio", top_k=10)
        assert {r.id for r in results} == {"d1"}

    def test_empty_query_short_circuits_without_calling_chroma(
        self, backend: ChromaBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``query_fts("")`` returns ``[]`` immediately so the server
        never sees an empty ``$contains`` filter (which Chroma rejects)."""
        backend.store(_make_doc(doc_id="d1", content="anything"))

        called = False

        def boom(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            raise AssertionError("collection.get must not be called for empty text")

        monkeypatch.setattr(backend.collection, "get", boom)
        assert backend.query_fts("", top_k=10) == []
        assert called is False

    def test_or_tag_filter_combines_with_where_document(
        self, backend: ChromaBackend
    ) -> None:
        """Multi-tag pre-filter composes into a ``$or`` over metadata flags
        and is applied alongside ``where_document`` on the fast path."""
        backend.store(_make_doc(doc_id="d1", content="Python guide", tags=["python"]))
        backend.store(_make_doc(doc_id="d2", content="Python deep dive", tags=["go"]))
        backend.store(_make_doc(doc_id="d3", content="Python ref", tags=["javascript"]))

        results = backend.query_fts("Python", top_k=10, tags=["python", "go"])
        assert {r.id for r in results} == {"d1", "d2"}


@pytest.mark.benchmark
class TestChromaBackendQueryFtsBenchmark:
    """Benchmark: server-side ``where_document`` halves page fetches at scale.

    TODO 26.3 DoD: with ≥1k docs the number of ``collection.get`` page
    fetches issued by ``query_fts`` is at least halved versus the pre-26.3
    client-side scan. Marked ``pytest.mark.benchmark`` so the default CI
    suite skips it; nightly / explicit runs trip it.
    """

    def test_server_side_filter_at_least_halves_page_fetches(
        self,
        backend: ChromaBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Spread matches across the full collection so the client-side
        # scan is forced to page through every chunk. Without a sparse
        # needle the scan returns ``top_k`` on the first page and the
        # comparison is meaningless.
        n_docs = 1024
        rare_token = "xyzrarefts26token"
        for i in range(n_docs):
            # 1 match every 128 docs -> 8 matches total, last one near the
            # tail. Forces the scan to read every page to fill ``top_k``.
            payload = f"row {i} {rare_token}" if (i % 128) == 0 else f"row {i} filler"
            backend.store(_make_doc(doc_id=f"b{i:05d}", content=payload))

        # Baseline: count what the pre-26.3 client-side scan would have
        # cost. ``_query_fts_scan_fallback`` is the exact replay path, so
        # any improvement we measure is honest.
        baseline_calls = 0
        real_get = backend.collection.get

        def count_baseline(*args: object, **kwargs: object) -> object:
            nonlocal baseline_calls
            baseline_calls += 1
            return real_get(*args, **kwargs)

        monkeypatch.setattr(backend.collection, "get", count_baseline)
        baseline = backend._query_fts_scan_fallback(rare_token, 10, None)
        assert len(baseline) == 8

        # Fast path: count the server-side path on the same dataset.
        fast_calls = 0

        def count_fast(*args: object, **kwargs: object) -> object:
            nonlocal fast_calls
            fast_calls += 1
            return real_get(*args, **kwargs)

        monkeypatch.setattr(backend.collection, "get", count_fast)
        fast = backend.query_fts(rare_token, top_k=10)
        assert len(fast) == 8

        # The client-side scan needs ``ceil(n / _ITER_PAGE_SIZE)`` page
        # reads (4 at the default 256-doc page size); the server-side
        # fast path collapses that to one ``collection.get`` call. Spec
        # demands at least half — assert the stronger result we deliver.
        assert baseline_calls >= 2
        assert fast_calls * 2 <= baseline_calls
