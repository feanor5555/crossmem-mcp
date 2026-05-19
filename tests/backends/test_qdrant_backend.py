"""Tests for QdrantBackend (skipped when qdrant-client is not installed)."""

from __future__ import annotations

import pytest

try:
    import qdrant_client
except ImportError:  # pragma: no cover - exercised only when extra missing
    qdrant_client = None

pytestmark = pytest.mark.skipif(
    qdrant_client is None, reason="qdrant-client not installed"
)

from crossmem.backends.qdrant_backend import QdrantBackend  # noqa: E402
from crossmem.core.models import Document, Metadata, generate_content_hash  # noqa: E402


def _hexify(label: str) -> str:
    """Return a deterministic 32-hex doc_id derived from ``label``.

    ``QdrantBackend._point_id`` parses ``doc.id[:16]`` as hex; non-hex
    inputs (``"test123"``, ``"meta-only"``, ``"z1"`` etc.) trip that
    check. Hex-valid labels (``"d1"``, ``"a1"``) pass through unchanged
    so existing readable test assertions keep working; everything else
    is hashed to a 32-hex string. Both branches return the same width
    so tests that compare ids by equality remain stable.
    """
    try:
        int(label[:16], 16)
    except ValueError:
        import hashlib

        return hashlib.sha256(label.encode("utf-8")).hexdigest()[:32]
    return label


def _make_doc(
    doc_id: str = "test123",
    content: str = "Test content",
    source_url: str = "https://example.com",
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> Document:
    """Create a Document for testing.

    Non-hex ``doc_id`` arguments are transparently mapped to a stable
    32-hex string via :func:`_hexify` so the backend's
    ``_point_id(doc.id[:16], 16)`` call always succeeds. Hex-valid
    labels (e.g. ``"d1"``) survive unchanged for readable assertions.
    """
    if tags is None:
        tags = ["python", "test"]
    doc_id = _hexify(doc_id)
    return Document(
        id=doc_id,
        content=content,
        embedding=embedding if embedding is not None else [0.1] * 384,
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
def backend() -> QdrantBackend:
    """In-memory QdrantBackend (no on-disk persistence, no server)."""
    return QdrantBackend(collection_name="test_crossmem")


class TestQdrantBackendStore:
    def test_store_and_query_vector(self, backend: QdrantBackend) -> None:
        doc = _make_doc()
        backend.store(doc)

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id
        assert results[0].content == "Test content"
        assert results[0].metadata.source_url == "https://example.com"
        assert results[0].metadata.tags == ("python", "test")

    def test_store_upsert(self, backend: QdrantBackend) -> None:
        doc1 = _make_doc(content="Version 1")
        backend.store(doc1)

        doc2 = _make_doc(content="Version 2")
        backend.store(doc2)

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].content == "Version 2"


class TestQdrantBackendDelete:
    def test_delete_removes_doc(self, backend: QdrantBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Alpha")
        doc2 = _make_doc(doc_id="d2", content="Beta")
        backend.store(doc1)
        backend.store(doc2)

        backend.delete("d1")

        results = backend.query_vector([0.1] * 384, top_k=5)
        ids = {r.id for r in results}
        assert "d1" not in ids
        assert "d2" in ids

    def test_delete_nonexistent_is_noop(self, backend: QdrantBackend) -> None:
        # deleting a never-stored id must not raise. ``_point_id`` parses
        # doc.id[:16] as hex, so we use a valid 32-hex id that was never
        # stored.
        backend.delete("beef0000000000000000000000000000")


class TestQdrantBackendGetByUrl:
    def test_get_by_url_finds_docs(self, backend: QdrantBackend) -> None:
        url = "https://example.com/page"
        doc1 = _make_doc(doc_id="d1", source_url=url, content="First")
        doc2 = _make_doc(doc_id="d2", source_url=url, content="Second")
        doc3 = _make_doc(doc_id="d3", source_url="https://other.example/", content="X")
        backend.store(doc1)
        backend.store(doc2)
        backend.store(doc3)

        results = backend.get_by_url(url)
        assert {r.id for r in results} == {"d1", "d2"}

    def test_get_by_url_empty(self, backend: QdrantBackend) -> None:
        assert backend.get_by_url("https://nope.example.com/") == []

    def test_get_by_url_returns_deterministic_order(
        self, backend: QdrantBackend
    ) -> None:
        """Order must be stable (sorted by id) regardless of store sequence."""
        url = "https://example.com/shared"
        backend.store(_make_doc(doc_id="z1", source_url=url, content="A"))
        backend.store(_make_doc(doc_id="a1", source_url=url, content="B"))
        backend.store(_make_doc(doc_id="m1", source_url=url, content="C"))

        ids = [d.id for d in backend.get_by_url(url)]
        assert ids == sorted(ids)


class TestQdrantBackendGetById:
    def test_get_by_id_finds_doc(self, backend: QdrantBackend) -> None:
        doc = _make_doc(doc_id="d1", content="Alpha")
        backend.store(doc)

        result = backend.get_by_id("d1")

        assert result is not None
        assert result.id == "d1"
        assert result.content == "Alpha"
        assert len(result.embedding) == 384

    def test_get_by_id_returns_none_for_missing(self, backend: QdrantBackend) -> None:
        # _point_id parses doc.id[:16] as hex — use a hex-valid id that
        # was never stored so the lookup reaches Qdrant and returns None.
        assert backend.get_by_id("dead0000000000000000000000000000") is None


class TestQdrantBackendQueryFts:
    def test_query_fts_substring_match(self, backend: QdrantBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Python asyncio tutorial")
        doc2 = _make_doc(doc_id="d2", content="Rust ownership guide")
        backend.store(doc1)
        backend.store(doc2)

        results = backend.query_fts("asyncio", top_k=10)
        ids = {r.id for r in results}
        assert "d1" in ids
        assert "d2" not in ids

    def test_query_fts_with_tag_filter(self, backend: QdrantBackend) -> None:
        doc1 = _make_doc(doc_id="d1", content="Python guide", tags=["python"])
        doc2 = _make_doc(doc_id="d2", content="Python ref", tags=["javascript"])
        backend.store(doc1)
        backend.store(doc2)

        results = backend.query_fts("Python", top_k=10, tags=["python"])
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_query_fts_empty_db(self, backend: QdrantBackend) -> None:
        assert backend.query_fts("anything", top_k=5) == []


class TestQdrantBackendStats:
    def test_stats_empty_db(self, backend: QdrantBackend) -> None:
        stats = backend.stats()
        assert stats["backend"] == "qdrant"
        assert stats["document_count"] == 0
        assert stats["top_tags"] == []
        # TODO 26.9 (variant a): db_size_bytes is part of the unified
        # contract. The in-memory mode (and clients that omit
        # ``disk_data_size``) report 0 — but the key MUST be present
        # as a non-negative integer.
        assert isinstance(stats["db_size_bytes"], int)
        assert stats["db_size_bytes"] >= 0

    def test_stats_counts_documents(self, backend: QdrantBackend) -> None:
        for i in range(3):
            backend.store(_make_doc(doc_id=f"doc{i}", content=f"Content {i}"))

        stats = backend.stats()
        assert stats["document_count"] == 3
        assert stats["backend"] == "qdrant"
        # Each doc carries ["python", "test"] tags -> both rank top.
        tags = {t for t, _ in stats["top_tags"]}
        assert {"python", "test"}.issubset(tags)

    def test_stats_db_size_reports_disk_data_size(self) -> None:
        """When the server exposes ``disk_data_size``, it surfaces in stats.

        Variant (a) of TODO 26.9 mandates a unified ``db_size_bytes``
        field. For Qdrant, the value comes from ``info.disk_data_size``
        when present; in-memory and older clients fall back to 0. A
        stubbed ``get_collection`` lets us exercise the present-path
        without standing up a real Qdrant server.
        """
        from unittest.mock import MagicMock

        from qdrant_client import QdrantClient

        client = QdrantClient(location=":memory:")
        b = QdrantBackend(client=client, collection_name="size_probe")

        info = MagicMock()
        info.points_count = 0
        info.disk_data_size = 12345
        client.get_collection = MagicMock(return_value=info)  # type: ignore[method-assign]

        stats = b.stats()
        assert stats["db_size_bytes"] == 12345


class TestQdrantBackendRoundtrip:
    def test_full_roundtrip(self, backend: QdrantBackend) -> None:
        doc = _make_doc(content="Roundtrip content")
        backend.store(doc)

        results = backend.query_vector([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].id == doc.id

        assert len(backend.get_by_url("https://example.com")) == 1

        backend.delete(doc.id)

        assert backend.query_vector([0.1] * 384, top_k=5) == []
        assert backend.get_by_url("https://example.com") == []


class TestQdrantBackendFindByTag:
    """Tests for the find_by_tag() native-index path (TODO 26.1)."""

    def test_find_by_tag_returns_docs_with_exact_tag(
        self, backend: QdrantBackend
    ) -> None:
        backend.store(_make_doc(doc_id="d1", tags=["python", "asyncio"]))
        backend.store(_make_doc(doc_id="d2", tags=["python"]))
        backend.store(_make_doc(doc_id="d3", tags=["rust"]))

        ids = {d.id for d in backend.find_by_tag("python")}
        assert ids == {"d1", "d2"}

    def test_find_by_tag_returns_iterator(self, backend: QdrantBackend) -> None:
        from collections.abc import Iterator

        backend.store(_make_doc(doc_id="d1", tags=["python"]))
        assert isinstance(backend.find_by_tag("python"), Iterator)

    def test_find_by_tag_finds_docs_whose_content_lacks_the_tag(
        self, backend: QdrantBackend
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
        assert [d.id for d in hits] == [_hexify("meta-only")]

    def test_find_by_tag_no_match_returns_empty(self, backend: QdrantBackend) -> None:
        backend.store(_make_doc(doc_id="d1", tags=["python"]))
        assert list(backend.find_by_tag("rust")) == []

    def test_find_by_tag_paginates(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossmem.backends import qdrant_backend as qb

        monkeypatch.setattr(qb, "_ITER_PAGE_SIZE", 4)
        n = 10
        for i in range(n):
            backend.store(_make_doc(doc_id=f"p{i}", tags=["bulk"]))

        ids = {d.id for d in backend.find_by_tag("bulk")}
        assert ids == {_hexify(f"p{i}") for i in range(n)}


class TestQdrantBackendIterAll:
    def test_iter_all_returns_all_stored_docs(self, backend: QdrantBackend) -> None:
        ids = {_hexify(f"doc{i}") for i in range(5)}
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
        backend: QdrantBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from crossmem.backends import qdrant_backend as qb

        monkeypatch.setattr(qb, "_ITER_PAGE_SIZE", 4)

        n = 10
        for i in range(n):
            backend.store(_make_doc(doc_id=f"p{i}", content=f"Page {i}"))

        items = list(backend.iter_all())
        ids = [d.id for d in items]
        assert len(ids) == n
        assert len(set(ids)) == n
        assert set(ids) == {_hexify(f"p{i}") for i in range(n)}

    def test_iter_all_empty(self, backend: QdrantBackend) -> None:
        assert list(backend.iter_all()) == []


class TestQdrantBackendInit:
    def test_default_in_memory_when_no_args(self) -> None:
        b = QdrantBackend(collection_name="test_default")
        assert b.stats()["document_count"] == 0

    def test_accepts_external_client(self) -> None:
        from qdrant_client import QdrantClient

        client = QdrantClient(location=":memory:")
        b = QdrantBackend(client=client, collection_name="test_external")
        b.store(_make_doc())
        assert b.stats()["document_count"] == 1


def _hex_doc_id(seed: int) -> str:
    """Deterministic 32-hex doc_id derived from ``seed``.

    The ``QdrantBackend._point_id`` helper hashes ``doc.id[:16]`` to a
    uint64 and rejects non-hex input. The legacy ``_make_doc`` helper
    above uses non-hex strings like ``"test123"`` which trip that check
    in the live in-memory client; tests added for TODO 26.3 must use
    valid hex ids so the new code paths can be exercised end-to-end.

    The seed is encoded into the **first 16 hex chars** because
    ``_point_id`` only consumes that prefix; encoding the seed lower in
    the string collapses every id to point-id zero and overwrites docs.
    """
    if seed >= 1 << 64:
        raise ValueError("seed must fit in uint64")
    return f"{seed:016x}{0:016x}"


def _make_hex_doc(
    seed: int,
    content: str = "Test content",
    source_url: str = "https://example.com",
    tags: list[str] | None = None,
) -> Document:
    return _make_doc(
        doc_id=_hex_doc_id(seed),
        content=content,
        source_url=source_url,
        tags=tags,
    )


class TestQdrantBackendQueryFtsServerSide:
    """Server-side filter path for ``query_fts`` (TODO 26.3).

    The backend now registers a TEXT payload index on ``content`` and
    routes ``query_fts`` through a single ``scroll`` call with a
    ``MatchText`` filter instead of paging the entire collection.
    A feature-probe ``try/except`` keeps the historical client-side
    scan available for clients that lack ``MatchText`` or servers that
    reject the filter shape.
    """

    def test_uses_match_text_on_fast_path(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend.store(_make_hex_doc(1, content="python asyncio tutorial"))
        backend.store(_make_hex_doc(2, content="rust ownership guide"))

        captured: list[object] = []
        real_scroll = backend._client.scroll

        def spy_scroll(*args: object, **kwargs: object) -> object:
            captured.append(kwargs.get("scroll_filter"))
            return real_scroll(*args, **kwargs)

        monkeypatch.setattr(backend._client, "scroll", spy_scroll)

        results = backend.query_fts("asyncio", top_k=10)

        assert {r.id for r in results} == {_hex_doc_id(1)}
        assert captured, "scroll was never called"
        # The first filter argument carries the ``MatchText`` clause.
        flt = captured[0]
        assert flt is not None
        text_clause = flt.must[0]
        assert text_clause.key == "content"
        assert hasattr(text_clause.match, "text")
        assert text_clause.match.text == "asyncio"

    def test_fast_path_uses_single_scroll_call(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One scroll, not N. The server-side filter replaces the page walk."""
        from crossmem.backends import qdrant_backend as qb

        monkeypatch.setattr(qb, "_ITER_PAGE_SIZE", 2)
        for i in range(6):
            backend.store(_make_hex_doc(i, content=f"python item {i}"))

        scroll_calls = 0
        real_scroll = backend._client.scroll

        def spy(*args: object, **kwargs: object) -> object:
            nonlocal scroll_calls
            scroll_calls += 1
            return real_scroll(*args, **kwargs)

        monkeypatch.setattr(backend._client, "scroll", spy)

        results = backend.query_fts("python", top_k=10)
        assert len(results) == 6
        assert scroll_calls == 1

    def test_falls_back_to_client_scan_when_match_text_missing(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client whose ``models`` module does not expose ``MatchText``
        (e.g. older qdrant-client builds) must transparently fall back to
        the historical paged substring scan."""
        backend.store(_make_hex_doc(1, content="python asyncio tutorial"))
        backend.store(_make_hex_doc(2, content="rust ownership guide"))

        # Simulate an older client by removing ``MatchText`` from the
        # cached models module the backend captured at __init__ time.
        monkeypatch.delattr(backend._qmodels, "MatchText", raising=False)

        results = backend.query_fts("asyncio", top_k=10)
        assert {r.id for r in results} == {_hex_doc_id(1)}

    def test_empty_query_short_circuits_without_calling_scroll(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``query_fts("")`` returns ``[]`` immediately so the server
        never sees an empty ``MatchText`` filter."""
        backend.store(_make_hex_doc(1, content="anything"))

        called = False

        def boom(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            raise AssertionError("scroll must not be called for empty text")

        monkeypatch.setattr(backend._client, "scroll", boom)
        assert backend.query_fts("", top_k=10) == []
        assert called is False

    def test_falls_back_to_client_scan_when_server_rejects_filter(
        self, backend: QdrantBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server that rejects the ``MatchText`` filter shape must
        still produce correct results via the fallback path."""
        backend.store(_make_hex_doc(1, content="python asyncio tutorial"))
        backend.store(_make_hex_doc(2, content="rust ownership guide"))

        real_scroll = backend._client.scroll

        def scroll_rejecting_match_text(*args: object, **kwargs: object) -> object:
            flt = kwargs.get("scroll_filter")
            if (
                flt is not None
                and getattr(flt, "must", None)
                and any(hasattr(getattr(c, "match", None), "text") for c in flt.must)
            ):
                raise RuntimeError("simulated: server rejects MatchText")
            return real_scroll(*args, **kwargs)

        monkeypatch.setattr(backend._client, "scroll", scroll_rejecting_match_text)

        results = backend.query_fts("asyncio", top_k=10)
        assert {r.id for r in results} == {_hex_doc_id(1)}


@pytest.mark.benchmark
class TestQdrantBackendQueryFtsBenchmark:
    """Benchmark: server-side ``MatchText`` halves page fetches at scale.

    TODO 26.3 DoD: with ≥1k docs the number of ``scroll`` calls issued by
    ``query_fts`` is at least halved versus the pre-26.3 paged substring
    scan. Marked ``pytest.mark.benchmark`` so the default CI suite skips
    it; nightly / explicit runs trip it.
    """

    def test_server_side_filter_at_least_halves_page_fetches(
        self,
        backend: QdrantBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Sparse rare-token pattern forces the scan to walk every page;
        # otherwise the first 256-doc page fills ``top_k`` and the
        # comparison is meaningless.
        n_docs = 1024
        rare_token = "xyzrarefts26token"
        for i in range(n_docs):
            payload = f"row {i} {rare_token}" if (i % 128) == 0 else f"row {i} filler"
            backend.store(_make_hex_doc(i, content=payload))

        real_scroll = backend._client.scroll

        baseline_calls = 0

        def count_baseline(*args: object, **kwargs: object) -> object:
            nonlocal baseline_calls
            baseline_calls += 1
            return real_scroll(*args, **kwargs)

        monkeypatch.setattr(backend._client, "scroll", count_baseline)
        baseline = backend._query_fts_scroll_fallback(rare_token, 10, None)
        assert len(baseline) == 8

        fast_calls = 0

        def count_fast(*args: object, **kwargs: object) -> object:
            nonlocal fast_calls
            fast_calls += 1
            return real_scroll(*args, **kwargs)

        monkeypatch.setattr(backend._client, "scroll", count_fast)
        fast = backend.query_fts(rare_token, top_k=10)
        assert len(fast) == 8

        assert baseline_calls >= 2
        assert fast_calls * 2 <= baseline_calls
