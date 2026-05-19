"""Cross-backend contract test for the BackendStats schema.

Every backend's :meth:`stats` MUST return a payload that matches the
:class:`BackendStats` TypedDict shape. This test runs the same assertions
against every available backend to guarantee a unified key set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.backends.base import (
    REQUIRED_STATS_KEYS,
    BackendStats,
    VectorStoreBase,
)
from crossmem.backends.sqlite_backend import SQLiteBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crossmem.core.models import Document

try:
    import chromadb
except ImportError:  # pragma: no cover - chromadb is optional
    chromadb = None

try:
    import qdrant_client
except ImportError:  # pragma: no cover - qdrant-client is optional
    qdrant_client = None


def _make_sqlite() -> SQLiteBackend:
    return SQLiteBackend(":memory:")


def _make_chroma() -> object:
    from crossmem.backends.chroma_backend import ChromaBackend

    client = chromadb.EphemeralClient()
    return ChromaBackend(client=client, collection_name="contract_test")


def _make_qdrant() -> object:
    from crossmem.backends.qdrant_backend import QdrantBackend

    return QdrantBackend(collection_name="contract_test")


_BACKEND_FACTORIES: list[pytest.param] = [
    pytest.param(_make_sqlite, id="sqlite"),
    pytest.param(
        _make_chroma,
        id="chroma",
        marks=pytest.mark.skipif(chromadb is None, reason="chromadb not installed"),
    ),
    pytest.param(
        _make_qdrant,
        id="qdrant",
        marks=pytest.mark.skipif(
            qdrant_client is None, reason="qdrant-client not installed"
        ),
    ),
]


@pytest.mark.parametrize("backend_factory", _BACKEND_FACTORIES)
def test_stats_returns_backendstats_shape(backend_factory) -> None:
    """Every backend exposes the unified BackendStats key set + types."""
    backend = backend_factory()
    try:
        stats = backend.stats()

        # Required keys present.
        assert set(stats) >= REQUIRED_STATS_KEYS

        # Required key types.
        assert isinstance(stats["document_count"], int)
        assert isinstance(stats["top_tags"], list)
        assert isinstance(stats["backend"], str)
        # ``db_size_bytes`` is part of the unified contract (TODO 26.9,
        # variant a): every backend must expose a non-negative integer.
        assert isinstance(stats["db_size_bytes"], int)
        assert stats["db_size_bytes"] >= 0

        # No legacy alias leaks through.
        assert "doc_count" not in stats
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def test_required_stats_keys_constant() -> None:
    """``REQUIRED_STATS_KEYS`` exposes the mandatory stats key set."""
    assert (
        frozenset({"document_count", "top_tags", "backend", "db_size_bytes"})
        == REQUIRED_STATS_KEYS
    )


class _BrokenBackend(VectorStoreBase):
    """Backend that violates the stats contract (omits ``backend`` key)."""

    def store(self, doc: Document) -> None:  # pragma: no cover - unused
        pass

    def query_vector(
        self, embedding: list[float], top_k: int
    ) -> list[Document]:  # pragma: no cover - unused
        return []

    def query_fts(
        self, text: str, top_k: int, tags: list[str] | None = None
    ) -> list[Document]:  # pragma: no cover - unused
        return []

    def delete(self, doc_id: str) -> None:  # pragma: no cover - unused
        pass

    def get_by_url(
        self, source_url: str
    ) -> list[Document]:  # pragma: no cover - unused
        return []

    def get_by_id(self, doc_id: str) -> Document | None:  # pragma: no cover - unused
        return None

    def stats(self) -> BackendStats:
        # Intentionally drops the ``backend`` required key.
        return {"document_count": 0, "top_tags": []}  # type: ignore[typeddict-item]

    def iter_all(self) -> Iterator[Document]:  # pragma: no cover - unused
        return iter(())

    def find_by_tag(self, tag: str) -> Iterator[Document]:  # pragma: no cover - unused
        return iter(())


def test_missing_required_key_raises_at_runtime() -> None:
    """A backend whose ``stats()`` omits a required key fails fast."""
    backend = _BrokenBackend()
    with pytest.raises(TypeError, match="missing required stats keys"):
        backend.stats()
