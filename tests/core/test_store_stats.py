"""Tests for KnowledgeStore.stats — thin pass-through to backend.stats()."""

from __future__ import annotations

from unittest.mock import MagicMock

from crossmem.core.store import KnowledgeStore


def test_stats_delegates_to_backend() -> None:
    backend = MagicMock()
    expected = {
        "document_count": 5,
        "db_size_bytes": 4096,
        "top_tags": [("python", 3)],
        "backend": "sqlite",
    }
    backend.stats.return_value = expected
    embedder = MagicMock()
    store = KnowledgeStore(backend=backend, embedder=embedder)

    result = store.stats()

    backend.stats.assert_called_once_with()
    assert result == expected
