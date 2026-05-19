"""Regression tests for the `_Embedder` Protocol contract on `KnowledgeStore`.

The Protocol requires a public ``model_name`` attribute plus the prefix-aware
``embed_query`` / ``embed_passage`` / ``embed_passage_batch`` methods.
``KnowledgeStore.store`` must read ``model_name`` directly — silent
``"unknown"`` fallbacks for embedders without the attribute are gone — and
route every document through ``embed_passage_batch``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crossmem.core.store import KnowledgeStore


class _FakeEmbedderWithModelName:
    """Minimal embedder that satisfies the Protocol."""

    model_name = "test-model-v1"

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passage(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passage_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        del batch_size
        return [[0.0] * 384 for _ in texts]


class _FakeEmbedderWithoutModelName:
    """Embedder missing the ``model_name`` attribute — must fail loud."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passage(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passage_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        del batch_size
        return [[0.0] * 384 for _ in texts]


def test_store_uses_embedder_model_name() -> None:
    """Every chunk's metadata records the embedder's public ``model_name``."""
    backend = MagicMock()
    store = KnowledgeStore(backend=backend, embedder=_FakeEmbedderWithModelName())
    ids = store.store(
        content="hello",
        source_url="https://example.com/a",
        title="A",
        source_type="web",
    )
    assert ids
    stored_docs: list = []
    for call in backend.upsert_many.call_args_list:
        stored_docs.extend(call.args[0])
    assert stored_docs
    for doc in stored_docs:
        assert doc.metadata.embedding_model == "test-model-v1"


def test_store_fails_loud_for_embedder_without_model_name() -> None:
    """A real class lacking ``model_name`` raises AttributeError — no fallback."""
    backend = MagicMock()
    store = KnowledgeStore(backend=backend, embedder=_FakeEmbedderWithoutModelName())
    with pytest.raises(AttributeError):
        store.store(
            content="hello",
            source_url="https://example.com/a",
            title="A",
            source_type="web",
        )
