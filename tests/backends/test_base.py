"""Tests for VectorStoreBase ABC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.backends.base import BackendStats, VectorStoreBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crossmem.core.models import Document


def test_cannot_instantiate():
    """VectorStoreBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VectorStoreBase()


def test_concrete_subclass():
    """A subclass implementing all abstract methods can be instantiated."""

    class ConcreteStore(VectorStoreBase):
        def store(self, doc: Document) -> None:
            pass

        def query_vector(self, embedding: list[float], top_k: int) -> list[Document]:
            return []

        def query_fts(
            self, text: str, top_k: int, tags: list[str] | None = None
        ) -> list[Document]:
            return []

        def delete(self, doc_id: str) -> None:
            pass

        def get_by_url(self, source_url: str) -> list[Document]:
            return []

        def get_by_id(self, doc_id: str) -> Document | None:
            return None

        def stats(self) -> BackendStats:
            return {
                "document_count": 0,
                "top_tags": [],
                "backend": "concrete",
            }

        def iter_all(self) -> Iterator[Document]:
            return iter(())

        def find_by_tag(self, tag: str) -> Iterator[Document]:
            return iter(())

    store = ConcreteStore()
    assert isinstance(store, VectorStoreBase)
