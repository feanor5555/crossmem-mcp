"""Tests for SourceBase ABC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.sources.base import SourceBase

if TYPE_CHECKING:
    from crossmem.core.models import Document


def test_cannot_instantiate():
    """SourceBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SourceBase()


def test_subclass_missing_method_cannot_instantiate():
    """A subclass that omits any abstract method cannot be instantiated."""

    class MissingFetch(SourceBase):
        def name(self) -> str:
            return "missing"

        def can_handle(self, uri: str) -> bool:
            return False

    with pytest.raises(TypeError):
        MissingFetch()

    class MissingName(SourceBase):
        def fetch(self, query: str, **kwargs) -> list[Document]:
            return []

        def can_handle(self, uri: str) -> bool:
            return False

    with pytest.raises(TypeError):
        MissingName()

    class MissingCanHandle(SourceBase):
        def name(self) -> str:
            return "missing"

        def fetch(self, query: str, **kwargs) -> list[Document]:
            return []

    with pytest.raises(TypeError):
        MissingCanHandle()


def test_concrete_subclass():
    """A subclass implementing all abstract methods can be instantiated."""

    class ConcreteSource(SourceBase):
        def name(self) -> str:
            return "concrete"

        def fetch(self, query: str, **kwargs) -> list[Document]:
            return []

        def can_handle(self, uri: str) -> bool:
            return uri.startswith("https://")

    source = ConcreteSource()
    assert isinstance(source, SourceBase)
    assert source.name() == "concrete"
    assert source.fetch("anything") == []
    assert source.can_handle("https://example.com") is True
    assert source.can_handle("ftp://example.com") is False
