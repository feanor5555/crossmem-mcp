"""Tests for the central source-adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.sources.base import SourceBase
from crossmem.sources.registry import (
    SourceRegistry,
    UnknownSourceError,
    default_registry,
)

if TYPE_CHECKING:
    from crossmem.core.models import Document


class _FakeSource(SourceBase):
    """Minimal in-memory SourceBase used to exercise the registry."""

    def __init__(self, source_name: str = "fake") -> None:
        self._name = source_name
        self.calls: list[str] = []

    def name(self) -> str:
        return self._name

    def can_handle(self, uri: str) -> bool:  # pragma: no cover - trivial
        return uri.startswith(f"{self._name}:")

    def fetch(self, query: str, **_kwargs: object) -> list[Document]:
        self.calls.append(query)
        return []


def test_register_and_get_round_trip() -> None:
    registry = SourceRegistry()
    src = _FakeSource("alpha")

    registry.register(src)

    assert registry.get("alpha") is src


def test_get_unknown_source_raises_with_available_names() -> None:
    registry = SourceRegistry()
    registry.register(_FakeSource("alpha"))
    registry.register(_FakeSource("beta"))

    with pytest.raises(UnknownSourceError) as exc:
        registry.get("missing")

    message = str(exc.value)
    assert "missing" in message
    # Known names appear in the error message so callers / LLMs can recover.
    assert "alpha" in message
    assert "beta" in message


def test_register_rejects_duplicate_name() -> None:
    registry = SourceRegistry()
    registry.register(_FakeSource("alpha"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_FakeSource("alpha"))


def test_available_returns_sorted_names() -> None:
    registry = SourceRegistry()
    registry.register(_FakeSource("zeta"))
    registry.register(_FakeSource("alpha"))
    registry.register(_FakeSource("mike"))

    assert registry.available() == ["alpha", "mike", "zeta"]


def test_default_registry_lists_builtin_sources() -> None:
    registry = default_registry()

    names = registry.available()
    assert "context7" in names
    assert "web" in names
    assert "github" in names


def test_default_registry_returns_fresh_instances() -> None:
    """Each call returns an independent registry so test mutations don't leak."""
    a = default_registry()
    b = default_registry()

    assert a is not b
    # Adapters themselves are also separate so monkeypatching one doesn't
    # bleed into the next test.
    assert a.get("context7") is not b.get("context7")
