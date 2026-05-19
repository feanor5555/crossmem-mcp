"""Central registry mapping source names to :class:`SourceBase` adapters.

The MCP ``query`` tool consults this registry when a caller supplies a
``source=`` argument. A cache miss then triggers
``registry.get(source).fetch(query)`` which feeds results back into the
knowledge store so the next query hits the local cache.

The built-in adapters (``context7``, ``web``, ``github``) are wired in
:func:`default_registry`. Tests use :class:`SourceRegistry` directly to
register fake adapters without touching the production wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from crossmem.sources.base import SourceBase


class UnknownSourceError(ValueError):
    """Raised when a caller requests an adapter that is not registered.

    Subclasses :class:`ValueError` so existing MCP error-translation paths
    (`server._value_error_payload`) serialize it as a structured payload
    instead of a stacktrace.
    """


class SourceRegistry:
    """In-memory mapping of source name -> :class:`SourceBase` instance."""

    def __init__(self) -> None:
        self._adapters: dict[str, SourceBase] = {}

    def register(self, adapter: SourceBase) -> None:
        """Register ``adapter`` under ``adapter.name()``.

        Raises ``ValueError`` if the name is already registered to keep
        accidental double-registration out of production wiring.
        """
        name = adapter.name()
        if name in self._adapters:
            raise ValueError(f"source {name!r} is already registered")
        self._adapters[name] = adapter

    def get(self, name: str) -> SourceBase:
        """Return the adapter registered under ``name``.

        Raises :class:`UnknownSourceError` listing every registered name so
        LLM callers can recover by retrying with a valid source.
        """
        try:
            return self._adapters[name]
        except KeyError as exc:
            known = ", ".join(self.available()) or "<none>"
            raise UnknownSourceError(
                f"unknown source {name!r}; known sources: {known}"
            ) from exc

    def available(self) -> list[str]:
        """Return every registered source name, alphabetically sorted."""
        return sorted(self._adapters)


def default_registry() -> SourceRegistry:
    """Build a fresh :class:`SourceRegistry` with all built-in adapters.

    Each call returns an independent registry (and independent adapter
    instances) so tests can mutate the wiring without leaking state into
    sibling tests.
    """
    from crossmem.sources.adapters.context7 import Context7Adapter
    from crossmem.sources.github import GitHubSource
    from crossmem.sources.web import WebSource

    registry = SourceRegistry()
    for adapter in _builtin_adapters(Context7Adapter, GitHubSource, WebSource):
        registry.register(adapter)
    return registry


def _builtin_adapters(*factories: type) -> Iterable[SourceBase]:
    """Yield one fresh instance per adapter factory."""
    for factory in factories:
        yield factory()


__all__ = [
    "SourceRegistry",
    "UnknownSourceError",
    "default_registry",
]
