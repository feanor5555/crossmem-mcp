"""Abstract base class for knowledge source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crossmem.core.models import Document


class SourceBase(ABC):
    """Abstract base class for knowledge source adapters."""

    @abstractmethod
    def name(self) -> str:
        """Return the adapter's unique source name (e.g. "web", "github")."""

    @abstractmethod
    def fetch(self, query: str, **kwargs) -> list[Document]:
        """Fetch documents matching the query from the underlying source."""

    @abstractmethod
    def can_handle(self, uri: str) -> bool:
        """Return True if this source can handle the given URI."""
