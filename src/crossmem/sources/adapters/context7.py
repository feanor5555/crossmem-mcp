"""Context7 docs-search API adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crossmem.core.models import Document
from crossmem.sources._http import borrowed_client, safe_get
from crossmem.sources.base import SourceBase

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx

DEFAULT_TIMEOUT = 10.0
USER_AGENT = "crossmem/0.1 (+https://github.com/crossmem/crossmem)"


class Context7Adapter(SourceBase):
    """Adapter for the Context7 docs-search API."""

    DEFAULT_BASE_URL = "https://context7.com/api/v1"

    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = client

    def name(self) -> str:
        return "context7"

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("context7:") or "context7.com" in uri

    def fetch(
        self,
        query: str,
        *,
        library: str | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[Document]:
        params: dict[str, Any] = {"q": query, "top_k": top_k}
        if library is not None:
            params["library"] = library
        params.update(kwargs)

        with borrowed_client(
            self._client,
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = safe_get(client, f"{self._base_url}/search", params=params)
            response.raise_for_status()
            payload = response.json()

        results = payload.get("results", []) if isinstance(payload, dict) else []
        return [_build_document(item) for item in results]


def _build_document(item: dict[str, Any]) -> Document:
    """Construct a ``Document`` from a single Context7 result item."""
    return Document.from_payload(
        content=str(item.get("content", "") or ""),
        source_url=str(item.get("url", "") or ""),
        title=str(item.get("title", "") or ""),
        source_type="context7",
    )


__all__: Iterable[str] = ("Context7Adapter",)
