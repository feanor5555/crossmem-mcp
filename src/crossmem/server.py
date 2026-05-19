"""FastMCP server exposing CrossMem's knowledge store via stdio MCP.

The server registers nine tools (``query``, ``store``, ``delete``,
``status``, ``cleanup``, ``export``, ``import_data``, ``empty_trash``,
``restore_from_trash``) backed by a
:class:`crossmem.core.store.KnowledgeStore`. The store is supplied via
:func:`create_server` for testability; :func:`main` reads the active backend
from ``~/.crossmem/config.toml`` (written by ``crossmem configure``) and
runs the server over stdio.

The ``configure`` command itself is intentionally **CLI-only** and never
exposed as an MCP tool: it writes ``~/.crossmem/config.toml`` which may
contain API keys and remote URLs. Exposing it over MCP would let a tool
call redirect the active backend or leak credentials — see CLAUDE.md
("Sicherheit" / "configure-Security-Entscheidung") for the full rationale.

Destructive MCP paths are gated behind ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``
(see :data:`DESTRUCTIVE_MCP_ENV`). Without the opt-in:

* ``cleanup`` ignores ``dry_run=False`` and is forced to ``dry_run=True``
  (preview-only). The response surfaces ``forced_dry_run=True`` so callers
  can detect the downgrade.
* ``empty_trash`` is a no-op that returns ``{"removed": 0, "blocked": True,
  "hint": ...}`` without invoking the underlying purge routine.

Rationale: a prompt-injected or otherwise compromised LLM must not be able
to permanently delete documents or wipe the soft-delete trash through MCP.
The CLI (``crossmem trash empty``) is unaffected — humans retain full
control. See CLAUDE.md ("Sicherheit" / "destructive MCP gate") for the
deployment story.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

# NOTE: ``BackendStats`` is imported at runtime (not behind ``TYPE_CHECKING``)
# because FastMCP/Pydantic evaluates tool return annotations at registration
# time to derive the wire schema. A forward reference would raise ``NameError``
# inside :func:`typing.get_type_hints`.
from crossmem.backends.base import BackendStats  # noqa: TC001 — runtime use, see note
from crossmem.cleanup import cleanup as cleanup_op
from crossmem.cleanup import empty_trash as empty_trash_op
from crossmem.cleanup import restore_from_trash as restore_from_trash_op
from crossmem.configure import (
    BackendConfigError,
    _default_sqlite_path,
    build_backend,
    load_config,
)
from crossmem.core.embedding import EmbeddingService
from crossmem.core.store import KnowledgeStore
from crossmem.sources.registry import (
    SourceRegistry,
    UnknownSourceError,
    default_registry,
)

if TYPE_CHECKING:
    from crossmem.core.models import Document


#: Environment variable that unlocks destructive MCP tool paths
#: (``cleanup`` with ``dry_run=False`` and ``empty_trash``). Only the literal
#: string ``"1"`` opens the gate — everything else (unset, ``"0"``, ``""``,
#: ``"true"``, ``"yes"``, ...) keeps the safe default. The conservative
#: matcher prevents accidental unlocks from copy/pasted shell snippets that
#: use truthy spellings.
DESTRUCTIVE_MCP_ENV = "CROSSMEM_ALLOW_DESTRUCTIVE_MCP"


def _destructive_mcp_allowed() -> bool:
    """Return ``True`` iff ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP`` is exactly ``"1"``.

    Read at every call site (not cached) so test fixtures and operators can
    flip the gate per-request via ``monkeypatch.setenv`` or ``os.environ``.
    """
    return os.environ.get(DESTRUCTIVE_MCP_ENV) == "1"


def _serialize_document(doc: Document) -> dict[str, Any]:
    """Convert a ``Document`` to a JSON-serializable dict (no embedding).

    ``Metadata.tags`` is a ``tuple[str, ...]`` internally for true
    immutability; the transport form is a JSON array, so we convert
    back to a list at the MCP boundary.
    """
    meta = asdict(doc.metadata)
    meta["tags"] = list(meta.get("tags", ()))
    return {
        "id": doc.id,
        "content": doc.content,
        "metadata": meta,
    }


def _value_error_payload(exc: ValueError) -> dict[str, str]:
    """Serialize a ``ValueError`` raised from the core into a structured payload.

    MCP tool wrappers catch ``ValueError`` and return ``{"error": ...,
    "code": "value_error"}`` so MCP clients see a normal JSON result
    instead of a Python stacktrace bubbling up through the FastMCP layer.
    """
    return {"error": str(exc), "code": "value_error"}


def _ingest_fetched(store: KnowledgeStore, fetched: list[Document]) -> None:
    """Persist documents returned by a source adapter into the store.

    Adapter outputs carry only ``content``/``source_url``/``title``/
    ``source_type`` plus any inherent tags; chunking, embedding and
    auto-tagging are performed by :meth:`KnowledgeStore.store` exactly
    like a normal manual ``store`` call. Documents with empty content
    are skipped so adapters that return placeholder entries (e.g. empty
    GitHub issue bodies) don't pollute the cache.
    """
    for doc in fetched:
        if not doc.content:
            continue
        meta = doc.metadata
        store.store(
            content=doc.content,
            source_url=meta.source_url,
            title=meta.title,
            source_type=meta.source_type,
            namespace=meta.namespace,
            tags=list(meta.tags) if meta.tags else None,
        )


def create_server(
    store: KnowledgeStore,
    source_registry: SourceRegistry | None = None,
) -> FastMCP:
    """Build a FastMCP server with ``query`` and ``store`` tools.

    Dependency injection of the :class:`KnowledgeStore` keeps the server
    testable: tests pass a ``MagicMock`` and assert call forwarding.

    ``source_registry`` controls the ``source=`` dispatch path of the
    ``query`` tool. When omitted, a fresh :func:`default_registry` is
    used so production callers get the built-in adapters wired
    automatically; tests pass a minimal registry that only contains the
    adapters they care about.
    """
    app: FastMCP = FastMCP("crossmem")
    registry = source_registry if source_registry is not None else default_registry()

    @app.tool(name="query")
    def query_tool(
        query: str,
        top_k: int = 10,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, str]:
        """Hybrid search over the knowledge base (FTS5 + vector via RRF).

        When ``source`` is supplied the registry is consulted: a cache
        miss triggers ``adapter.fetch(query)`` whose documents are
        ingested via :meth:`KnowledgeStore.store` before the query is
        retried so the caller sees freshly-cached hits. An unknown source
        returns a structured ``value_error`` payload listing the
        available source names.
        """
        try:
            adapter = registry.get(source) if source is not None else None
        except UnknownSourceError as exc:
            return _value_error_payload(exc)
        try:
            docs = store.query(query, top_k=top_k, tags=tags)
            if adapter is not None and not docs:
                fetched = adapter.fetch(query)
                if fetched:
                    _ingest_fetched(store, fetched)
                    docs = store.query(query, top_k=top_k, tags=tags)
        except ValueError as exc:
            return _value_error_payload(exc)
        return [_serialize_document(d) for d in docs]

    @app.tool(name="store")
    def store_tool(
        content: str,
        source_url: str,
        title: str = "",
        source_type: str = "manual",
        namespace: str = "default",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a document. Content is chunked and auto-tagged automatically.

        Returns ``{"ids": [...], "count": N}`` — one ID per persisted chunk.
        """
        ids = store.store(
            content=content,
            source_url=source_url,
            title=title,
            source_type=source_type,
            namespace=namespace,
            tags=tags,
        )
        return {"ids": ids, "count": len(ids)}

    @app.tool(name="delete")
    def delete_tool(
        doc_id: str | None = None,
        source_url: str | None = None,
        permanent: bool = False,
    ) -> dict[str, Any]:
        """Delete by ``doc_id`` or ``source_url``.

        Soft-deletes go to the trash JSONL; ``permanent=True`` bypasses it.
        Returns ``{"deleted": <count>}`` on success or
        ``{"error": ..., "code": "value_error"}`` on a core ``ValueError``.
        """
        try:
            n = store.delete(doc_id=doc_id, source_url=source_url, permanent=permanent)
        except ValueError as exc:
            return _value_error_payload(exc)
        return {"deleted": n}

    @app.tool(name="status")
    def status_tool() -> BackendStats:
        """Return backend stats (doc count, DB size, top tags, backend info).

        Returns the unified :class:`BackendStats` payload — every key
        (``document_count``, ``top_tags``, ``backend``, ``db_size_bytes``)
        is guaranteed by the backend contract (TODO 26.9 variant a).
        Backends that cannot measure an exact on-disk footprint
        (ephemeral Chroma, in-memory Qdrant, Qdrant servers without
        ``disk_data_size``) report ``0`` rather than omitting the key.
        """
        return store.stats()

    @app.tool(name="export")
    def export_tool(path: str, format: str = "zip") -> dict[str, Any]:  # noqa: A002 — MCP tool param name
        """Export the knowledge base as JSONL or ZIP (atomic write)."""
        n = store.export(Path(path), format=format)
        return {"exported": n, "path": path}

    @app.tool(name="import_data")
    def import_tool(path: str) -> dict[str, Any]:
        """Import a previously exported JSONL/ZIP file (validates EOF + dim).

        Returns ``{"imported": <count>}`` on success or
        ``{"error": ..., "code": "value_error"}`` if validation fails.
        """
        try:
            n = store.import_data(Path(path))
        except ValueError as exc:
            return _value_error_payload(exc)
        return {"imported": n}

    @app.tool(name="cleanup")
    def cleanup_tool(
        query: str, dry_run: bool = True, mode: str = "tag"
    ) -> dict[str, Any]:
        """Preview or execute a tag/semantic cleanup.

        Delegates to :func:`crossmem.cleanup.cleanup`:

        * ``mode="tag"`` (default) treats ``query`` as an exact tag and
          matches every document carrying that tag.
        * ``mode="semantic"`` runs a hybrid search and matches the top-K
          most relevant documents.

        Preview (``dry_run=True``) is always allowed. The destructive path
        (``dry_run=False``) requires ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1`` in
        the environment that launched the MCP server; without it the call
        is silently downgraded to a preview and the response carries
        ``forced_dry_run=True`` so the caller can detect the downgrade.

        Returns ``{"matched": [doc_ids], "deleted": <count>,
        "dry_run": <bool>, "mode": <str>}`` on success or
        ``{"error": ..., "code": "value_error"}`` if ``mode`` is invalid.
        When the destructive gate downgraded the request, the additional
        key ``"forced_dry_run": True`` is present.
        """
        effective_dry_run = dry_run
        forced = False
        if not dry_run and not _destructive_mcp_allowed():
            effective_dry_run = True
            forced = True
        try:
            result = cleanup_op(store, query, dry_run=effective_dry_run, mode=mode)
        except ValueError as exc:
            return _value_error_payload(exc)
        payload: dict[str, Any] = {
            "matched": result.previewed_ids,
            "deleted": len(result.deleted_ids),
            "dry_run": effective_dry_run,
            "mode": result.mode,
        }
        if forced:
            payload["forced_dry_run"] = True
        return payload

    @app.tool(name="empty_trash")
    def empty_trash_tool(ttl_days: int = 30) -> dict[str, Any]:
        """Drop trash entries older than ``ttl_days`` (default 30; 0 wipes all).

        Gated behind ``CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1``: without the
        opt-in this tool is a no-op that returns
        ``{"removed": 0, "blocked": True, "hint": ...}`` so the caller
        can see why nothing happened. With the env-switch the call
        delegates to :func:`crossmem.cleanup.empty_trash` and mirrors the
        ``crossmem trash empty`` CLI behavior, returning
        ``{"removed": <count>}``.
        """
        if not _destructive_mcp_allowed():
            return {
                "removed": 0,
                "blocked": True,
                "hint": (
                    f"empty_trash is blocked over MCP by default. Set "
                    f"{DESTRUCTIVE_MCP_ENV}=1 in the MCP server "
                    "environment, or run 'crossmem trash empty' from the "
                    "CLI."
                ),
            }
        removed = empty_trash_op(ttl_days=ttl_days)
        return {"removed": removed}

    @app.tool(name="restore_from_trash")
    def restore_from_trash_tool(doc_id: str) -> dict[str, Any]:
        """Re-insert a soft-deleted document from the trash.

        Returns the restored document payload (no embedding) on success
        or ``{"error": ..., "code": "value_error"}`` if ``doc_id`` is
        not present in the trash file.
        """
        try:
            doc = restore_from_trash_op(store, doc_id)
        except ValueError as exc:
            return _value_error_payload(exc)
        return _serialize_document(doc)

    return app


def main() -> None:
    """Build the active store from ``config.toml`` and run the MCP server.

    The backend choice (``sqlite``/``chroma``/``qdrant``) plus any URL or
    API key live in ``~/.crossmem/config.toml`` and are populated by
    ``crossmem configure``. SQLite uses
    :func:`crossmem.configure._default_sqlite_path` (which honours
    ``CROSSMEM_DB_PATH``) so dev/test environments can redirect the DB
    without touching the config file.
    """
    config = load_config()
    sqlite_path = _default_sqlite_path()
    try:
        backend = build_backend(config, sqlite_path=sqlite_path)
    except BackendConfigError as exc:
        # stdio MCP — write the human-readable hint to stderr and bail.
        print(f"crossmem: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    embedder = EmbeddingService()
    store = KnowledgeStore(backend, embedder)
    app = create_server(store)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
