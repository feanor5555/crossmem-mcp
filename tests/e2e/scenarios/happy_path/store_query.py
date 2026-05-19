"""store -> query roundtrip via the in-process MCP store (task 27.3c).

The scenario is intentionally LLM-free at the data-plane level: the
matrix layer decides whether the *runner* talks to a real or mocked
LLM, but the assertion under test is the CrossMem MCP roundtrip
(store a document, query it back). That keeps the scenario fast,
deterministic, and meaningful even when no live endpoint is reachable.

The callable returns ``0`` when the stored content is found by a
plain-text query, non-zero otherwise. The :func:`_build_store`
indirection exists so unit tests can inject a broken store and
verify the failure path without monkeypatching deep internals.
"""

from __future__ import annotations

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

#: A keyword that survives chunking / tokenisation untouched and is
#: unique enough that a hybrid FTS+vector search will surface the
#: exact document we just stored without false positives.
_PROBE_KEYWORD = "crossmem-e2e-happy-roundtrip-marker"

_DOC_CONTENT = (
    "End-to-end happy-path probe document. "
    f"Unique marker: {_PROBE_KEYWORD}. "
    "If a query for the marker fails to surface this document, the "
    "MCP store roundtrip is broken."
)


def _build_store() -> object:
    """Construct an in-memory ``KnowledgeStore`` for the scenario.

    Split into its own helper so unit tests can substitute a broken
    implementation and assert the scenario propagates the failure.
    """
    backend = SQLiteBackend(":memory:")
    embedder = FixedEmbedder(model_name="e2e-mock")
    return KnowledgeStore(backend=backend, embedder=embedder)


def run() -> int:
    """Execute the store -> query roundtrip; return ``0`` on success."""
    store = _build_store()
    ids = store.store(  # type: ignore[attr-defined]
        content=_DOC_CONTENT,
        source_url="https://example.invalid/e2e/happy-roundtrip",
        title="E2E Happy Path Probe",
        source_type="manual",
    )
    if not ids:
        return 1
    hits = store.query(_PROBE_KEYWORD, top_k=5)  # type: ignore[attr-defined]
    if not hits:
        return 2
    # Make sure the result really refers to the doc we just stored,
    # not some accidental noise hit.
    if not any(_PROBE_KEYWORD in (getattr(h, "content", "") or "") for h in hits):
        return 3
    return 0


__all__ = ["run"]
