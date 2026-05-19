"""Document and Metadata dataclasses for CrossMem."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


# Public type alias for embedding vectors.
#
# Stored as ``tuple[float, ...]`` so the frozen-dataclass promise covers
# the vector container — :class:`Document` instances stay truly immutable
# and remain hashable. Values carry fp16 precision: the
# ``EmbeddingService`` rounds every vector to ``np.float16`` before it
# leaves the embedder, and the SQLite backend widens it back to fp32 for
# ``vec0`` storage without recovering the lost mantissa bits. Callers
# that need bit-exact fp32 vectors will not get them — see the module
# docstring in ``crossmem.core.embedding`` and the "Embedding" section in
# ``CLAUDE.md`` for the rationale.
Vec = tuple[float, ...]  # fp16 precision, immutable container


@dataclass(frozen=True)
class Metadata:
    """Metadata associated with a stored document.

    ``tags`` is stored as a ``tuple[str, ...]`` so the frozen-dataclass
    promise is enforced down to the container level: callers cannot
    mutate ``meta.tags`` via ``append`` / ``clear`` / ``[i] = ...``.
    Construction accepts any iterable and coerces to tuple in
    ``__post_init__``.
    """

    source_url: str
    title: str
    source_type: str  # "web" | "github" | "context7" | ...
    stored_at: str  # ISO 8601
    embedding_model: str
    embedding_dim: int
    namespace: str = "default"
    tags: tuple[str, ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        # Coerce any iterable input (list, generator, tuple) to a tuple so
        # frozen-ness extends to the container itself. ``object.__setattr__``
        # is required because the dataclass is frozen.
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class Document:
    """A document stored in the knowledge database.

    ``embedding`` is stored as a ``tuple[float, ...]`` (see :data:`Vec`)
    so the frozen-dataclass promise covers the vector container. This
    also restores automatic ``__hash__`` generation, which mutable fields
    had silently suppressed — :class:`Document` instances are now usable
    as dict keys / set members. Construction accepts any iterable and
    coerces to tuple in ``__post_init__``.

    The vector's values carry **fp16 precision**: the
    :class:`crossmem.core.embedding.EmbeddingService` rounds every vector
    to ``np.float16`` before returning it, and the SQLite backend widens
    it back to fp32 for sqlite-vec's ``vec0`` storage without recovering
    the bits lost to the round-trip. Callers comparing two embeddings
    must therefore allow for fp16 tolerances even when they re-read the
    value from the fp32 store. See :data:`Vec` and the ``embedding``
    module docstring for the full rationale.
    """

    id: str  # SHA-256(namespace + source_url + content_hash)[:32]
    content: str
    embedding: Vec  # 384-dim, fp16 precision, immutable tuple
    metadata: Metadata

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, tuple):
            object.__setattr__(self, "embedding", tuple(self.embedding))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Document to a JSON-friendly dict.

        ``metadata`` becomes a nested dict; ``embedding`` is materialized
        as a list. Inverse of :meth:`from_dict`.
        """
        return asdict(self)

    @classmethod
    def from_payload(
        cls,
        *,
        content: str,
        source_url: str,
        title: str,
        source_type: str,
        namespace: str = "default",
        tags: Iterable[str] | None = None,
        embedding: Vec | None = None,
        embedding_model: str = "",
        embedding_dim: int = 0,
        stored_at: str | None = None,
        chunk_index: int | None = None,
    ) -> Document:
        """Build a fresh Document with id and content_hash derived from inputs.

        ``stored_at`` defaults to the current UTC time in ISO 8601.
        ``embedding`` defaults to an empty list (caller embeds later).
        ``embedding_dim`` defaults to ``len(embedding)`` when an embedding
        is supplied and ``embedding_dim`` was left at its zero default.

        ``chunk_index`` is mixed into the ID when supplied so duplicate-
        content chunks within the same document never collide. Callers that
        emit one Document per fetch (source adapters) should leave it as
        ``None``; ``KnowledgeStore.store`` passes ``ch.chunk_index`` per
        chunk.
        """
        content_hash = generate_content_hash(content)
        doc_id = generate_id(namespace, source_url, content_hash, chunk_index)
        if stored_at is None:
            stored_at = datetime.now(timezone.utc).isoformat()
        embedding_tuple: Vec = (
            tuple(float(x) for x in embedding) if embedding is not None else ()
        )
        if embedding_dim == 0 and embedding_tuple:
            embedding_dim = len(embedding_tuple)
        metadata = Metadata(
            source_url=source_url,
            title=title,
            source_type=source_type,
            stored_at=stored_at,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            namespace=namespace,
            tags=list(tags) if tags is not None else [],
            content_hash=content_hash,
        )
        return cls(
            id=doc_id,
            content=content,
            embedding=embedding_tuple,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Inverse of :meth:`to_dict`. Reconstruct a Document from a dict.

        Accepts the bare ``to_dict()`` output. Trash records wrap the doc
        as ``{"deleted_at": ..., "doc": {...}}`` — callers must unwrap
        the ``"doc"`` key before passing it here.

        Raises ``ValueError`` if required fields are missing.
        """
        if "embedding" not in data:
            raise ValueError(
                "Document.from_dict: missing 'embedding' field; "
                f"got keys={sorted(data)}"
            )
        try:
            meta = data["metadata"]
            return cls(
                id=data["id"],
                content=data["content"],
                embedding=list(data["embedding"]),
                metadata=Metadata(
                    source_url=meta["source_url"],
                    title=meta.get("title", ""),
                    source_type=meta.get("source_type", ""),
                    stored_at=meta.get("stored_at", ""),
                    embedding_model=meta.get("embedding_model", ""),
                    embedding_dim=meta.get("embedding_dim", 0),
                    namespace=meta.get("namespace", "default"),
                    tags=list(meta.get("tags", [])),
                    content_hash=meta.get("content_hash", ""),
                ),
            )
        except KeyError as exc:
            raise ValueError(f"document JSON missing required field: {exc}") from exc


def generate_id(
    namespace: str,
    source_url: str,
    content_hash: str,
    chunk_index: int | None = None,
) -> str:
    """Generate a deterministic document ID.

    When ``chunk_index`` is supplied it is mixed into the hash so two
    chunks of the same document with identical content (boilerplate, high
    overlap) still receive distinct IDs and never silently overwrite each
    other at the backend layer. ``chunk_index=None`` (the default)
    preserves the legacy single-document derivation used by source
    adapters that emit one ``Document`` per fetch — those callers don't
    chunk and don't need chunk-level disambiguation.
    """
    base = namespace + source_url + content_hash
    if chunk_index is not None:
        # Use a delimiter that cannot appear in the preceding components
        # so the indexed and unindexed namespaces stay disjoint.
        base += f"\x00chunk={chunk_index}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def generate_content_hash(content: str) -> str:
    """Generate a SHA-256 hash of the content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
