"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crossmem.core.models import Document


def as_embedding_payload(vec: Any) -> list[float]:
    """Return ``vec`` as the ``list[float]`` form optional backends expect.

    :class:`crossmem.core.models.Document` stores embeddings as a
    :data:`crossmem.core.models.Vec` tuple (frozen-dataclass promise +
    hashability). Some optional backend clients (notably chromadb's
    ``normalize_embeddings`` validator in 1.x) explicitly reject tuples
    and accept only ``list[float]`` or ``numpy.ndarray``. This single
    helper localises the conversion so the chroma and qdrant backends
    don't sprinkle ``list(...)`` casts across every call site; if a
    future client release widens its accepted types, the cast is
    removed in exactly one place.
    """
    return list(vec)


class _BackendStatsRequired(TypedDict):
    """Required-key portion of the stats contract (``total=True`` default).

    TODO 26.9 (variant a) folded ``db_size_bytes`` into the required-key
    set: every backend now reports an on-disk footprint as an integer so
    the MCP ``status`` payload has a uniform shape regardless of which
    backend is in use. The split between this required base and
    :class:`BackendStats` is kept so future optional keys remain easy to
    add without re-typing the existing shape — but for now every key is
    required and ``BackendStats`` simply re-exposes the required set.
    """

    document_count: int
    top_tags: list[tuple[str, int]]
    backend: str
    db_size_bytes: int


class BackendStats(_BackendStatsRequired, total=False):
    """Stats payload returned by every backend's :meth:`stats` method.

    Inherits every key from :class:`_BackendStatsRequired`. After TODO 26.9
    (variant a) all four keys — ``document_count``, ``top_tags``,
    ``backend`` and ``db_size_bytes`` — are required: each backend
    reports a non-negative integer footprint (SQLite: ``stat().st_size``
    or pragma-derived page math; Chroma: walked persist directory or
    ``0`` for ephemeral clients; Qdrant: ``info.disk_data_size`` when
    the server exposes it, ``0`` otherwise). The ``total=False`` shell
    is kept so future optional keys can be appended without re-typing
    the required base — no key is currently optional. Backends that
    omit a required key are caught by the type checker at the dict
    literal site and at runtime by :meth:`VectorStoreBase.stats` (see
    ``__init_subclass__``).
    """


REQUIRED_STATS_KEYS: frozenset[str] = frozenset(_BackendStatsRequired.__annotations__)
"""Runtime mirror of the mandatory keys in :class:`_BackendStatsRequired`.

Derived from ``__annotations__`` so the constant cannot drift away from
the TypedDict declaration. Used by :meth:`VectorStoreBase.stats` to
validate concrete backends and by the cross-backend contract test.
"""


TOP_TAGS_SAMPLE_LIMIT: int = 1000
"""Upper bound for the best-effort ``stats()`` tag sample in non-SQLite backends.

Tag frequency in real corpora follows a Zipf distribution: the top 10 tags
stabilise long before 1000 sampled documents, so scanning further would
only burn I/O without improving the result.
"""


class VectorStoreBase(ABC):
    """Abstract base class for vector store backends."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap concrete ``stats`` implementations with a runtime key check.

        A backend that forgets a required :class:`BackendStats` key would
        otherwise pass the type checker silently if the offending dict
        literal escapes (e.g. via ``Any``). Wrapping each concrete subclass'
        ``stats`` ensures the contract is also enforced at call time, so a
        regression surfaces in the existing test suite instead of leaking
        through the MCP ``status`` tool. Wrapping happens at class creation;
        subclasses that don't override ``stats`` inherit the wrapped version
        from their parent and need no extra wrap.
        """
        super().__init_subclass__(**kwargs)
        own_stats = cls.__dict__.get("stats")
        if own_stats is None or getattr(own_stats, "__isabstractmethod__", False):
            return

        def _checked_stats(self: VectorStoreBase, _impl=own_stats) -> BackendStats:
            payload = _impl(self)
            missing = REQUIRED_STATS_KEYS - set(payload)
            if missing:
                raise TypeError(
                    f"{type(self).__name__}.stats() missing required stats "
                    f"keys: {sorted(missing)}"
                )
            return payload

        _checked_stats.__wrapped__ = own_stats  # type: ignore[attr-defined]
        _checked_stats.__doc__ = own_stats.__doc__
        _checked_stats.__name__ = own_stats.__name__
        _checked_stats.__qualname__ = own_stats.__qualname__
        cls.stats = _checked_stats  # type: ignore[method-assign]

    @abstractmethod
    def store(self, doc: Document) -> None:
        """Store a document with its embedding."""

    def upsert_many(self, docs: list[Document]) -> None:
        """Persist ``docs`` as a single atomic batch.

        Used by :meth:`crossmem.core.store.KnowledgeStore.store` to commit
        every chunk of a logical document in one transaction so a crash
        mid-flight cannot leave a half-persisted document behind. Backends
        MUST implement true all-or-nothing semantics: if any element of
        ``docs`` cannot be stored, the persistent state for the whole batch
        is rolled back. The default fallback iterates one-by-one and is
        therefore NOT atomic; backends that participate in multi-chunk
        ``store()`` MUST override this method.

        An empty ``docs`` list is a no-op.
        """
        for doc in docs:
            self.store(doc)

    @abstractmethod
    def query_vector(self, embedding: list[float], top_k: int) -> list[Document]:
        """Query documents by vector similarity."""

    @abstractmethod
    def query_fts(
        self, text: str, top_k: int, tags: list[str] | None = None
    ) -> list[Document]:
        """Query documents by full-text search, optionally filtered by tags."""

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""

    @abstractmethod
    def get_by_url(self, source_url: str) -> list[Document]:
        """Retrieve documents by source URL."""

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a document by primary id, or ``None`` if not present.

        Returns a fully hydrated :class:`Document` (including its embedding)
        so the caller can soft-delete it (i.e. append the full record to the
        trash JSONL) without a second backend round-trip. Implementations
        MUST return ``None`` for unknown ids rather than raising.
        """

    @abstractmethod
    def stats(self) -> BackendStats:
        """Return store statistics (doc count, size, top tags, backend info)."""

    @abstractmethod
    def iter_all(self) -> Iterator[Document]:
        """Yield every stored document with its embedding (used by export)."""

    @abstractmethod
    def find_by_tag(self, tag: str) -> Iterator[Document]:
        """Yield every document whose metadata tags contain ``tag`` exactly.

        Each backend MUST use a native index-backed lookup so the call is
        cheap even on million-doc corpora (SQLite: ``INNER JOIN
        document_tags`` on ``idx_tag``; Chroma: ``collection.get`` with a
        tag-membership ``where`` clause + pagination; Qdrant: ``scroll``
        with a ``FieldCondition`` on the ``tags`` payload index).

        Used by :func:`crossmem.cleanup._find_by_tag` to drive
        ``cleanup --mode tag`` without going through the RRF / over-fetch
        path of :meth:`crossmem.core.store.KnowledgeStore.query`; the
        contract therefore returns an iterator so callers can stream
        large hit sets one document at a time. Embeddings MAY be omitted
        from the returned :class:`Document` (cleanup only needs the id,
        but soft-delete paths consume the full document via the existing
        ``get_by_id`` round-trip).
        """
