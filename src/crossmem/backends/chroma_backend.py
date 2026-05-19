"""ChromaDB backend (optional).

Implements VectorStoreBase via the chromadb client. The chromadb dependency
is optional (`pip install crossmem[chroma]`) — importing this module without
chromadb installed raises ImportError. Tests for this backend skip when
chromadb is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from crossmem.backends.base import (
    TOP_TAGS_SAMPLE_LIMIT,
    BackendStats,
    VectorStoreBase,
    as_embedding_payload,
)
from crossmem.core.models import Document, Metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

_ITER_PAGE_SIZE = 256

# Prefix for per-tag boolean metadata keys. Chroma metadata is a flat
# scalar map, so we denormalise each tag into its own ``tag:<name>: True``
# entry. That lets ``collection.get(where={"tag:<name>": True})`` filter
# server-side (native index path). ``tags_csv`` is kept as the source of
# truth for round-tripping the tag list on read. The colon prefix avoids
# collisions with any current/future metadata field names.
_TAG_KEY_PREFIX = "tag:"


def _tag_key(tag: str) -> str:
    return f"{_TAG_KEY_PREFIX}{tag}"


class ChromaBackend(VectorStoreBase):
    """ChromaDB-backed vector store.

    Mapping notes:
      - One collection holds all documents.
      - ``tags`` is stored both as the CSV scalar ``tags_csv`` (read path,
        source of truth on reconstruct) and as per-tag boolean keys
        ``tag:<name>: True`` so :meth:`find_by_tag` can filter server-side
        via ``collection.get(where=...)`` instead of a client-side scan.
      - ``documents`` is stored **lowercased** so ``query_fts`` can use
        Chroma's server-side ``where_document={"$contains": text.lower()}``
        case-insensitively. The original-cased content is mirrored into
        metadata under ``content`` (round-trip readback) plus a
        ``content_lower`` metadata key (denormalised twin of ``documents``).
        Older clients fall back to the historical paged client-side scan.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = "crossmem",
        client: Any | None = None,
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        # Persist directory remembered so :meth:`stats` can report a unified
        # ``db_size_bytes`` for the unified contract (TODO 26.9, variant a).
        # ``None`` means ephemeral / externally-provided client — no
        # filesystem footprint is attributable to this backend.
        self._persist_path: Path | None = Path(path) if path is not None else None
        if client is not None:
            self._client = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=str(path))
        else:
            self._client = chromadb.EphemeralClient()
        self.collection = self._client.get_or_create_collection(name=collection_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _meta_to_dict(doc: Document) -> dict[str, Any]:
        m = doc.metadata
        payload: dict[str, Any] = {
            "source_url": m.source_url,
            "title": m.title,
            "source_type": m.source_type,
            "stored_at": m.stored_at,
            "embedding_model": m.embedding_model,
            "embedding_dim": m.embedding_dim,
            "namespace": m.namespace,
            "tags_csv": ",".join(m.tags),
            "content_hash": m.content_hash,
            # Original content is mirrored into metadata so read paths
            # (query_vector, get_by_url, get_by_id, iter_all, find_by_tag)
            # can reconstruct the document with its true casing even though
            # the ``documents`` field is stored lowercased — see ``store``.
            "content": doc.content or "",
            # Lowercased content denormalised at write time so ``query_fts``
            # can use Chroma's server-side ``where_document={"$contains": ...}``
            # case-insensitively. Mirrors the lowercased ``documents`` value
            # and is also surfaced as a metadata key for callers that want
            # to inspect the FTS-side text without touching ``documents``.
            "content_lower": (doc.content or "").lower(),
        }
        # Per-tag boolean keys power the server-side native-index path used
        # by find_by_tag. See _TAG_KEY_PREFIX for the rationale.
        for tag in m.tags:
            payload[_tag_key(tag)] = True
        return payload

    @staticmethod
    def _dict_to_meta(meta: dict[str, Any]) -> Metadata:
        tags_csv = meta.get("tags_csv", "") or ""
        tags = [t for t in tags_csv.split(",") if t]
        return Metadata(
            source_url=meta.get("source_url", ""),
            title=meta.get("title", ""),
            source_type=meta.get("source_type", ""),
            stored_at=meta.get("stored_at", ""),
            embedding_model=meta.get("embedding_model", ""),
            embedding_dim=int(meta.get("embedding_dim", 0)),
            namespace=meta.get("namespace", "default"),
            tags=tags,
            content_hash=meta.get("content_hash", ""),
        )

    def _build_doc(
        self,
        doc_id: str,
        content: str,
        meta: dict[str, Any],
        embedding: list[float] | None,
    ) -> Document:
        # Prefer the original-cased ``content`` mirrored in metadata.
        # ``content`` from the ``documents`` field is lowercased on write
        # (see ``store``) so the metadata copy is the source of truth for
        # readback. The positional argument is kept as a fallback for any
        # legacy row that pre-dates the denormalisation.
        original = meta.get("content") if isinstance(meta, dict) else None
        return Document(
            id=doc_id,
            content=original if original is not None else content,
            embedding=embedding if embedding is not None else (),
            metadata=self._dict_to_meta(meta),
        )

    def _stale_tag_clears(self, payloads: list[dict[str, Any]]) -> None:
        """Mutate ``payloads`` to clear tag keys removed since last upsert.

        Chroma's ``upsert`` *merges* metadata rather than replacing it, so a
        re-upsert that drops a tag would leave its ``tag:<name>: True`` key
        behind and cause stale ``find_by_tag`` hits. We fetch the current
        metadata for the affected ids, diff the old tag keys against the new
        ones, and inject ``tag:<old>: None`` into the payload to evict them
        in the same operation.
        """
        ids = [p["__id"] for p in payloads]
        if not ids:
            return
        existing = self.collection.get(ids=ids, include=["metadatas"])
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or []
        old_by_id: dict[str, set[str]] = {}
        for i, doc_id in enumerate(existing_ids):
            meta = existing_metas[i] if i < len(existing_metas) else {}
            old_by_id[doc_id] = {
                k for k in (meta or {}) if k.startswith(_TAG_KEY_PREFIX)
            }

        for payload in payloads:
            doc_id = payload["__id"]
            old_keys = old_by_id.get(doc_id, set())
            new_keys = {k for k in payload if k.startswith(_TAG_KEY_PREFIX)}
            for stale_key in old_keys - new_keys:
                payload[stale_key] = None

    # ------------------------------------------------------------------
    # VectorStoreBase
    # ------------------------------------------------------------------

    def _build_upsert_payloads(self, docs: list[Document]) -> list[dict[str, Any]]:
        """Build per-doc metadata payloads with the doc id stashed under
        ``__id`` so :meth:`_stale_tag_clears` can correlate them with the
        existing rows. The ``__id`` key is stripped before the upsert call.
        """
        payloads: list[dict[str, Any]] = []
        for doc in docs:
            payload = self._meta_to_dict(doc)
            payload["__id"] = doc.id
            payloads.append(payload)
        self._stale_tag_clears(payloads)
        return payloads

    def store(self, doc: Document) -> None:
        payloads = self._build_upsert_payloads([doc])
        meta = {k: v for k, v in payloads[0].items() if k != "__id"}
        # ``documents`` is intentionally stored lowercased: Chroma's
        # ``where_document={"$contains": ...}`` filter is case-sensitive,
        # and the FTS path needs to match queries regardless of casing.
        # The original-cased content is mirrored into metadata (see
        # ``_meta_to_dict``) and is what read paths return to callers.
        self.collection.upsert(
            ids=[doc.id],
            embeddings=[as_embedding_payload(doc.embedding)],
            documents=[(doc.content or "").lower()],
            metadatas=[meta],
        )

    def upsert_many(self, docs: list[Document]) -> None:
        """Persist every document in ``docs`` in a single ``upsert`` call.

        Chroma's client accepts parallel id/embedding/document/metadata
        arrays and dispatches them as one server-side operation, so a
        validation error rejects the whole batch before any row lands.
        """
        if not docs:
            return
        payloads = self._build_upsert_payloads(docs)
        metadatas = [
            {k: v for k, v in payload.items() if k != "__id"} for payload in payloads
        ]
        self.collection.upsert(
            ids=[doc.id for doc in docs],
            embeddings=[as_embedding_payload(doc.embedding) for doc in docs],
            # See ``store``: lowercased text is the FTS-indexed payload,
            # original casing lives in ``metadatas[i]["content"]``.
            documents=[(doc.content or "").lower() for doc in docs],
            metadatas=metadatas,
        )

    def query_vector(self, embedding: list[float], top_k: int = 10) -> list[Document]:
        result = self.collection.query(
            query_embeddings=[as_embedding_payload(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        embs = (result.get("embeddings") or [[None] * len(ids)])[0]

        out: list[Document] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            emb = embs[i] if i < len(embs) else None
            out.append(self._build_doc(doc_id, content or "", meta or {}, emb))
        return out

    def query_fts(
        self, text: str, top_k: int = 10, tags: list[str] | None = None
    ) -> list[Document]:
        """Substring search over document content with optional tag filter.

        Uses Chroma's server-side ``where_document={"$contains": text}``
        against the lowercased ``documents`` field (see :meth:`store`):
        both sides are lowercased so the case-insensitive contract is
        preserved while letting Chroma do the scan internally. The tag
        pre-filter is composed via ``where`` so only candidate rows are
        materialised. If the client refuses the filter shape (older
        builds or future schema changes), :meth:`_query_fts_scan_fallback`
        replays the historical client-side page walk.
        """
        if not text:
            return []
        needle = text.lower()
        where = self._tag_where(tags) if tags else None
        try:
            result = self.collection.get(
                where=where,
                where_document={"$contains": needle},
                limit=top_k,
                include=["documents", "metadatas", "embeddings"],
            )
        except (TypeError, ValueError) as exc:
            # Feature-probe fallback: a client that does not accept the
            # ``where_document`` shape (older API, mocked client, server
            # downgrade) raises a validation error rather than a runtime
            # one. Fall through to the client-side scan so the backend
            # keeps working without server-side acceleration.
            return self._query_fts_scan_fallback(needle, top_k, tags, exc=exc)
        return self._unpack_fts_result(result, top_k)

    def _query_fts_scan_fallback(
        self,
        needle: str,
        top_k: int,
        tags: list[str] | None,
        *,
        exc: Exception | None = None,
    ) -> list[Document]:
        """Client-side paginated substring scan used when the server-side
        ``where_document`` filter is not available. Mirrors the pre-26.3
        behaviour: page through the collection, match locally, stop once
        ``top_k`` rows accumulate.
        """
        del exc  # logging hook for future diagnostics
        out: list[Document] = []
        offset = 0
        while True:
            page = self.collection.get(
                limit=_ITER_PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = page.get("ids") or []
            if not ids:
                break
            docs = page.get("documents") or []
            metas = page.get("metadatas") or []
            embs = page.get("embeddings")
            if embs is None:
                embs = [None] * len(ids)

            for i, doc_id in enumerate(ids):
                content = docs[i] if i < len(docs) else ""
                if needle not in (content or "").lower():
                    continue
                meta = metas[i] if i < len(metas) else {}
                emb = embs[i] if i < len(embs) else None
                doc_obj = self._build_doc(doc_id, content or "", meta or {}, emb)
                if tags and not any(t in doc_obj.metadata.tags for t in tags):
                    continue
                out.append(doc_obj)
                if len(out) >= top_k:
                    return out

            if len(ids) < _ITER_PAGE_SIZE:
                break
            offset += len(ids)
        return out

    def _unpack_fts_result(self, result: dict[str, Any], top_k: int) -> list[Document]:
        """Decode a ``collection.get`` payload into Document objects.

        Used by the server-side fast path of :meth:`query_fts`. The
        ``where`` clause has already constrained the result to matching
        rows, so no additional filtering is required here.
        """
        ids = result.get("ids") or []
        if not ids:
            return []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        embs = result.get("embeddings")
        if embs is None:
            embs = [None] * len(ids)
        out: list[Document] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            emb = embs[i] if i < len(embs) else None
            out.append(self._build_doc(doc_id, content or "", meta or {}, emb))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _tag_where(tags: list[str]) -> dict[str, Any]:
        """Compose the metadata ``where`` clause for a tag pre-filter.

        Each stored doc carries a ``tag:<name>: True`` key per tag (see
        :meth:`_meta_to_dict`), so a single tag becomes an equality
        check and a list of tags becomes an ``$or`` of equality checks.
        """
        clauses = [{_tag_key(t): True} for t in tags]
        if len(clauses) == 1:
            return clauses[0]
        return {"$or": clauses}

    def delete(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def get_by_url(self, source_url: str) -> list[Document]:
        """Retrieve documents by source URL.

        Results are sorted by document id so callers see a deterministic
        sequence — Chroma's ``collection.get`` does not guarantee any
        particular order, so we impose one explicitly to match the
        contract shared with the SQLite and Qdrant backends.
        """
        result = self.collection.get(
            where={"source_url": source_url},
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        # ``embeddings`` is returned as a numpy array by recent chromadb
        # releases. ``or [None] * len(ids)`` raises on the array truthiness
        # check, so use an explicit ``is None`` test (same pattern as
        # ``query_fts``/``get_by_id``).
        embs = result.get("embeddings")
        if embs is None:
            embs = [None] * len(ids)

        out: list[Document] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            emb = embs[i] if i < len(embs) else None
            out.append(self._build_doc(doc_id, content or "", meta or {}, emb))
        out.sort(key=lambda d: d.id)
        return out

    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document by id, including its embedding."""
        result = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        embs = result.get("embeddings")
        if embs is None:
            embs = [None] * len(ids)
        content = docs[0] if docs else ""
        meta = metas[0] if metas else {}
        emb = embs[0] if embs is not None and len(embs) > 0 else None
        return self._build_doc(ids[0], content or "", meta or {}, emb)

    def iter_all(self) -> Iterator[Document]:
        """Yield every stored document with its embedding (paginated)."""
        offset = 0
        while True:
            result = self.collection.get(
                limit=_ITER_PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = result.get("ids") or []
            if not ids:
                return

            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            embs = result.get("embeddings")
            if embs is None:
                embs = [None] * len(ids)

            for i, doc_id in enumerate(ids):
                content = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                emb = embs[i] if i < len(embs) else None
                yield self._build_doc(doc_id, content or "", meta or {}, emb)

            if len(ids) < _ITER_PAGE_SIZE:
                return
            offset += len(ids)

    def find_by_tag(self, tag: str) -> Iterator[Document]:
        """Yield every document tagged ``tag`` using Chroma's native index.

        Each tag is denormalised at write time into a ``tag:<name>: True``
        metadata key (see :meth:`_meta_to_dict`), so the lookup is a
        server-side equality filter via ``collection.get(where=...)``. This
        keeps memory bounded by paginating with ``limit``/``offset`` and
        avoids the prior full-collection scan + CSV split. Tokenised tags
        cannot collide on prefixes either: a doc tagged only ``python``
        carries ``tag:python`` but no ``tag:py`` key, so ``find_by_tag("py")``
        returns no match.
        """
        offset = 0
        where = {_tag_key(tag): True}
        while True:
            result = self.collection.get(
                where=where,
                limit=_ITER_PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = result.get("ids") or []
            if not ids:
                return
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            embs = result.get("embeddings")
            if embs is None:
                embs = [None] * len(ids)

            for i, doc_id in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                content = docs[i] if i < len(docs) else ""
                emb = embs[i] if i < len(embs) else None
                yield self._build_doc(doc_id, content or "", meta or {}, emb)

            if len(ids) < _ITER_PAGE_SIZE:
                return
            offset += len(ids)

    def _disk_usage_bytes(self) -> int:
        """Return the on-disk footprint of the persist directory.

        Chroma's persistent client materialises a SQLite file plus index
        sub-directories under ``self._persist_path``; the unified
        ``db_size_bytes`` contract (TODO 26.9, variant a) asks for the
        sum of every file in that tree. Ephemeral clients (no path) and
        externally-supplied clients have no attributable footprint, so
        they report ``0``. Missing/inaccessible files are skipped rather
        than raising — stats() must never break the MCP ``status`` tool.
        """
        path = self._persist_path
        if path is None or not path.exists():
            return 0
        total = 0
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                # Files that vanish mid-walk or that the process cannot
                # stat (permission, race) are ignored — the result stays
                # best-effort consistent with the SQLite path.
                continue
        return total

    def stats(self) -> BackendStats:
        document_count = self.collection.count()

        # Best-effort top-tags from a sample.
        top_tags: list[tuple[str, int]] = []
        if document_count:
            sample = self.collection.get(
                limit=min(document_count, TOP_TAGS_SAMPLE_LIMIT),
                include=["metadatas"],
            )
            counts: dict[str, int] = {}
            for meta in sample.get("metadatas") or []:
                tags_csv = (meta or {}).get("tags_csv", "") or ""
                for tag in tags_csv.split(","):
                    if tag:
                        counts[tag] = counts.get(tag, 0) + 1
            top_tags = [
                (t, c)
                for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            ]

        return {
            "backend": "chroma",
            "document_count": document_count,
            "top_tags": top_tags,
            "db_size_bytes": self._disk_usage_bytes(),
        }
