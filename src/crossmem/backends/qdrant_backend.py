"""Qdrant backend (optional).

Implements VectorStoreBase via the qdrant-client. The qdrant-client dependency
is optional (`pip install crossmem[qdrant]`) — importing this module without
qdrant-client installed raises ImportError. Tests for this backend skip when
qdrant-client is unavailable.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from crossmem.backends.base import (
    TOP_TAGS_SAMPLE_LIMIT,
    BackendStats,
    VectorStoreBase,
    as_embedding_payload,
)
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import Document, Metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

_ITER_PAGE_SIZE = 256


class QdrantBackend(VectorStoreBase):
    """Qdrant-backed vector store.

    Mapping notes:
      - One collection holds all documents.
      - The Qdrant point id is a 64-bit hash of the CrossMem ``doc.id``
        (Qdrant requires uint64 or UUID). The original ``doc.id`` is kept
        in the payload as ``doc_id`` and used for round-trip identity.
      - Metadata is stored flat in the point payload (including ``tags``
        as a list — Qdrant supports list payload values natively).
      - ``query_fts`` uses Qdrant's ``MatchText`` filter against the
        ``content`` payload, backed by a TEXT payload index created in
        :meth:`_ensure_collection`. Older clients/servers fall back to
        the historical client-side scroll scan via the ``MatchText``
        feature probe.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "crossmem",
        client: Any | None = None,
        vector_size: int = EMBEDDING_DIM,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self.collection_name = collection_name
        self.vector_size = vector_size

        if client is not None:
            self._client = client
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # In-memory mode (also used by tests).
            self._client = QdrantClient(location=":memory:")

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        qm = self._qmodels
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
            # Payload index on tags speeds up tag filters.
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tags",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_url",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            # TEXT payload index on ``content`` so ``query_fts`` can use
            # ``MatchText`` server-side instead of paging the full
            # collection. Older qdrant builds without the TEXT schema
            # type are detected via ``hasattr`` so we keep the collection
            # scrollable on the fallback path; remote servers that
            # reject the create call (already-exists or unsupported)
            # are swallowed for the same reason.
            text_schema = getattr(qm.PayloadSchemaType, "TEXT", None)
            if text_schema is not None:
                # pragma: no cover - depends on server state
                with contextlib.suppress(Exception):
                    self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="content",
                        field_schema=text_schema,
                    )

    @staticmethod
    def _point_id(doc_id: str) -> int:
        """Stable uint64 point id derived from doc.id.

        Qdrant accepts uint64 (0..2**64-1) or a UUID as the point id. We
        use the full 64-bit hash of the first 16 hex chars of ``doc.id``;
        ``doc.id`` is itself a 32-char SHA-256 prefix, so 16 hex chars is
        the maximum that fits in uint64 deterministically. No top-bit
        masking — the previous 63-bit clamp was a defensive workaround
        against signed-int serializers and is no longer required by the
        Qdrant client. The original ``doc_id`` is stored in the payload
        for collision-resistant identity checks (see :meth:`get_by_id`
        and :meth:`delete`).
        """
        return int(doc_id[:16], 16)

    @staticmethod
    def _meta_to_payload(doc: Document) -> dict[str, Any]:
        m = doc.metadata
        return {
            "doc_id": doc.id,
            "content": doc.content,
            "source_url": m.source_url,
            "title": m.title,
            "source_type": m.source_type,
            "stored_at": m.stored_at,
            "embedding_model": m.embedding_model,
            "embedding_dim": m.embedding_dim,
            "namespace": m.namespace,
            "tags": list(m.tags),
            "content_hash": m.content_hash,
        }

    @staticmethod
    def _payload_to_doc(
        payload: dict[str, Any], embedding: list[float] | None
    ) -> Document:
        tags = payload.get("tags") or []
        return Document(
            id=payload.get("doc_id", ""),
            content=payload.get("content", ""),
            embedding=embedding if embedding is not None else (),
            metadata=Metadata(
                source_url=payload.get("source_url", ""),
                title=payload.get("title", ""),
                source_type=payload.get("source_type", ""),
                stored_at=payload.get("stored_at", ""),
                embedding_model=payload.get("embedding_model", ""),
                embedding_dim=int(payload.get("embedding_dim", 0)),
                namespace=payload.get("namespace", "default"),
                tags=list(tags),
                content_hash=payload.get("content_hash", ""),
            ),
        )

    # ------------------------------------------------------------------
    # VectorStoreBase
    # ------------------------------------------------------------------

    def store(self, doc: Document) -> None:
        qm = self._qmodels
        point = qm.PointStruct(
            id=self._point_id(doc.id),
            vector=as_embedding_payload(doc.embedding),
            payload=self._meta_to_payload(doc),
        )
        self._client.upsert(collection_name=self.collection_name, points=[point])

    def upsert_many(self, docs: list[Document]) -> None:
        """Persist every document in ``docs`` in one ``upsert`` request.

        Qdrant's ``upsert`` accepts a list of points and applies them as
        one operation on the remote — either every chunk lands or the
        call raises before any of them does.
        """
        if not docs:
            return
        qm = self._qmodels
        points = [
            qm.PointStruct(
                id=self._point_id(doc.id),
                vector=as_embedding_payload(doc.embedding),
                payload=self._meta_to_payload(doc),
            )
            for doc in docs
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query_vector(self, embedding: list[float], top_k: int = 10) -> list[Document]:
        # ``QdrantClient.search`` was removed in qdrant-client 1.13; the
        # current replacement is ``query_points``, which returns a
        # ``QueryResponse`` (``.points`` is the list of ``ScoredPoint``s
        # with the same ``payload`` / ``vector`` shape).
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=as_embedding_payload(embedding),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        out: list[Document] = []
        for hit in response.points:
            payload = hit.payload or {}
            vector = hit.vector if hit.vector is not None else None
            out.append(self._payload_to_doc(payload, vector))
        return out

    def query_fts(
        self, text: str, top_k: int = 10, tags: list[str] | None = None
    ) -> list[Document]:
        """Substring search over payload ``content`` with optional tag filter.

        Fast path: a single ``scroll`` call with a filter that combines
        ``MatchText(text=text)`` against the ``content`` payload (backed
        by the TEXT index registered in :meth:`_ensure_collection`) with
        the existing tag membership clause. The server returns only
        matching points, replacing the pre-26.3 page walk.

        Feature probe: if the installed client lacks ``MatchText`` or
        the server refuses the filter (older qdrant build, mocked
        client, payload index missing), :meth:`_query_fts_scroll_fallback`
        replays the historical full-scan substring loop.
        """
        if not text:
            return []
        qm = self._qmodels
        match_text_cls = getattr(qm, "MatchText", None)
        if match_text_cls is None:
            return self._query_fts_scroll_fallback(text, top_k, tags)
        must: list[Any] = [
            qm.FieldCondition(key="content", match=match_text_cls(text=text))
        ]
        should = (
            [qm.FieldCondition(key="tags", match=qm.MatchValue(value=t)) for t in tags]
            if tags
            else None
        )
        scroll_filter = (
            qm.Filter(must=must, should=should) if should else qm.Filter(must=must)
        )
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            # Feature-probe fallback: any client/server rejection lands
            # on the client-side scan so the backend remains functional.
            return self._query_fts_scroll_fallback(text, top_k, tags)
        out: list[Document] = []
        for p in points:
            payload = p.payload or {}
            out.append(self._payload_to_doc(payload, p.vector))
            if len(out) >= top_k:
                break
        return out

    def _query_fts_scroll_fallback(
        self,
        text: str,
        top_k: int,
        tags: list[str] | None,
    ) -> list[Document]:
        """Client-side paginated substring scan over payload ``content``.

        Used when the server-side ``MatchText`` filter is unavailable
        (client without ``MatchText``, mocked client, server rejects the
        filter shape). Mirrors the pre-26.3 behaviour: scroll the
        collection in pages of ``_ITER_PAGE_SIZE`` applying a tag
        pre-filter when supplied, and break as soon as ``top_k`` rows
        accumulate.
        """
        qm = self._qmodels
        scroll_filter = None
        if tags:
            scroll_filter = qm.Filter(
                should=[
                    qm.FieldCondition(key="tags", match=qm.MatchValue(value=tag))
                    for tag in tags
                ]
            )
        out: list[Document] = []
        next_offset: Any = None
        text_lower = text.lower()
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=_ITER_PAGE_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                content = payload.get("content", "") or ""
                if text_lower not in content.lower():
                    continue
                out.append(self._payload_to_doc(payload, p.vector))
                if len(out) >= top_k:
                    return out
            if next_offset is None:
                break
        return out

    def delete(self, doc_id: str) -> None:
        """Delete the document identified by ``doc_id``.

        Mirrors :meth:`get_by_id`: we first ``retrieve`` the derived point
        id and confirm the stored ``payload["doc_id"]`` equals the input
        ``doc_id`` before issuing the destructive ``delete`` call. If the
        retrieved point's payload identifies a different CrossMem
        document (extremely unlikely 64-bit hash collision), we raise
        ``ValueError`` rather than silently deleting the wrong row.
        Missing points are a no-op (consistent with the prior behaviour).
        """
        qm = self._qmodels
        point_id = self._point_id(doc_id)
        points = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return
        payload = points[0].payload or {}
        stored_id = payload.get("doc_id")
        if stored_id != doc_id:
            raise ValueError(
                f"qdrant delete: point id collision — stored doc_id "
                f"{stored_id!r} does not match requested {doc_id!r}"
            )
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qm.PointIdsList(points=[point_id]),
        )

    def get_by_url(self, source_url: str) -> list[Document]:
        """Retrieve documents by source URL.

        Qdrant's ``scroll`` paginates by internal point id (the 63-bit
        hash of the CrossMem doc id) and offers no stable ordering on
        payload fields, so we sort the assembled list by the original
        CrossMem id. This matches the order returned by the SQLite and
        Chroma backends and keeps soft-delete trash output reproducible.
        """
        qm = self._qmodels
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="source_url", match=qm.MatchValue(value=source_url)
                )
            ]
        )
        out: list[Document] = []
        next_offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=_ITER_PAGE_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                out.append(self._payload_to_doc(payload, p.vector))
            if next_offset is None:
                break
        out.sort(key=lambda d: d.id)
        return out

    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document by its CrossMem id, embedding included.

        Qdrant point ids are the 63-bit hash of ``doc_id`` (see
        :meth:`_point_id`); the original CrossMem id is also stored in the
        payload as ``doc_id`` for round-trip identity. We look up the
        derived point id and confirm the payload ``doc_id`` to defend
        against extremely unlikely hash collisions.
        """
        points = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[self._point_id(doc_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        point = points[0]
        payload = point.payload or {}
        if payload.get("doc_id") != doc_id:
            return None
        return self._payload_to_doc(payload, point.vector)

    def iter_all(self) -> Iterator[Document]:
        """Yield every stored document with its embedding (paginated)."""
        next_offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=_ITER_PAGE_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                return
            for p in points:
                payload = p.payload or {}
                yield self._payload_to_doc(payload, p.vector)
            if next_offset is None:
                return

    def find_by_tag(self, tag: str) -> Iterator[Document]:
        """Yield every document whose payload tags contain ``tag``.

        Uses the ``tags`` payload index registered in
        :meth:`_ensure_collection`: ``FieldCondition(key="tags",
        match=MatchValue(value=tag))`` against a list-valued payload
        matches any point whose ``tags`` list contains ``tag``.
        Pagination via ``scroll`` keeps memory bounded; the yield is
        lazy so callers can stream large hit sets.
        """
        qm = self._qmodels
        flt = qm.Filter(
            must=[qm.FieldCondition(key="tags", match=qm.MatchValue(value=tag))]
        )
        next_offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=_ITER_PAGE_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                return
            for p in points:
                payload = p.payload or {}
                yield self._payload_to_doc(payload, p.vector)
            if next_offset is None:
                return

    def stats(self) -> BackendStats:
        info = self._client.get_collection(collection_name=self.collection_name)
        document_count = int(getattr(info, "points_count", 0) or 0)

        # TODO 26.9 (variant a): unified ``db_size_bytes`` contract.
        # Qdrant servers expose the on-disk footprint via
        # ``CollectionInfo.disk_data_size`` on recent builds; the in-memory
        # client and older releases omit the attribute, in which case we
        # fall back to ``0`` so the contract still holds. ``None`` is
        # treated as "unknown" and also reported as ``0`` — the unified
        # type promises an integer, and a missing remote field is no
        # different from an empty collection from the caller's
        # perspective.
        disk_size_raw = getattr(info, "disk_data_size", 0)
        db_size_bytes = int(disk_size_raw) if disk_size_raw else 0

        top_tags: list[tuple[str, int]] = []
        if document_count:
            counts: dict[str, int] = {}
            sampled = 0
            next_offset: Any = None
            while sampled < min(document_count, TOP_TAGS_SAMPLE_LIMIT):
                points, next_offset = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=_ITER_PAGE_SIZE,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                for p in points:
                    for tag in (p.payload or {}).get("tags") or []:
                        counts[tag] = counts.get(tag, 0) + 1
                    sampled += 1
                if next_offset is None:
                    break
            top_tags = [
                (t, c)
                for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            ]

        return {
            "backend": "qdrant",
            "document_count": document_count,
            "top_tags": top_tags,
            "db_size_bytes": db_size_bytes,
        }
