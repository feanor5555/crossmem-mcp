"""KnowledgeStore facade over a vector backend and an embedder."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from crossmem.core.chunking import chunk as chunk_content
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import Document, Vec, generate_content_hash, generate_id
from crossmem.core.tagging import extract_tags

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossmem.backends.base import BackendStats, VectorStoreBase

_EXPORT_JSONL_NAME = "documents.jsonl"

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    """Embedder contract consumed by :class:`KnowledgeStore`.

    All ``embed_*`` methods return :data:`crossmem.core.models.Vec` —
    fp16-precision values transported as ``list[float]``. The production
    implementation in :mod:`crossmem.core.embedding` rounds every vector
    to ``np.float16`` before it leaves the embedder; the SQLite backend
    widens it back to fp32 for ``vec0`` storage without recovering the
    lost bits. Custom embedders plugged in via this Protocol SHOULD make
    the same precision contract explicit (either by rounding themselves
    or by documenting that they emit higher-precision values that the
    store will not preserve).
    """

    @property
    def model_name(self) -> str: ...
    def embed_query(self, text: str) -> Vec: ...
    def embed_passage(self, text: str) -> Vec: ...
    def embed_passage_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[Vec]: ...


class KnowledgeStore:
    """Facade combining an embedder and a vector store backend.

    Pure dependency injection: both the backend and the embedder are
    supplied by the caller. No I/O happens in the constructor.
    """

    def __init__(
        self,
        backend: VectorStoreBase,
        embedder: _Embedder,
        trash_path: Path | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._trash_path = trash_path

    def _resolve_trash_path(self) -> Path:
        """Return the configured trash path or the default ~/.crossmem location."""
        if self._trash_path is not None:
            return self._trash_path
        return Path.home() / ".crossmem" / ".crossmem-trash.jsonl"

    def store(
        self,
        content: str,
        source_url: str,
        title: str,
        source_type: str,
        namespace: str = "default",
        tags: list[str] | None = None,
    ) -> list[str]:
        """Chunk, auto-tag, embed and persist ``content``.

        The raw input is split via :func:`crossmem.core.chunking.chunk` using
        the strategy implied by ``source_type``. Each chunk becomes its own
        :class:`Document` whose tags are the union of caller-supplied ``tags``
        and auto-tags derived from ``source_url`` / ``title`` / chunk content
        via :func:`crossmem.core.tagging.extract_tags`.

        Returns the list of persisted document IDs in chunk order. An empty
        ``content`` (or one that produces no chunks) yields an empty list.

        All chunk Documents are handed to ``backend.upsert_many`` as a
        single atomic batch — a crash between chunks rolls back the whole
        document rather than leaving it half-stored.
        """
        chunks = chunk_content(content, source_type, title=title)
        if not chunks:
            return []
        caller_tags = list(tags) if tags else []
        # Batch every chunk through one ``embed_passage_batch`` call so the
        # fastembed model is invoked at most once per document (~1.5 ms per
        # chunk amortised vs ~50 ms when called one-by-one). The batch path
        # is itself cache-aware, so re-storing identical content still hits
        # the LRU instead of the model.
        embeddings = self._embedder.embed_passage_batch([ch.content for ch in chunks])
        model_name = self._embedder.model_name
        docs: list[Document] = []
        for ch, embedding in zip(chunks, embeddings, strict=True):
            auto_tags = extract_tags(source_url, title, ch.content)
            merged_tags = sorted(set(caller_tags) | set(auto_tags))
            docs.append(
                Document.from_payload(
                    content=ch.content,
                    source_url=source_url,
                    title=title,
                    source_type=source_type,
                    namespace=namespace,
                    tags=merged_tags,
                    embedding=embedding,
                    embedding_model=model_name,
                    embedding_dim=len(embedding),
                    # Mix the chunk position into the doc ID so two chunks
                    # with identical content (boilerplate / high overlap)
                    # remain individually addressable.
                    chunk_index=ch.chunk_index,
                )
            )
        # One atomic write per logical document: backends override
        # ``upsert_many`` so a crash mid-batch rolls back every chunk
        # instead of leaving a half-stored document behind.
        self._backend.upsert_many(docs)
        return [doc.id for doc in docs]

    def restore(self, doc: Document) -> None:
        """Re-insert an already-embedded Document (e.g. trash restore), no re-embed."""
        self._backend.store(doc)

    def query(
        self,
        query: str,
        top_k: int = 10,
        tags: list[str] | None = None,
    ) -> list[Document]:
        """Hybrid search via Reciprocal Rank Fusion (RRF) of FTS and vector hits.

        FTS receives ``tags`` as a pre-filter. Vector hits are post-filtered
        to those whose metadata tags intersect ``tags``. Results from both
        lists are merged with the standard RRF formula (k=60) and the top
        ``top_k`` documents are returned in descending score order.

        Vector backends today cannot pre-filter by tag (sqlite-vec has no
        WHERE-clause on the KNN MATCH, Chroma/Qdrant filters live behind
        backend-specific APIs we don't surface). A naive ``top_k * 3`` fetch
        therefore loses on-tag hits whenever off-tag docs dominate the top
        of the rank. We compensate by widening the vector window when
        ``tags`` is set: keep doubling the fetched window until enough
        on-tag hits have surfaced, the backend returns fewer rows than
        requested (dataset exhausted), or a hard upper bound is reached.
        """
        rrf_k = 60
        fetch = top_k * 3

        fts_results = self._backend.query_fts(query, fetch, tags=tags)
        embedding = self._embedder.embed_query(query)
        vec_results = self._fetch_vector_candidates(embedding, fetch, tags)

        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranked_list in (fts_results, vec_results):
            for rank, doc in enumerate(ranked_list):
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank + 1)
                docs.setdefault(doc.id, doc)

        ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in ordered_ids[:top_k]]

    # Hard cap on the vector over-fetch loop. ``vec0`` happily accepts very
    # large ``k`` values, but we never want a runaway query — 10 doublings
    # from ``top_k * 3`` cover datasets up to ~3k * top_k candidates, which
    # is well beyond any realistic top_k where the tag pre-filter matters.
    _VECTOR_TAG_OVERFETCH_MAX_ITERS = 10

    def _fetch_vector_candidates(
        self,
        embedding: Vec,
        fetch: int,
        tags: list[str] | None,
    ) -> list[Document]:
        """Return vector hits, widening the fetch window when ``tags`` filters.

        When ``tags`` is falsy the original ``fetch`` window is used and the
        result is returned unfiltered — preserves the no-tag behaviour
        exactly. With tags, the window doubles each iteration until at
        least ``fetch`` post-filtered hits exist, the backend returns fewer
        rows than requested (dataset exhausted), or the iteration cap is
        hit.
        """
        if not tags:
            return self._backend.query_vector(embedding, fetch)

        tag_set = set(tags)
        k = fetch
        for _ in range(self._VECTOR_TAG_OVERFETCH_MAX_ITERS):
            raw = self._backend.query_vector(embedding, k)
            filtered = [d for d in raw if tag_set.intersection(d.metadata.tags)]
            backend_exhausted = len(raw) < k
            if len(filtered) >= fetch or backend_exhausted:
                return filtered
            k *= 2
        return filtered

    def delete(
        self,
        doc_id: str | None = None,
        source_url: str | None = None,
        permanent: bool = False,
    ) -> int:
        """Delete by id OR by source_url. Returns number of docs deleted.

        Soft-delete writes each doc to the trash JSONL before
        ``backend.delete()``; ``permanent=True`` bypasses the trash. Exactly
        one of ``doc_id`` / ``source_url`` must be supplied. Both soft- and
        permanent-delete by ``doc_id`` resolve the document via
        ``backend.get_by_id`` first and are no-ops (return ``0``) when the
        id is unknown — the returned count therefore reflects actual hits,
        not delete attempts.
        """
        if (doc_id is None) == (source_url is None):
            raise ValueError("delete() requires exactly one of doc_id or source_url")

        if permanent:
            if doc_id is not None:
                if self._backend.get_by_id(doc_id) is None:
                    return 0
                self._backend.delete(doc_id)
                return 1
            docs = self._backend.get_by_url(source_url)  # type: ignore[arg-type]
            logger.info(
                "delete: matched %d document(s) for source_url=%s (permanent)",
                len(docs),
                source_url,
            )
            for doc in docs:
                self._backend.delete(doc.id)
            return len(docs)

        # Soft-delete path — needs the full document(s) for the trash record.
        if doc_id is not None:
            doc = self._backend.get_by_id(doc_id)
            if doc is None:
                return 0
            self._append_to_trash([doc])
            self._backend.delete(doc.id)
            return 1

        docs = self._backend.get_by_url(source_url)  # type: ignore[arg-type]
        logger.info(
            "delete: matched %d document(s) for source_url=%s",
            len(docs),
            source_url,
        )
        if not docs:
            return 0

        self._append_to_trash(docs)
        for doc in docs:
            self._backend.delete(doc.id)
        return len(docs)

    def find_by_tag(self, tag: str):  # type: ignore[no-untyped-def]
        """Yield every document whose metadata tags contain ``tag``.

        Thin pass-through to ``backend.find_by_tag`` so cleanup (and any
        future consumer) reaches the native index-backed lookup without
        touching the private ``_backend`` attribute. The return value is
        an iterator — callers MUST iterate it (or wrap in ``list(...)``)
        to materialise the hits; nothing is fetched eagerly here.
        """
        return self._backend.find_by_tag(tag)

    def stats(self) -> BackendStats:
        """Return backend statistics (doc count, db size, top tags, backend info).

        Thin pass-through to ``backend.stats()`` so callers (e.g. the MCP
        server) do not need to reach into the private ``_backend`` attribute.
        The returned :class:`BackendStats` carries the unified key contract
        enforced by :class:`crossmem.backends.base.VectorStoreBase`.
        """
        return self._backend.stats()

    def _append_to_trash(self, docs: list[Document]) -> None:
        """Append one JSONL record per doc to the trash file (atomic per line)."""
        trash_path = self._resolve_trash_path()
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        deleted_at = datetime.now(timezone.utc).isoformat()
        with open(trash_path, "a", encoding="utf-8") as fh:
            for doc in docs:
                record = {"deleted_at": deleted_at, "doc": doc.to_dict()}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export(self, path: Path, format: str = "zip") -> int:  # noqa: A002 — MCP tool param name
        """Export every document to ``path`` as JSONL or ZIP.

        Each document is serialized as one JSON line; a final EOF marker
        records the count and a sha256 of all content lines (with trailing
        ``\\n``) so importers can detect truncation or tampering. The file is
        written atomically: contents go to ``<path>.tmp`` (or ``<path>.tmp.zip``
        for ZIP) and only at the end ``Path.replace()`` swaps it into place.

        Returns the number of documents written.
        """
        if format not in ("jsonl", "zip"):
            raise ValueError(f"unsupported export format: {format!r}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            return self._atomic_export(path, self._write_jsonl)

        # ZIP path: stream JSONL into a ZIP entry atomically.
        #
        # ``zf.open(name, "w")`` returns a write handle backed by the ZIP
        # archive's deflate stream — each ``write`` call compresses and
        # appends to the archive on disk without ever materialising the
        # full JSONL payload in RAM. Compared to the previous
        # ``writestr(buf)`` path (which built ``"\n".join(lines) + "\n"``
        # over every document first), peak memory for a million-doc
        # export drops from O(N * line_size) to O(one_line + deflate_window).
        # ``force_zip64=True`` keeps the entry valid even when the
        # uncompressed payload crosses the 2 GiB classic-ZIP limit.
        return self._atomic_export(path, self._write_zip)

    def _atomic_export(self, path: Path, writer: Callable[[Path], int]) -> int:
        """Run ``writer`` against ``<path>.tmp`` and atomically swap on success.

        ``writer`` receives the tmp ``Path`` and returns the number of
        documents written. On any exception the tmp file is unlinked in
        ``finally`` so neither the JSONL nor the ZIP export path can leave
        a stale ``.tmp`` artefact behind. The successful path likewise
        finds the tmp gone after ``Path.replace()`` swapped it into place.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            count = writer(tmp)
            tmp.replace(path)
            return count
        finally:
            if tmp.exists():
                tmp.unlink()

    def _stream_jsonl(self, fh) -> int:  # type: ignore[no-untyped-def]
        """Write JSONL content lines + EOF marker to a binary file-like ``fh``.

        Used for both the JSONL-file and ZIP-entry export paths so the
        hashing + EOF protocol stays in one place. ``fh`` MUST accept
        ``bytes`` writes — text-mode handles are wrapped by the caller.
        """
        hasher = hashlib.sha256()
        count = 0
        for doc in self._backend.iter_all():
            line = json.dumps(doc.to_dict(), sort_keys=True, ensure_ascii=False)
            encoded = (line + "\n").encode("utf-8")
            hasher.update(encoded)
            fh.write(encoded)
            count += 1
        eof = {"type": "eof", "count": count, "sha256": hasher.hexdigest()}
        fh.write((json.dumps(eof, sort_keys=True) + "\n").encode("utf-8"))
        return count

    def _write_jsonl(self, path: Path) -> int:
        """Stream every document to ``path`` as JSONL with an EOF marker.

        Opens the file in binary mode and reuses :meth:`_stream_jsonl` so
        the JSONL and ZIP export paths share the exact same hashing +
        EOF protocol — divergence between them would silently break
        importers.
        """
        with open(path, "wb") as fh:
            return self._stream_jsonl(fh)

    def _write_zip(self, path: Path) -> int:
        """Stream JSONL into a ZIP entry at ``path`` and return the doc count.

        Mirrors :meth:`_write_jsonl` so :meth:`_atomic_export` can treat
        both formats uniformly. The ZIP archive itself is opened on
        ``path`` directly — :meth:`_atomic_export` owns the tmp / swap /
        cleanup contract.
        """
        with (
            zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf,
            zf.open(_EXPORT_JSONL_NAME, "w", force_zip64=True) as fh,
        ):
            return self._stream_jsonl(fh)

    def import_data(self, path: Path) -> int:
        """Import a previously exported JSONL or ZIP file.

        Validates the EOF marker (count + sha256 of all content lines) and
        the embedding dimension of every document. Mismatches raise
        ``ValueError`` and abort the import without storing any document.

        Returns the number of documents imported.
        """
        path = Path(path)
        text = _read_export_text(path)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("import file is empty (no EOF marker)")

        try:
            eof = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid EOF marker: {exc}") from exc
        if not isinstance(eof, dict) or eof.get("type") != "eof":
            raise ValueError("missing EOF marker (last line must be type=eof)")

        content_lines = lines[:-1]

        hasher = hashlib.sha256()
        for line in content_lines:
            hasher.update((line + "\n").encode("utf-8"))
        if hasher.hexdigest() != eof.get("sha256"):
            raise ValueError("EOF sha256 does not match content lines")
        if eof.get("count") != len(content_lines):
            raise ValueError(
                f"EOF count mismatch: marker={eof.get('count')} "
                f"actual={len(content_lines)}"
            )

        docs: list[Document] = []
        for raw in content_lines:
            doc = Document.from_dict(json.loads(raw))
            _verify_document_integrity(doc)
            if len(doc.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"embedding dim mismatch: got {len(doc.embedding)}, "
                    f"expected {EMBEDDING_DIM}"
                )
            docs.append(doc)

        # One atomic batch instead of a per-doc loop: ``upsert_many`` is the
        # contracted all-or-nothing entry point, so a backend crash partway
        # through the import rolls back every doc rather than leaving a
        # half-loaded DB behind.
        self._backend.upsert_many(docs)
        return len(docs)


# Hard cap on the chunk-index brute force inside the integrity check.
# ``Document.from_payload`` mixes ``chunk_index`` into the id hash so
# duplicate-content chunks within the same document keep distinct ids,
# but the index itself is not persisted alongside the document. The
# import-time verifier therefore re-derives the id with ``chunk_index``
# = ``None`` first (source adapters) and then sweeps ``0 .. LIMIT``
# (chunking pipeline). The cap covers any realistic document layout —
# chunk splits run at the 512-token boundary, so 1024 is roughly a
# 1 MiB worst-case payload — and keeps the per-doc verification cost
# bounded to ~1024 SHA-256 hashes (~ms range).
_INTEGRITY_CHUNK_INDEX_LIMIT = 1024


def _verify_document_integrity(doc: Document) -> None:
    """Raise ``ValueError`` if ``doc`` does not hash to its stored id.

    Re-derives ``content_hash`` from the current content and ``id`` from
    the metadata triple (namespace, source_url, content_hash) plus
    ``chunk_index``. Either re-derivation diverging from the stored
    value means the export was tampered with after the SHA-256 EOF
    marker was computed (or, equivalently, the marker was rewritten to
    match) — both checks together close the gap left by the marker-only
    test.

    ``chunk_index`` is not part of the persisted metadata (it lives in
    the in-memory ``Chunk`` only), so the id check first tries the
    unindexed form used by source adapters and then sweeps
    ``0 .. _INTEGRITY_CHUNK_INDEX_LIMIT`` for the chunking-pipeline
    case. The cap keeps a hostile file from forcing an unbounded search.

    The error message intentionally pins the offending ``doc.id`` so
    operators can locate the bad record in the export without having
    to diff the whole file.
    """
    expected_hash = generate_content_hash(doc.content)
    if expected_hash != doc.metadata.content_hash:
        raise ValueError(f"import file: document {doc.id} failed integrity check")
    namespace = doc.metadata.namespace
    source_url = doc.metadata.source_url
    content_hash = doc.metadata.content_hash
    if generate_id(namespace, source_url, content_hash) == doc.id:
        return
    for chunk_index in range(_INTEGRITY_CHUNK_INDEX_LIMIT):
        if generate_id(namespace, source_url, content_hash, chunk_index) == doc.id:
            return
    raise ValueError(f"import file: document {doc.id} failed integrity check")


def _read_export_text(path: Path) -> str:
    """Read the JSONL payload from a JSONL file or a ZIP archive.

    For ZIP inputs every member name is validated up-front against
    classic Zip-Slip vectors (``..`` traversal, absolute POSIX paths,
    Windows backslash paths and drive-letter prefixes). Hostile names
    raise ``ValueError`` even when the legitimate ``documents.jsonl``
    member is present, so a tampered export cannot smuggle additional
    files past the importer.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                _ensure_safe_zip_member(name)
            if _EXPORT_JSONL_NAME not in zf.namelist():
                raise ValueError(f"ZIP export is missing {_EXPORT_JSONL_NAME!r} entry")
            with zf.open(_EXPORT_JSONL_NAME) as fh:
                return fh.read().decode("utf-8")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"unsupported import file: {path.name}") from exc


def _ensure_safe_zip_member(name: str) -> None:
    """Reject ZIP entry names that look like Zip-Slip payloads.

    A legitimate export contains only ``documents.jsonl`` at the archive
    root. Anything else — traversal segments, absolute paths, Windows
    drive-letter prefixes, or pure backslash separators — is treated as
    a tampered export and rejected with ``ValueError``.
    """
    # Normalise backslashes so Windows-style paths are caught even on POSIX.
    normalised = name.replace("\\", "/")
    if normalised.startswith("/"):
        raise ValueError(f"unsafe absolute path in ZIP entry: {name!r}")
    if len(normalised) >= 2 and normalised[1] == ":":
        raise ValueError(f"unsafe absolute (drive-letter) ZIP entry: {name!r}")
    parts = normalised.split("/")
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe traversal in ZIP entry: {name!r}")
