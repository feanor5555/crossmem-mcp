"""Cleanup, trash and restore for the CrossMem knowledge database.

This module exposes a small workflow on top of :class:`KnowledgeStore`:

* :func:`cleanup` finds candidate documents (by exact tag or by semantic
  similarity), previews them when ``dry_run`` is true, or moves them to
  the trash and removes them from the store when ``dry_run`` is false.
* :func:`empty_trash` drops trash entries older than ``ttl_days`` and
  rewrites the trash file atomically. ``ttl_days=0`` purges everything
  (the DSGVO escape hatch documented in CLAUDE.md).
* :func:`restore_from_trash` re-inserts a previously trashed document
  back into the store and removes its line from the trash file.

The trash file is JSONL — one record per line — at the path described
in CLAUDE.md: ``~/.crossmem/.crossmem-trash.jsonl`` by default. Each
record is::

    {"deleted_at": "<ISO8601 UTC>", "doc": {<full Document.to_dict()>}}
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from crossmem.core.models import Document

if TYPE_CHECKING:
    from crossmem.core.store import KnowledgeStore


# Top_k used for semantic-mode cleanup unless overridden by the caller.
_SEMANTIC_MODE_TOP_K = 50


@dataclass(frozen=True)
class TrashEntry:
    """A single entry in the trash JSONL file.

    Exposes only the fields callers (CLI ``trash list``, MCP clients)
    need for display: the trashed document's id, when it was trashed,
    its original source URL and title. The full document payload stays
    in the file so :func:`restore_from_trash` can rehydrate it later
    without re-reading every line.
    """

    doc_id: str
    deleted_at: str
    source_url: str
    title: str


@dataclass
class CleanupResult:
    """Outcome of a single :func:`cleanup` call.

    ``previewed_ids`` is always populated with every match, regardless
    of ``dry_run``. ``deleted_ids`` is populated only when ``dry_run``
    is false. ``mode`` echoes the requested mode for the caller.
    """

    mode: str
    previewed_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)


def _default_trash_path() -> Path:
    return Path.home() / ".crossmem" / ".crossmem-trash.jsonl"


def _resolve_trash_path(trash_path: Path | None) -> Path:
    return trash_path if trash_path is not None else _default_trash_path()


def _append_records(trash_path: Path, docs: list[Document]) -> None:
    """Append one JSONL record per doc to the trash file."""
    trash_path.parent.mkdir(parents=True, exist_ok=True)
    deleted_at = datetime.now(timezone.utc).isoformat()
    with open(trash_path, "a", encoding="utf-8") as fh:
        for doc in docs:
            record = {"deleted_at": deleted_at, "doc": doc.to_dict()}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _secure_mode(trash_path)


def _read_trash_lines(trash_path: Path) -> list[str]:
    if not trash_path.exists():
        return []
    text = trash_path.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip()]


def _atomic_rewrite(trash_path: Path, lines: list[str]) -> None:
    """Atomically rewrite the trash file with ``lines`` (no trailing blank)."""
    trash_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = trash_path.with_suffix(trash_path.suffix + ".tmp")
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(trash_path)
    _secure_mode(trash_path)


def _secure_mode(path: Path) -> None:
    """Restrict ``path`` to owner read/write (0o600) on Linux/Mac.

    The trash file contains the full content of soft-deleted documents,
    which may include private notes or copies of cached web pages. On
    POSIX systems clamp the permissions to ``0o600`` so a shared-host
    neighbor cannot read them via the default ``0o644`` umask outcome.
    On Windows POSIX modes are a no-op (cross-platform rule #4 in
    CLAUDE.md).
    """
    if sys.platform != "win32":
        os.chmod(path, 0o600)


def _find_by_tag(store: KnowledgeStore, tag: str) -> list[Document]:
    """Return every doc whose metadata tags contain ``tag`` exactly.

    Delegates to :meth:`KnowledgeStore.find_by_tag`, which routes through
    each backend's native tag-index path (TODO 26.1). No RRF / over-fetch
    is involved, so docs that carry the tag purely in metadata (with no
    content match) are still returned.
    """
    return list(store.find_by_tag(tag))


def _find_by_semantic(store: KnowledgeStore, query: str, top_k: int) -> list[Document]:
    return store.query(query=query, top_k=top_k)


def cleanup(
    store: KnowledgeStore,
    query: str,
    *,
    dry_run: bool = True,
    mode: str = "tag",
    top_k: int = _SEMANTIC_MODE_TOP_K,
    trash_path: Path | None = None,
) -> CleanupResult:
    """Preview or delete documents matching ``query``.

    Parameters
    ----------
    store:
        The :class:`KnowledgeStore` to operate on.
    query:
        For ``mode="tag"``: an exact tag string to match.
        For ``mode="semantic"``: a free-text query for hybrid search.
    dry_run:
        When true (default), nothing is deleted and the result only
        contains ``previewed_ids``.
    mode:
        ``"tag"`` (default) or ``"semantic"``.
    top_k:
        Top-K for semantic mode. Ignored for tag mode.
    trash_path:
        Override the default trash file location.
    """
    if mode == "tag":
        matches = _find_by_tag(store, query)
    elif mode == "semantic":
        matches = _find_by_semantic(store, query, top_k)
    else:
        raise ValueError(f"unknown cleanup mode: {mode!r}")

    matched_ids = [doc.id for doc in matches]
    result = CleanupResult(mode=mode, previewed_ids=list(matched_ids))

    if dry_run or not matches:
        return result

    target = _resolve_trash_path(trash_path)
    _append_records(target, matches)
    for doc in matches:
        store.delete(doc_id=doc.id, permanent=True)
    result.deleted_ids = list(matched_ids)
    return result


def empty_trash(
    trash_path: Path | None = None,
    *,
    ttl_days: int = 30,
    now: datetime | None = None,
) -> int:
    """Drop trash entries older than ``ttl_days`` and rewrite the file.

    Returns the number of entries removed. ``ttl_days=0`` removes every
    entry regardless of timestamp (DSGVO behavior). If the trash file
    does not exist, returns 0 without touching the filesystem.
    """
    target = _resolve_trash_path(trash_path)
    if not target.exists():
        return 0

    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ttl_days)

    lines = _read_trash_lines(target)
    if ttl_days == 0:
        _atomic_rewrite(target, [])
        return len(lines)

    keep: list[str] = []
    removed = 0
    for line in lines:
        try:
            record = json.loads(line)
            deleted_at = datetime.fromisoformat(record["deleted_at"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Corrupt line — keep it; manual inspection required.
            keep.append(line)
            continue
        if deleted_at < cutoff:
            removed += 1
        else:
            keep.append(line)

    _atomic_rewrite(target, keep)
    return removed


def list_trash(trash_path: Path | None = None) -> list[TrashEntry]:
    """Return one :class:`TrashEntry` per readable line in the trash file.

    Missing file -> empty list. Corrupt or incomplete records are
    silently skipped so a single bad line never hides the rest. Entries
    are returned in file order (oldest deletion first when the file was
    appended chronologically).
    """
    target = _resolve_trash_path(trash_path)
    if not target.exists():
        return []

    entries: list[TrashEntry] = []
    for line in _read_trash_lines(target):
        try:
            record = json.loads(line)
            doc = record["doc"]
            doc_id = doc["id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        meta = doc.get("metadata") or {}
        entries.append(
            TrashEntry(
                doc_id=doc_id,
                deleted_at=str(record.get("deleted_at", "")),
                source_url=str(meta.get("source_url", "")),
                title=str(meta.get("title", "")),
            )
        )
    return entries


def restore_from_trash(
    store: KnowledgeStore,
    doc_id: str,
    trash_path: Path | None = None,
) -> Document:
    """Re-insert ``doc_id`` from the trash into ``store`` and drop its line.

    Raises ``ValueError`` if the doc_id is not present in the trash.
    """
    target = _resolve_trash_path(trash_path)
    lines = _read_trash_lines(target)

    matched_doc: Document | None = None
    keep: list[str] = []
    for line in lines:
        try:
            record = json.loads(line)
            payload = record["doc"]
        except (json.JSONDecodeError, KeyError, TypeError):
            keep.append(line)
            continue
        if matched_doc is None and payload.get("id") == doc_id:
            matched_doc = Document.from_dict(payload)
            continue
        keep.append(line)

    if matched_doc is None:
        raise ValueError(f"doc_id not found in trash: {doc_id!r}")

    store.restore(matched_doc)
    _atomic_rewrite(target, keep)
    return matched_doc
