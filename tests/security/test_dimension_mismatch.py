"""Embedding-dimension validation on import.

A trusted exporter always emits 384-dimension vectors (matching the
pinned model). A tampered or cross-model export carries vectors of the
wrong dimension; storing them would either crash sqlite-vec or silently
corrupt the index. ``KnowledgeStore.import_data`` must therefore reject
any document whose ``embedding`` length does not equal ``EMBEDDING_DIM``,
*before* any document is written to the backend.

These tests complement the existing happy-path coverage in
``tests/core/test_store_export.py`` by exercising the boundary cases
(off-by-one short, off-by-one long, way too short, empty, way too long)
and by asserting the all-or-nothing transactional guarantee.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import generate_content_hash, generate_id
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from pathlib import Path


def _make_store(db_path: Path) -> KnowledgeStore:
    return KnowledgeStore(SQLiteBackend(db_path), FixedEmbedder())


def _write_jsonl(
    path: Path, docs: list[dict], *, sha_override: str | None = None
) -> None:
    """Write ``docs`` as JSONL with a valid (or overridden) EOF marker."""
    hasher = hashlib.sha256()
    lines: list[str] = []
    for d in docs:
        line = json.dumps(d, sort_keys=True)
        hasher.update((line + "\n").encode("utf-8"))
        lines.append(line)
    sha = sha_override if sha_override is not None else hasher.hexdigest()
    eof = {"type": "eof", "count": len(docs), "sha256": sha}
    lines.append(json.dumps(eof, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _doc_with_dim(
    dim: int,
    *,
    source_url: str = "https://example.com/x",
    content: str = "x",
) -> dict:
    """Build a serialized document whose embedding has ``dim`` components.

    ``id`` and ``content_hash`` are derived from the canonical helpers so
    the doc passes the import integrity check (TODO 26.7) and the dim
    failure is the only thing left for ``import_data`` to reject.
    """
    namespace = "default"
    content_hash = generate_content_hash(content)
    doc_id = generate_id(namespace, source_url, content_hash)
    return {
        "id": doc_id,
        "content": content,
        "embedding": [0.1] * dim,
        "metadata": {
            "source_url": source_url,
            "title": "T",
            "source_type": "web",
            "stored_at": "2026-01-01T00:00:00+00:00",
            "embedding_model": "mock",
            "embedding_dim": dim,
            "namespace": namespace,
            "tags": [],
            "content_hash": content_hash,
        },
    }


# ---------------------------------------------------------------------------
# Wrong-dimension payloads
# ---------------------------------------------------------------------------

WRONG_DIMS = [
    EMBEDDING_DIM - 1,  # off-by-one short
    EMBEDDING_DIM + 1,  # off-by-one long
    EMBEDDING_DIM // 2,  # half (e.g. 192)
    EMBEDDING_DIM * 2,  # double (e.g. 768)
    256,  # common BERT-base / cross-model mismatch
    768,  # full BERT-base
    1,  # absurdly short
    0,  # empty embedding
]


@pytest.mark.parametrize("dim", WRONG_DIMS)
def test_import_rejects_wrong_dimension(tmp_path: Path, dim: int) -> None:
    """Any embedding-length other than EMBEDDING_DIM raises ValueError."""
    out = tmp_path / f"bad_{dim}.jsonl"
    _write_jsonl(out, [_doc_with_dim(dim)])

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)dim"):
        dst.import_data(out)


def test_import_wrong_dim_writes_no_documents(tmp_path: Path) -> None:
    """Even if some docs are well-formed, a single bad-dim doc aborts the
    whole import — no documents are partially committed."""
    out = tmp_path / "mixed.jsonl"
    _write_jsonl(
        out,
        [
            _doc_with_dim(
                EMBEDDING_DIM, source_url="https://example.com/ok-1", content="ok-1"
            ),
            _doc_with_dim(256, source_url="https://example.com/bad", content="bad"),
            _doc_with_dim(
                EMBEDDING_DIM, source_url="https://example.com/ok-2", content="ok-2"
            ),
        ],
    )

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)dim"):
        dst.import_data(out)

    # No partial state — backend remains empty.
    assert dst.stats()["document_count"] == 0


def test_import_missing_eof_marker(tmp_path: Path) -> None:
    """Missing EOF marker is rejected as tampering — not silently ignored."""
    out = tmp_path / "no_eof.jsonl"
    line = json.dumps(_doc_with_dim(EMBEDDING_DIM), sort_keys=True)
    # No EOF marker line at all.
    out.write_text(line + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)eof"):
        dst.import_data(out)


def test_import_corrupted_eof_sha(tmp_path: Path) -> None:
    """Tampered sha256 in the EOF marker is rejected before any doc lands."""
    out = tmp_path / "bad_sha.jsonl"
    _write_jsonl(out, [_doc_with_dim(EMBEDDING_DIM)], sha_override="0" * 64)

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)sha256"):
        dst.import_data(out)
    assert dst.stats()["document_count"] == 0


def test_import_empty_file_rejected(tmp_path: Path) -> None:
    """Empty file has no EOF marker -> rejected."""
    out = tmp_path / "empty.jsonl"
    out.write_text("", encoding="utf-8")
    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)empty|eof"):
        dst.import_data(out)
