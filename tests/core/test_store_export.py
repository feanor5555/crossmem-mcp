"""Tests for KnowledgeStore.export and KnowledgeStore.import_data."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
import zipfile
from typing import TYPE_CHECKING

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import (
    Document,
    Metadata,
    generate_content_hash,
    generate_id,
)
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _make_store(db_path: Path) -> KnowledgeStore:
    backend = SQLiteBackend(db_path)
    return KnowledgeStore(backend, FixedEmbedder())


def _seed_three_docs(store: KnowledgeStore) -> list[str]:
    """Store three short docs and return the flat list of all chunk IDs."""
    ids: list[str] = []
    ids.extend(
        store.store(
            content="Alpha content about Python",
            source_url="https://example.com/a",
            title="Alpha",
            source_type="web",
            tags=["python"],
        )
    )
    ids.extend(
        store.store(
            content="Beta content about Rust",
            source_url="https://example.com/b",
            title="Beta",
            source_type="web",
            tags=["rust"],
        )
    )
    ids.extend(
        store.store(
            content="Gamma content about Go",
            source_url="https://example.com/c",
            title="Gamma",
            source_type="web",
            tags=["go", "concurrency"],
        )
    )
    return ids


# -------- iter_all on backend --------


def test_backend_iter_all_yields_all_docs(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "iter.db")
    store = KnowledgeStore(backend, FixedEmbedder())
    seeded_ids = _seed_three_docs(store)

    seen_ids = {d.id for d in backend.iter_all()}
    assert seen_ids == set(seeded_ids)

    # Each yielded doc has its embedding populated (length matches model dim)
    for doc in backend.iter_all():
        assert len(doc.embedding) == EMBEDDING_DIM
    backend.close()


def test_backend_iter_all_empty(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "empty.db")
    assert list(backend.iter_all()) == []
    backend.close()


# -------- export / import roundtrip (JSONL) --------


def _read_jsonl_lines(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_export_jsonl_roundtrip(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.jsonl"
    n = src.export(out, format="jsonl")
    assert n == 3
    assert out.exists()

    # File ends with EOF marker referencing count and a sha256 of content lines
    parsed = _read_jsonl_lines(out)
    assert len(parsed) == 4
    eof = parsed[-1]
    assert eof == {"type": "eof", "count": 3, "sha256": eof["sha256"]}
    assert isinstance(eof["sha256"], str) and len(eof["sha256"]) == 64

    # Fresh store
    dst = _make_store(tmp_path / "dst.db")
    imported = dst.import_data(out)
    assert imported == 3

    # Compare via iter_all on both sides — both go through SQLite's float[384]
    # storage, so the 32-bit-precision embeddings match exactly.
    all_dst = sorted(dst._backend.iter_all(), key=lambda d: d.id)
    all_src = sorted(src._backend.iter_all(), key=lambda d: d.id)
    assert [d.id for d in all_dst] == [d.id for d in all_src]
    for a, b in zip(all_dst, all_src, strict=True):
        assert a.content == b.content
        assert a.embedding == b.embedding
        assert a.metadata == b.metadata


def test_export_jsonl_atomic_no_partial_on_completion(tmp_path: Path) -> None:
    """Final file is written atomically; no .tmp left behind on success."""
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "atomic.jsonl"
    src.export(out, format="jsonl")

    assert out.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_export_jsonl_cleans_tmp_when_writer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If JSONL write fails mid-stream, no ``.tmp`` is left in the target dir."""
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    def boom(self, fh):  # type: ignore[no-untyped-def]
        raise RuntimeError("writer boom")

    monkeypatch.setattr(KnowledgeStore, "_stream_jsonl", boom)

    out = tmp_path / "atomic.jsonl"
    with pytest.raises(RuntimeError, match="writer boom"):
        src.export(out, format="jsonl")

    assert not out.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"stale tmp files left behind: {leftovers}"


def test_export_zip_cleans_tmp_when_writer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ZIP write fails mid-stream, no ``.tmp`` is left in the target dir."""
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    def boom(self, fh):  # type: ignore[no-untyped-def]
        raise RuntimeError("writer boom")

    monkeypatch.setattr(KnowledgeStore, "_stream_jsonl", boom)

    out = tmp_path / "atomic.zip"
    with pytest.raises(RuntimeError, match="writer boom"):
        src.export(out, format="zip")

    assert not out.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"stale tmp files left behind: {leftovers}"


# -------- export / import roundtrip (ZIP, default) --------


def test_export_zip_default_format(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.zip"
    n = src.export(out)  # default format
    assert n == 3
    assert out.exists()

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "documents.jsonl" in names


def test_export_zip_roundtrip(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.zip"
    src.export(out, format="zip")

    dst = _make_store(tmp_path / "dst.db")
    imported = dst.import_data(out)
    assert imported == 3

    all_dst = sorted(dst._backend.iter_all(), key=lambda d: d.id)
    all_src = sorted(src._backend.iter_all(), key=lambda d: d.id)
    for a, b in zip(all_dst, all_src, strict=True):
        assert a.content == b.content
        assert a.embedding == b.embedding
        assert a.metadata == b.metadata


# -------- error paths --------


def test_import_tampered_eof_raises(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.jsonl"
    src.export(out, format="jsonl")

    # Corrupt the EOF marker by replacing the sha256
    lines = out.read_text(encoding="utf-8").splitlines()
    eof = json.loads(lines[-1])
    eof["sha256"] = "0" * 64
    lines[-1] = json.dumps(eof)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="sha256"):
        dst.import_data(out)


def test_import_missing_eof_raises(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.jsonl"
    src.export(out, format="jsonl")

    # Drop the EOF marker
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="EOF"):
        dst.import_data(out)


def test_import_wrong_count_raises(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "dump.jsonl"
    src.export(out, format="jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    eof = json.loads(lines[-1])
    eof["count"] = 99
    lines[-1] = json.dumps(eof)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="count"):
        dst.import_data(out)


def test_import_wrong_embedding_dim_raises(tmp_path: Path) -> None:
    """A doc with the wrong embedding dimension is rejected.

    The integrity check (TODO 26.7) verifies ``content_hash`` and ``id``
    before the dim check, so this fixture builds both from the canonical
    helpers — only the embedding dimension is deliberately wrong.
    """
    content = "x"
    namespace = "default"
    source_url = "https://example.com/x"
    content_hash = generate_content_hash(content)
    doc_id = generate_id(namespace, source_url, content_hash)
    out = tmp_path / "bad.jsonl"
    bad_doc = {
        "id": doc_id,
        "content": content,
        "embedding": [0.1] * (EMBEDDING_DIM - 1),  # wrong dim
        "metadata": {
            "source_url": source_url,
            "title": "T",
            "source_type": "web",
            "stored_at": "2026-01-01T00:00:00+00:00",
            "embedding_model": "mock",
            "embedding_dim": EMBEDDING_DIM - 1,
            "namespace": namespace,
            "tags": [],
            "content_hash": content_hash,
        },
    }
    line = json.dumps(bad_doc, sort_keys=True)
    sha = hashlib.sha256((line + "\n").encode("utf-8")).hexdigest()
    eof = {"type": "eof", "count": 1, "sha256": sha}
    out.write_text(line + "\n" + json.dumps(eof) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="dim"):
        dst.import_data(out)


def test_import_unknown_format_raises(tmp_path: Path) -> None:
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"\x00\x01\x02")
    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError):
        dst.import_data(bad)


def test_export_unknown_format_raises(tmp_path: Path) -> None:
    src = _make_store(tmp_path / "src.db")
    with pytest.raises(ValueError):
        src.export(tmp_path / "x.bin", format="bogus")


# -------- streaming ZIP export (24.5) --------


class _StreamingBackend:
    """Test backend that yields ``n`` synthetic docs from ``iter_all``.

    Other backend methods raise — only ``iter_all`` is exercised by the
    export path. Documents are produced lazily by a generator so the
    backend itself never materialises the full corpus in RAM.
    """

    def __init__(self, n: int, payload_bytes: int) -> None:
        self._n = n
        # Pre-build one shared payload string and reuse it per doc so the
        # backend's own working set stays bounded regardless of ``n``. The
        # doc id varies per iteration so EOF sha256 sees ``n`` distinct
        # lines and is not falsely happy on a buggy implementation.
        self._content = "x" * payload_bytes
        self._embedding = tuple(0.0 for _ in range(EMBEDDING_DIM))

    def iter_all(self) -> Iterator[Document]:
        meta = Metadata(
            source_url="https://example.com/streamed",
            title="streamed",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=EMBEDDING_DIM,
            namespace="default",
            tags=("stream",),
            content_hash="h",
        )
        for i in range(self._n):
            yield Document(
                id=f"doc{i:06d}",
                content=self._content,
                embedding=self._embedding,
                metadata=meta,
            )

    # Stubs — the export path must never touch these.
    def store(self, doc: Document) -> None:  # pragma: no cover - guard
        raise AssertionError("export must not call store")

    def stats(self) -> dict:  # pragma: no cover - guard
        raise AssertionError("export must not call stats")


def test_export_zip_streams_payload_does_not_buffer_full_jsonl(
    tmp_path: Path,
) -> None:
    """ZIP export must not materialise the whole JSONL in RAM at once.

    Negative DoD for TODO 24.5: with a synthetic corpus whose total JSONL
    payload is well above ``payload_bytes * n``, peak Python allocation
    during ``export`` stays an order of magnitude below the full payload.
    A buggy ``"\\n".join(lines) + "\\n"`` implementation peaks at roughly
    the full payload size; the streaming implementation peaks at one
    line plus the ZIP deflate window.
    """
    n = 400
    payload_bytes = 8 * 1024  # 8 KiB per doc -> ~3.2 MiB total payload
    backend = _StreamingBackend(n=n, payload_bytes=payload_bytes)
    store = KnowledgeStore(backend, FixedEmbedder())  # type: ignore[arg-type]

    out = tmp_path / "streamed.zip"

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        count = store.export(out, format="zip")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert count == n

    full_payload_bytes = payload_bytes * n
    # Streaming peak must be well below the full uncompressed payload —
    # the buggy in-RAM-buffer path peaks at ~full_payload_bytes plus the
    # joined-string copy, so any threshold under ``full_payload_bytes``
    # rejects it. We pick half the payload as a generous ceiling that
    # still catches the regression even on platforms where tracemalloc
    # accounting is noisy.
    assert peak < full_payload_bytes // 2, (
        f"export peak {peak} >= half of full payload {full_payload_bytes}; "
        "ZIP export is still buffering the JSONL in RAM"
    )


# -------- atomic import (26.6) --------


class _HalfwayFailingBackend:
    """Test backend whose ``upsert_many`` raises after persisting half the docs.

    Mirrors the all-or-nothing contract from :meth:`VectorStoreBase.upsert_many`:
    a backend that cannot commit the whole batch MUST roll back the docs it
    already wrote. ``store`` is intentionally implemented as the legacy
    per-doc path so a regressed import (calling ``store`` in a loop instead of
    ``upsert_many``) leaks half the batch into ``persisted`` and trips the
    atomicity assertion.
    """

    def __init__(self) -> None:
        self.persisted: list[Document] = []
        self.upsert_calls = 0

    def store(self, doc: Document) -> None:
        # Legacy per-doc path. If import_data ever falls back to this method
        # the half-batch leak below will surface in ``persisted``.
        self.persisted.append(doc)

    def upsert_many(self, docs: list[Document]) -> None:
        self.upsert_calls += 1
        half = len(docs) // 2
        # Stage on a scratch list. We only extend ``self.persisted`` on a
        # successful full-batch commit, mirroring a real backend's
        # transactional rollback when the second half raises.
        staged: list[Document] = []
        for i, doc in enumerate(docs):
            if i == half:
                raise RuntimeError("simulated mid-batch crash")
            staged.append(doc)
        self.persisted.extend(staged)


def _build_valid_export(path: Path, n: int = 4) -> None:
    """Write a syntactically valid export with ``n`` docs to ``path``.

    Every doc passes the import integrity check (TODO 26.7): id and
    content_hash are derived from the canonical helpers, not hand-rolled
    placeholders.
    """
    lines: list[str] = []
    for i in range(n):
        content = f"content {i}"
        namespace = "default"
        source_url = f"https://example.com/{i}"
        content_hash = generate_content_hash(content)
        doc_id = generate_id(namespace, source_url, content_hash)
        doc = {
            "id": doc_id,
            "content": content,
            "embedding": [0.0] * EMBEDDING_DIM,
            "metadata": {
                "source_url": source_url,
                "title": f"Title {i}",
                "source_type": "web",
                "stored_at": "2026-01-01T00:00:00+00:00",
                "embedding_model": "mock",
                "embedding_dim": EMBEDDING_DIM,
                "namespace": namespace,
                "tags": [],
                "content_hash": content_hash,
            },
        }
        lines.append(json.dumps(doc, sort_keys=True))

    hasher = hashlib.sha256()
    for line in lines:
        hasher.update((line + "\n").encode("utf-8"))
    eof = {"type": "eof", "count": n, "sha256": hasher.hexdigest()}
    lines.append(json.dumps(eof, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_import_uses_upsert_many_for_atomicity(tmp_path: Path) -> None:
    """``import_data`` MUST commit every doc via a single ``upsert_many``.

    A regression that falls back to a per-doc ``store`` loop would leak half
    the batch into the backend when the second half raises — exactly the
    half-loaded DB state the DoD for TODO 26.6 forbids.
    """
    out = tmp_path / "valid.jsonl"
    _build_valid_export(out, n=4)

    backend = _HalfwayFailingBackend()
    store = KnowledgeStore(backend, FixedEmbedder())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="simulated mid-batch crash"):
        store.import_data(out)

    # All-or-nothing: the failed batch must have left zero docs behind.
    assert backend.persisted == [], (
        f"import_data leaked {len(backend.persisted)} docs after a "
        "mid-batch crash; the per-doc store fallback is back"
    )
    # And the import path must have taken the single-call upsert route.
    assert backend.upsert_calls == 1, (
        f"import_data did not call upsert_many exactly once "
        f"(got {backend.upsert_calls})"
    )


# -------- import integrity (26.7) --------


def _rebuild_eof_marker(content_lines: list[str]) -> str:
    """Return a JSON EOF marker line whose sha256 matches ``content_lines``.

    Mirrors ``KnowledgeStore._stream_jsonl`` so a tampered fixture can
    keep the EOF marker (count + sha256) consistent with the doctored
    payload — the only thing the integrity check (26.7) still has to
    catch.
    """
    hasher = hashlib.sha256()
    for line in content_lines:
        hasher.update((line + "\n").encode("utf-8"))
    eof = {"type": "eof", "count": len(content_lines), "sha256": hasher.hexdigest()}
    return json.dumps(eof, sort_keys=True)


def test_import_valid_export_passes_integrity_check(tmp_path: Path) -> None:
    """DoD 26.7 (a): a pristine export imports without raising.

    Exercises the canonical store -> export -> import roundtrip end-to-end
    against the new integrity check. Failure here would mean the verifier
    rejects ids / content_hashes that ``KnowledgeStore.store`` produced
    itself — i.e. the check is self-inconsistent with the writer.
    """
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "valid.jsonl"
    src.export(out, format="jsonl")

    dst = _make_store(tmp_path / "dst.db")
    assert dst.import_data(out) == 3


def test_import_tampered_content_fails_integrity_check(tmp_path: Path) -> None:
    """DoD 26.7 (b): rewritten content + matching EOF sha is still rejected.

    The attacker swaps a document's content but keeps the JSONL line's
    bytes consistent with the EOF marker (by recomputing the sha256).
    Without the per-doc integrity check the import would silently accept
    the doctored payload; with it, the content_hash re-derivation diverges
    and the import aborts.
    """
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "tampered_content.jsonl"
    src.export(out, format="jsonl")

    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    content_lines = lines[:-1]

    # Doctor the first document's content; leave its content_hash and id alone.
    record = json.loads(content_lines[0])
    record["content"] = record["content"] + " TAMPERED"
    content_lines[0] = json.dumps(record, sort_keys=True)
    # Re-seal the EOF marker so the existing sha256 check passes — the
    # integrity check has to be what catches the change.
    new_eof = _rebuild_eof_marker(content_lines)
    out.write_text("\n".join([*content_lines, new_eof]) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="failed integrity check"):
        dst.import_data(out)


def test_import_tampered_id_fails_integrity_check(tmp_path: Path) -> None:
    """DoD 26.7 (c): a flipped id with otherwise valid metadata is rejected.

    The attacker keeps content and metadata untouched but rewrites the
    ``id`` field (and rebuilds the EOF marker). The id re-derivation from
    ``(namespace, source_url, content_hash, chunk_index)`` then no longer
    matches the stored id and the import aborts.
    """
    src = _make_store(tmp_path / "src.db")
    _seed_three_docs(src)

    out = tmp_path / "tampered_id.jsonl"
    src.export(out, format="jsonl")

    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    content_lines = lines[:-1]

    record = json.loads(content_lines[0])
    # Flip a single hex nibble; the result is still a syntactically valid id
    # so the failure can only come from the integrity check.
    original_id = record["id"]
    flipped = ("1" if original_id[0] == "0" else "0") + original_id[1:]
    assert flipped != original_id
    record["id"] = flipped
    content_lines[0] = json.dumps(record, sort_keys=True)
    new_eof = _rebuild_eof_marker(content_lines)
    out.write_text("\n".join([*content_lines, new_eof]) + "\n", encoding="utf-8")

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="failed integrity check"):
        dst.import_data(out)


def test_export_zip_streamed_eof_marker_preserved(tmp_path: Path) -> None:
    """Streaming ZIP export still emits a valid EOF marker (count + sha256)."""
    n = 50
    backend = _StreamingBackend(n=n, payload_bytes=64)
    store = KnowledgeStore(backend, FixedEmbedder())  # type: ignore[arg-type]

    out = tmp_path / "eof.zip"
    count = store.export(out, format="zip")
    assert count == n

    with zipfile.ZipFile(out) as zf:
        assert "documents.jsonl" in zf.namelist()
        with zf.open("documents.jsonl") as fh:
            text = fh.read().decode("utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == n + 1  # n content lines + EOF marker

    eof = json.loads(lines[-1])
    assert eof["type"] == "eof"
    assert eof["count"] == n

    hasher = hashlib.sha256()
    for line in lines[:-1]:
        hasher.update((line + "\n").encode("utf-8"))
    assert eof["sha256"] == hasher.hexdigest()
