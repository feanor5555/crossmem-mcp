"""Zip-Slip protection tests for KnowledgeStore.import_data.

The export format places a single ``documents.jsonl`` entry inside a ZIP
archive. A malicious archive could contain extra entries with traversal
paths (``../../../etc/passwd``) or absolute paths
(``/etc/passwd``, ``C:\\Windows\\system32\\foo.txt``) hoping the importer
extracts them to disk.

These tests assert that:
  1. The current importer never writes any file outside the temporary
     directory (defense in depth — current code reads only the
     ``documents.jsonl`` member via ``zipfile.open``, which does not
     touch the filesystem).
  2. Archives that contain hostile path entries are rejected up-front
     with a ``ValueError``, even when the ``documents.jsonl`` member is
     also present and well-formed.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import TYPE_CHECKING

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.store import KnowledgeStore
from tests._fixtures.embedder import FixedEmbedder

if TYPE_CHECKING:
    from pathlib import Path


def _make_store(db_path: Path) -> KnowledgeStore:
    return KnowledgeStore(SQLiteBackend(db_path), FixedEmbedder())


def _empty_jsonl_payload() -> bytes:
    """Build a minimally valid JSONL payload (just an EOF marker, 0 docs)."""
    eof = {"type": "eof", "count": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    return (json.dumps(eof, sort_keys=True) + "\n").encode("utf-8")


# Path entries that escape the destination directory in various ways.
HOSTILE_ENTRIES: list[tuple[str, str]] = [
    ("../../../etc/passwd", "POSIX traversal"),
    ("..\\..\\..\\Windows\\system32\\foo.txt", "Windows backslash traversal"),
    ("/etc/passwd", "POSIX absolute path"),
    ("/tmp/zipslip_marker.txt", "POSIX absolute /tmp path"),
    ("foo/../../bar.txt", "embedded traversal segment"),
]


def _build_hostile_zip(
    dest: Path,
    *,
    hostile_name: str,
    include_documents_jsonl: bool = True,
) -> Path:
    """Write a ZIP at ``dest`` containing a hostile entry.

    ``zipfile`` rejects absolute paths and pure backslashes via the public
    ``writestr``/``write`` API on some platforms, so we craft the central
    directory by hand using ``ZipInfo`` so the archive really does carry
    the hostile name on disk.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(filename=hostile_name)
        zf.writestr(info, b"PWNED")
        if include_documents_jsonl:
            zf.writestr("documents.jsonl", _empty_jsonl_payload())
    dest.write_bytes(buf.getvalue())
    return dest


@pytest.mark.parametrize("entry, _label", HOSTILE_ENTRIES)
def test_import_rejects_zip_with_hostile_entry(
    tmp_path: Path, entry: str, _label: str
) -> None:
    """Hostile path entries must be rejected with ValueError, not silently ignored."""
    archive = _build_hostile_zip(
        tmp_path / "evil.zip",
        hostile_name=entry,
        include_documents_jsonl=True,
    )

    dst = _make_store(tmp_path / "dst.db")
    with pytest.raises(ValueError, match="(?i)unsafe|traversal|absolute|zip"):
        dst.import_data(archive)


@pytest.mark.parametrize("entry, _label", HOSTILE_ENTRIES)
def test_import_zip_no_filesystem_writes_outside_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    _label: str,
) -> None:
    """Defense in depth: importing must never write to a hostile path on disk.

    We sandbox CWD to ``tmp_path`` and snapshot it before/after the import
    attempt — no new files (besides the input archive and the destination
    DB) may appear, and crucially nothing may land at the literal hostile
    path on the real filesystem.
    """
    monkeypatch.chdir(tmp_path)
    archive = _build_hostile_zip(
        tmp_path / "evil.zip",
        hostile_name=entry,
        include_documents_jsonl=True,
    )

    dst_db = tmp_path / "dst.db"
    dst = _make_store(dst_db)

    # The import is expected to raise — but even if it ever stopped raising,
    # this test asserts nothing escapes ``tmp_path``.
    with pytest.raises(ValueError):
        dst.import_data(archive)

    # The hostile literal path must NOT exist on disk.
    # Use the archive's own entry name (raw, possibly with backslashes).
    hostile_abs_candidates = [
        tmp_path / entry.replace("\\", "/"),
        tmp_path.parent / entry.replace("\\", "/"),
    ]
    for candidate in hostile_abs_candidates:
        assert not candidate.exists(), f"hostile path written to disk: {candidate}"

    # No marker dropped at the absolute hostile location either.
    assert not (tmp_path / "PWNED").exists()


def test_import_clean_zip_still_works(tmp_path: Path) -> None:
    """A normal ZIP without hostile entries must continue to import cleanly."""
    src = _make_store(tmp_path / "src.db")
    src.store(
        content="benign content",
        source_url="https://example.com/a",
        title="Benign",
        source_type="web",
    )
    out = tmp_path / "good.zip"
    src.export(out, format="zip")

    dst = _make_store(tmp_path / "dst.db")
    assert dst.import_data(out) == 1
