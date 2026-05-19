"""Sensitive-file permissions hardening (chmod 600 on Linux/Mac).

CrossMem writes three on-disk artifacts that may contain secrets or
private data:

* ``~/.crossmem/config.toml`` — may include an ``api_key`` for the active
  remote backend (Chroma/Qdrant).
* ``~/.crossmem/.crossmem-trash.jsonl`` — soft-deleted documents with
  full content prior to TTL purge.
* ``~/.crossmem/knowledge.db`` — the SQLite knowledge base.

On Linux/Mac these files MUST be ``0o600`` (owner read/write only) so a
user with shell access on a shared host cannot read them via the
default ``0o644``/``0o664`` umask outcome. On Windows file modes are a
no-op — the tests assert that we don't crash and don't try to set a
mode there (cross-platform rule #4 in CLAUDE.md).

The fourth check belongs to ``crossmem doctor``: warn the user when an
existing file on disk has looser permissions than ``0o600`` so they can
remediate manually.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from crossmem import cleanup, configure, doctor
from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata

_IS_WINDOWS = sys.platform == "win32"
posix_only = pytest.mark.skipif(
    _IS_WINDOWS,
    reason="POSIX file modes — no-op on Windows (chmod 600 only on Linux/Mac)",
)


def _mode(path: Path) -> int:
    """Return the POSIX permission bits of ``path`` (mode & 0o777)."""
    return path.stat().st_mode & 0o777


def _make_doc(doc_id: str) -> Document:
    """Build a minimal Document for trash-write tests."""
    return Document(
        id=doc_id,
        content=f"content of {doc_id}",
        embedding=[0.1, 0.2, 0.3],
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=f"title {doc_id}",
            source_type="web",
            stored_at="2026-01-01T00:00:00+00:00",
            embedding_model="mock",
            embedding_dim=3,
            namespace="default",
            tags=["sample"],
            content_hash=f"hash-{doc_id}",
        ),
    )


# ---------------------------------------------------------------------------
# Linux / Mac branch — real chmod 0o600
# ---------------------------------------------------------------------------


@posix_only
def test_configure_writes_config_toml_with_mode_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``configure.configure`` chmods config.toml to 0o600 on Linux/Mac."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = configure.configure(
        backend="qdrant",
        url="https://q.example",
        api_key="secret-key-do-not-leak",
    )
    assert result.path.exists()
    assert _mode(result.path) == 0o600


@posix_only
def test_configure_rewrites_config_with_mode_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``configure`` call also lands on 0o600 (idempotent)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    configure.configure(backend="sqlite")
    cfg = configure.config_path()
    os.chmod(cfg, 0o644)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    configure.configure(backend="chroma", url="http://h:8000")
    assert _mode(cfg) == 0o600


@posix_only
def test_trash_append_chmods_file_600(tmp_path: Path) -> None:
    """Appending to the trash file leaves it on 0o600."""
    target = tmp_path / ".crossmem-trash.jsonl"
    cleanup._append_records(target, [_make_doc("a")])
    assert target.exists()
    assert _mode(target) == 0o600


@posix_only
def test_trash_rewrite_chmods_file_600(tmp_path: Path) -> None:
    """Atomic rewrite of the trash file leaves it on 0o600."""
    target = tmp_path / ".crossmem-trash.jsonl"
    target.write_text('{"x": 1}\n', encoding="utf-8")
    os.chmod(target, 0o644)
    cleanup._atomic_rewrite(target, ['{"y": 2}'])
    assert _mode(target) == 0o600


@posix_only
def test_sqlite_backend_creates_db_with_mode_600(tmp_path: Path) -> None:
    """A new file-based SQLite DB is created with 0o600."""
    db_file = tmp_path / "knowledge.db"
    backend = SQLiteBackend(db_file)
    try:
        assert db_file.exists()
        assert _mode(db_file) == 0o600
    finally:
        backend.close()


@posix_only
def test_doctor_warns_when_config_has_lax_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``doctor`` flags an existing config.toml with mode != 0o600 as warn."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    crossmem_dir = tmp_path / ".crossmem"
    crossmem_dir.mkdir()
    cfg = crossmem_dir / "config.toml"
    cfg.write_text('[backend]\nname = "sqlite"\n', encoding="utf-8")
    os.chmod(cfg, 0o644)

    result = doctor._check_sensitive_file_permissions()
    assert result.status == "warn"
    assert "config.toml" in result.detail


@posix_only
def test_doctor_ok_when_files_have_mode_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``doctor`` returns ok when every sensitive file is 0o600."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    crossmem_dir = tmp_path / ".crossmem"
    crossmem_dir.mkdir()
    for name in ("config.toml", ".crossmem-trash.jsonl", "knowledge.db"):
        p = crossmem_dir / name
        p.write_text("x", encoding="utf-8")
        os.chmod(p, 0o600)

    result = doctor._check_sensitive_file_permissions()
    assert result.status == "ok"


@posix_only
def test_doctor_ok_when_no_sensitive_files_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh install — no files yet — is reported as ok (nothing to warn about)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".crossmem").mkdir()
    result = doctor._check_sensitive_file_permissions()
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Windows branch — no crash, no chmod call
#
# We simulate the win32 codepath on every OS by monkeypatching the
# ``sys.platform`` attribute that lives inside each module under test.
# That guarantees coverage of the Windows guard even when the suite runs
# on Linux/Mac, and lets the test run unchanged on Windows itself.
# ---------------------------------------------------------------------------


def _patch_sys_platform_to_win32(
    monkeypatch: pytest.MonkeyPatch, module_path: str
) -> list[tuple[str, int]]:
    """Pretend ``module_path.sys.platform == 'win32'`` and record chmod calls."""
    monkeypatch.setattr(f"{module_path}.sys.platform", "win32", raising=False)
    calls: list[tuple[str, int]] = []

    def spy_chmod(path: object, mode: int) -> None:
        calls.append((str(path), mode))

    monkeypatch.setattr(f"{module_path}.os.chmod", spy_chmod, raising=False)
    return calls


def test_configure_skips_chmod_when_platform_is_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The win32 branch of ``configure`` writes the file but never chmods it."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = _patch_sys_platform_to_win32(monkeypatch, "crossmem.configure")
    result = configure.configure(backend="sqlite")
    assert result.path.exists()
    assert calls == []


def test_trash_append_skips_chmod_when_platform_is_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The win32 branch of cleanup writes the trash file but never chmods it."""
    calls = _patch_sys_platform_to_win32(monkeypatch, "crossmem.cleanup")
    target = tmp_path / ".crossmem-trash.jsonl"
    cleanup._append_records(target, [_make_doc("a")])
    cleanup._atomic_rewrite(target, ['{"y": 2}'])
    assert target.exists()
    assert calls == []


def test_sqlite_backend_skips_chmod_when_platform_is_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The win32 branch of SQLiteBackend creates the DB without chmodding."""
    calls = _patch_sys_platform_to_win32(
        monkeypatch, "crossmem.backends.sqlite_backend"
    )
    db_file = tmp_path / "knowledge.db"
    backend = SQLiteBackend(db_file)
    try:
        assert db_file.exists()
        assert calls == []
    finally:
        backend.close()


def test_doctor_check_passes_on_win32_when_files_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On win32 the doctor permission check is a no-op (always ``ok``).

    Windows ACLs are not expressible as POSIX mode bits, so the check is
    short-circuited rather than producing spurious warnings.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    crossmem_dir = tmp_path / ".crossmem"
    crossmem_dir.mkdir()
    (crossmem_dir / "config.toml").write_text("x", encoding="utf-8")
    monkeypatch.setattr("crossmem.doctor.sys.platform", "win32", raising=False)
    result = doctor._check_sensitive_file_permissions()
    assert result.status == "ok"
    assert (
        "windows" in result.detail.lower() or "not applicable" in result.detail.lower()
    )
