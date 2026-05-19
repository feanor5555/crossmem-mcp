"""Tests for ``tools/check_editable_install.py`` pre-push guard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD_PATH = REPO_ROOT / "tools" / "check_editable_install.py"


def _load_guard() -> ModuleType:
    """Load the guard script as a module, bypassing the ``tools`` package."""
    spec = importlib.util.spec_from_file_location("check_editable_install", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_repo(tmp_path: Path) -> Path:
    """Create a fake repo layout with ``src/crossmem/__init__.py``."""
    src = tmp_path / "src" / "crossmem"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _patch_git(monkeypatch, repo_root: Path) -> None:
    """Force ``git rev-parse --show-toplevel`` to return ``repo_root``."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=str(repo_root) + "\n",
                stderr="",
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)


def _patch_crossmem(monkeypatch, init_path: Path) -> None:
    """Make ``import crossmem`` inside the guard return a stub at ``init_path``."""
    stub = ModuleType("crossmem")
    stub.__file__ = str(init_path)
    monkeypatch.setitem(sys.modules, "crossmem", stub)


def test_guard_passes_when_import_resolves_inside_repo_src(
    tmp_path, monkeypatch
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    init_path = fake_repo / "src" / "crossmem" / "__init__.py"
    _patch_git(monkeypatch, fake_repo)
    _patch_crossmem(monkeypatch, init_path)

    guard = _load_guard()
    assert guard.main() == 0


def test_guard_fails_when_import_resolves_elsewhere(
    tmp_path, monkeypatch, capsys
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    # crossmem.__file__ points to a directory OUTSIDE fake_repo/src.
    elsewhere = tmp_path / "other" / "crossmem"
    elsewhere.mkdir(parents=True)
    elsewhere_init = elsewhere / "__init__.py"
    elsewhere_init.write_text("", encoding="utf-8")

    _patch_git(monkeypatch, fake_repo)
    _patch_crossmem(monkeypatch, elsewhere_init)

    guard = _load_guard()
    assert guard.main() == 1

    captured = capsys.readouterr()
    assert "pip install -e" in captured.err
    assert str(fake_repo) in captured.err
