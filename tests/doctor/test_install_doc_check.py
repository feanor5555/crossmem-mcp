"""Tests for ``_check_install_doc_present`` (task 17.3).

When a connector reports ``detect() == True``, the LLM-addressed install
guide ``install/<cli>.md`` must exist so an LLM driving the install can
follow it. If the file is missing, doctor emits a ``warn`` per missing
connector that points at ``crossmem docs install --cli <cli>`` (task
15.4).

The check is **per detected connector** — undetected connectors are
silently skipped (no warn, no ok), mirroring the existing
``_check_pi_mcp_adapter`` pattern. This keeps the doctor output free of
noise for users who only have one or two CLIs installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from crossmem import doctor

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubConnector:
    """Minimal fake connector exposing ``name()``, ``detect()``, ``config_path()``.

    ``config_path()`` returns a path inside a non-existent directory so the
    backup-retention doctor check (which shares this connector list via
    :func:`crossmem.doctor._install_doc_connectors`) sees zero ``.bak.*``
    siblings and stays quiet during install-doc tests.
    """

    def __init__(self, name: str, detected: bool) -> None:
        self._name = name
        self._detected = detected

    def name(self) -> str:
        return self._name

    def detect(self) -> bool:
        return self._detected

    def config_path(self) -> Path:
        # Path that does not exist -> _count_backup_siblings returns 0.
        return Path("/__crossmem_test_stub_nonexistent__") / f"{self._name}.json"


def _patch_install_dir(monkeypatch: pytest.MonkeyPatch, install_dir: Path) -> None:
    """Redirect doctor's install-dir resolver at ``install_dir``."""
    monkeypatch.setattr(doctor, "_install_docs_dir", lambda: install_dir)


def _patch_connectors(
    monkeypatch: pytest.MonkeyPatch, connectors: list[_StubConnector]
) -> None:
    """Replace doctor's connector factories with the given stubs."""
    monkeypatch.setattr(doctor, "_install_doc_connectors", lambda: list(connectors))


# ---------------------------------------------------------------------------
# _check_install_doc_present (per-connector)
# ---------------------------------------------------------------------------


def test_install_doc_skipped_when_connector_not_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undetected connector -> the check is omitted entirely (returns [])."""
    _patch_install_dir(monkeypatch, tmp_path)
    _patch_connectors(monkeypatch, [_StubConnector("claude_code", detected=False)])

    results = doctor._check_install_doc_present()

    assert results == []


def test_install_doc_ok_when_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connector detected + ``install/<cli>.md`` present -> single ``ok`` result."""
    (tmp_path / "claude_code.md").write_text("# stub", encoding="utf-8")
    _patch_install_dir(monkeypatch, tmp_path)
    _patch_connectors(monkeypatch, [_StubConnector("claude_code", detected=True)])

    results = doctor._check_install_doc_present()

    assert len(results) == 1
    assert results[0].name == "install_doc_claude_code"
    assert results[0].status == "ok"
    assert "claude_code.md" in results[0].detail


def test_install_doc_warn_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connector detected + file missing -> ``warn`` pointing at docs subcmd."""
    _patch_install_dir(monkeypatch, tmp_path)
    _patch_connectors(monkeypatch, [_StubConnector("cursor", detected=True)])

    results = doctor._check_install_doc_present()

    assert len(results) == 1
    result = results[0]
    assert result.name == "install_doc_cursor"
    assert result.status == "warn"
    # Detail must mention the missing-doc text + the regenerate command.
    assert "missing install doc for cursor" in result.detail.lower()
    assert "crossmem docs install" in result.detail
    assert "--cli cursor" in result.detail


def test_install_doc_warn_when_install_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the install/ dir itself is missing, every detected CLI warns."""
    nonexistent = tmp_path / "no-such-dir"
    _patch_install_dir(monkeypatch, nonexistent)
    _patch_connectors(monkeypatch, [_StubConnector("zed", detected=True)])

    results = doctor._check_install_doc_present()

    assert len(results) == 1
    assert results[0].name == "install_doc_zed"
    assert results[0].status == "warn"


def test_install_doc_mixed_results_only_for_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed: detected-with-file -> ok, detected-without-file -> warn,
    undetected -> omitted entirely.
    """
    (tmp_path / "claude_code.md").write_text("# stub", encoding="utf-8")
    _patch_install_dir(monkeypatch, tmp_path)
    _patch_connectors(
        monkeypatch,
        [
            _StubConnector("claude_code", detected=True),
            _StubConnector("cursor", detected=True),
            _StubConnector("zed", detected=False),
        ],
    )

    results = doctor._check_install_doc_present()
    by_name = {r.name: r for r in results}

    assert set(by_name) == {"install_doc_claude_code", "install_doc_cursor"}
    assert by_name["install_doc_claude_code"].status == "ok"
    assert by_name["install_doc_cursor"].status == "warn"


# ---------------------------------------------------------------------------
# run_checks() integration
# ---------------------------------------------------------------------------


def test_run_checks_omits_install_doc_when_no_connector_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandboxed home: no connector detected -> no install_doc_* entry emitted."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    results = doctor.run_checks()
    names = [r.name for r in results]
    assert not any(n.startswith("install_doc_") for n in names)


def test_run_checks_emits_install_doc_warn_for_detected_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected connector without an install doc surfaces a warn in run_checks."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    _patch_install_dir(monkeypatch, install_dir)
    _patch_connectors(monkeypatch, [_StubConnector("claude_code", detected=True)])

    results = doctor.run_checks()
    install_doc_results = [r for r in results if r.name.startswith("install_doc_")]

    assert len(install_doc_results) == 1
    assert install_doc_results[0].name == "install_doc_claude_code"
    assert install_doc_results[0].status == "warn"


def test_run_checks_emits_install_doc_ok_when_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected connector with an install doc surfaces an ``ok`` in run_checks."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / "cursor.md").write_text("# cursor", encoding="utf-8")
    _patch_install_dir(monkeypatch, install_dir)
    _patch_connectors(monkeypatch, [_StubConnector("cursor", detected=True)])

    results = doctor.run_checks()
    install_doc_results = [r for r in results if r.name.startswith("install_doc_")]

    assert len(install_doc_results) == 1
    assert install_doc_results[0].name == "install_doc_cursor"
    assert install_doc_results[0].status == "ok"


# ---------------------------------------------------------------------------
# Default install-dir resolution
# ---------------------------------------------------------------------------


def test_install_docs_dir_resolves_to_repo_root_install_when_present() -> None:
    """In a source checkout, ``_install_docs_dir()`` points at ``<repo>/install``.

    Resolved relative to the ``crossmem`` package location so it works
    both from a worktree (``src/crossmem/...``) and from a pip-installed
    layout. If neither exists, the function returns a non-existent path
    (callers treat that as "missing dir -> warn", not as an error).
    """
    path = doctor._install_docs_dir()
    # In the repo worktree, install/ exists at the repo root.
    assert path.name == "install"
