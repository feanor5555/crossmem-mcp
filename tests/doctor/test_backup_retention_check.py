"""Tests for ``_check_backup_retention`` (task 21.7).

After 21.7 :mod:`crossmem.connectors.config_io` actively prunes
``<config>.bak.<ts>`` files down to ``BACKUP_RETENTION`` generations on
every write. The doctor check exists for two reasons:

* legacy environments installed before 21.7 may already carry more than
  the cap on disk — those won't shrink until the user touches the
  config again, so the warning gives the LLM a hint to clean up;
* a config path that is never written to (the connector is detected but
  never registers crossmem) can also accumulate backups via other
  tooling — same hint applies.

The check is **per detected connector** (mirrors
``_check_install_doc_present``) so undetected CLIs add no noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from crossmem import doctor
from crossmem.connectors.config_io import BACKUP_RETENTION

if TYPE_CHECKING:
    import pytest


class _StubConnector:
    """Minimal fake connector exposing ``name()``, ``detect()``, ``config_path()``."""

    def __init__(self, name: str, detected: bool, config_path: Path) -> None:
        self._name = name
        self._detected = detected
        self._config_path = config_path

    def name(self) -> str:
        return self._name

    def detect(self) -> bool:
        return self._detected

    def config_path(self) -> Path:
        return self._config_path


def _patch_connectors(
    monkeypatch: pytest.MonkeyPatch, connectors: list[_StubConnector]
) -> None:
    monkeypatch.setattr(doctor, "_install_doc_connectors", lambda: list(connectors))


def _make_backups(cfg: Path, count: int) -> None:
    """Drop ``count`` synthetic ``cfg.bak.<i>`` siblings into ``cfg.parent``."""
    cfg.parent.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        stamp = f"2026-01-01T00-00-{i:02d}.000000Z"
        (cfg.parent / f"{cfg.name}.bak.{stamp}").write_text("x", encoding="utf-8")


# ---------------------------------------------------------------------------
# _check_backup_retention (per-connector)
# ---------------------------------------------------------------------------


def test_backup_retention_skipped_when_connector_not_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undetected connector -> no result emitted."""
    cfg = tmp_path / "cfg.json"
    _make_backups(cfg, count=BACKUP_RETENTION + 3)
    _patch_connectors(
        monkeypatch, [_StubConnector("claude_code", detected=False, config_path=cfg)]
    )

    results = doctor._check_backup_retention()

    assert results == []


def test_backup_retention_ok_when_count_at_or_below_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At or below ``BACKUP_RETENTION`` siblings -> single ``ok`` result."""
    cfg = tmp_path / "cfg.json"
    _make_backups(cfg, count=BACKUP_RETENTION)
    _patch_connectors(
        monkeypatch, [_StubConnector("cursor", detected=True, config_path=cfg)]
    )

    results = doctor._check_backup_retention()

    assert len(results) == 1
    assert results[0].name == "backup_retention_cursor"
    assert results[0].status == "ok"


def test_backup_retention_warn_when_count_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More than ``BACKUP_RETENTION`` siblings -> ``warn`` with cleanup hint."""
    cfg = tmp_path / "cfg.json"
    _make_backups(cfg, count=BACKUP_RETENTION + 4)
    _patch_connectors(
        monkeypatch, [_StubConnector("cursor", detected=True, config_path=cfg)]
    )

    results = doctor._check_backup_retention()

    assert len(results) == 1
    result = results[0]
    assert result.name == "backup_retention_cursor"
    assert result.status == "warn"
    assert str(BACKUP_RETENTION) in result.detail
    # Detail must surface the actual count so the user knows how far over
    # the cap they are.
    assert str(BACKUP_RETENTION + 4) in result.detail


def test_backup_retention_ok_when_no_backups_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``.bak.*`` siblings -> ``ok`` (zero is well under the cap)."""
    cfg = tmp_path / "cfg.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    _patch_connectors(
        monkeypatch, [_StubConnector("zed", detected=True, config_path=cfg)]
    )

    results = doctor._check_backup_retention()

    assert len(results) == 1
    assert results[0].name == "backup_retention_zed"
    assert results[0].status == "ok"


def test_backup_retention_only_for_detected_connectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mix of detected/undetected -> entries only for detected ones."""
    cfg_a = tmp_path / "a.json"
    cfg_b = tmp_path / "b.json"
    cfg_c = tmp_path / "c.json"
    _make_backups(cfg_a, count=2)
    _make_backups(cfg_b, count=BACKUP_RETENTION + 1)
    _make_backups(cfg_c, count=BACKUP_RETENTION + 1)
    _patch_connectors(
        monkeypatch,
        [
            _StubConnector("a", detected=True, config_path=cfg_a),
            _StubConnector("b", detected=True, config_path=cfg_b),
            _StubConnector("c", detected=False, config_path=cfg_c),
        ],
    )

    results = doctor._check_backup_retention()
    by_name = {r.name: r.status for r in results}

    assert set(by_name) == {"backup_retention_a", "backup_retention_b"}
    assert by_name["backup_retention_a"] == "ok"
    assert by_name["backup_retention_b"] == "warn"


# ---------------------------------------------------------------------------
# run_checks() integration
# ---------------------------------------------------------------------------


def test_run_checks_omits_backup_retention_when_no_connector_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandboxed home: no connector detected -> no ``backup_retention_*`` entry."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    results = doctor.run_checks()

    assert not any(r.name.startswith("backup_retention_") for r in results)


def test_run_checks_emits_backup_retention_warn_for_overcap_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected connector with > N backups surfaces a warn in run_checks."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = tmp_path / "cfg.json"
    _make_backups(cfg, count=BACKUP_RETENTION + 2)
    _patch_connectors(
        monkeypatch, [_StubConnector("cursor", detected=True, config_path=cfg)]
    )

    results = doctor.run_checks()
    retention_results = [r for r in results if r.name.startswith("backup_retention_")]

    assert len(retention_results) == 1
    assert retention_results[0].name == "backup_retention_cursor"
    assert retention_results[0].status == "warn"
