"""Tests for ``crossmem uninstall`` (uninstaller.py).

The uninstaller mirrors the installer:

1. Iterate over registered :class:`CLIConnector` classes and call their
   ``unregister()`` only when ``detect()`` is True.
2. Optionally filter to a single connector via ``cli="..."``.
3. Optionally purge the on-disk ``~/.crossmem`` directory, guarded by a
   double confirmation flag.

Tests use a fake ``$HOME`` and inject mock connectors so no real CLI
config is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from crossmem import installer, uninstaller
from crossmem.connectors.base import CLIConnector
from crossmem.connectors.config_io import (
    register_mcp_server,
    unregister_mcp_server,
)


@dataclass
class _RecordingConnector(CLIConnector):
    """Mock connector that records ``register``/``unregister`` calls."""

    cli_name: str
    detected: bool
    config_file: Path
    register_calls: list[str] = field(default_factory=list)
    unregister_calls: int = 0

    def name(self) -> str:
        return self.cli_name

    def detect(self) -> bool:
        return self.detected

    def config_path(self) -> Path:
        return self.config_file

    def register(self, server_cmd: str) -> None:
        self.register_calls.append(server_cmd)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text("{}", encoding="utf-8")

    def unregister(self) -> None:
        self.unregister_calls += 1


@dataclass
class _RealConfigConnector(CLIConnector):
    """Connector backed by a real JSON file (uses shared config_io helpers).

    Lets the round-trip test verify the on-disk config really is restored
    to the backup state after ``register`` -> ``uninstall``.
    """

    cli_name: str
    config_file: Path

    def name(self) -> str:
        return self.cli_name

    def detect(self) -> bool:
        return self.config_file.exists()

    def config_path(self) -> Path:
        return self.config_file

    def register(self, server_cmd: str) -> None:
        register_mcp_server(self.config_file, server_cmd)

    def unregister(self) -> None:
        unregister_mcp_server(self.config_file)


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    """Provide a temp directory that stands in for ``$HOME``.

    Returned via ``home_factory`` parameter rather than monkeypatching
    ``Path.home``: keeps the tests narrow and avoids leaking the patch
    into ``installer.ALL_CONNECTORS`` instantiation.
    """
    return tmp_path


def _make_connectors(
    fake_home: Path,
) -> tuple[_RecordingConnector, _RecordingConnector]:
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    missing = _RecordingConnector(
        cli_name="beta",
        detected=False,
        config_file=fake_home / ".beta" / "config.json",
    )
    return detected, missing


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------


def test_uninstall_unregisters_only_detected(fake_home: Path) -> None:
    """Detected connector gets ``unregister()``; missing one is skipped."""
    detected, missing = _make_connectors(fake_home)

    result = uninstaller.uninstall(
        connectors=[detected, missing],
        home_factory=lambda: fake_home,
    )

    assert detected.unregister_calls == 1
    assert missing.unregister_calls == 0
    assert result.unregistered_clis == ["alpha"]
    assert result.skipped_clis == ["beta"]
    assert result.purged_home is False


def test_uninstall_uses_default_connector_registry(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``connectors`` arg, the registry is taken from installer.

    Task 26.20 made ``installer.instantiate_connectors`` the canonical
    factory; the uninstaller's no-arg branch funnels through it, so
    patching ``installer.ALL_CONNECTORS`` is the single point of control.
    """
    detected, missing = _make_connectors(fake_home)
    monkeypatch.setattr(
        installer,
        "ALL_CONNECTORS",
        [lambda d=detected: d, lambda m=missing: m],
    )

    result = uninstaller.uninstall(home_factory=lambda: fake_home)

    assert result.unregistered_clis == ["alpha"]
    assert result.skipped_clis == ["beta"]


def test_uninstall_prints_progress(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Uninstall prints a human-readable progress trace to stdout."""
    detected, _missing = _make_connectors(fake_home)

    uninstaller.uninstall(connectors=[detected], home_factory=lambda: fake_home)

    out = capsys.readouterr().out
    assert "alpha" in out
    assert "removed crossmem entry" in out.lower()
    assert str(detected.config_file) in out


def test_uninstall_returns_dataclass(fake_home: Path) -> None:
    """:class:`UninstallResult` exposes the four expected fields."""
    detected, _missing = _make_connectors(fake_home)

    result = uninstaller.uninstall(
        connectors=[detected], home_factory=lambda: fake_home
    )

    assert isinstance(result, uninstaller.UninstallResult)
    assert isinstance(result.unregistered_clis, list)
    assert isinstance(result.skipped_clis, list)
    assert isinstance(result.purged_home, bool)
    assert isinstance(result.home_path, Path)


# ---------------------------------------------------------------------------
# --cli filter
# ---------------------------------------------------------------------------


def test_uninstall_cli_filter_touches_only_selected(fake_home: Path) -> None:
    """``cli="alpha"`` calls ``unregister`` on alpha; gamma is untouched."""
    alpha = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    gamma = _RecordingConnector(
        cli_name="gamma",
        detected=True,
        config_file=fake_home / ".gamma" / "config.json",
    )

    result = uninstaller.uninstall(
        connectors=[alpha, gamma],
        cli="alpha",
        home_factory=lambda: fake_home,
    )

    assert alpha.unregister_calls == 1
    assert gamma.unregister_calls == 0
    assert result.unregistered_clis == ["alpha"]
    # Filtered-out connectors do not appear in skipped either — they
    # were never considered for this run.
    assert result.skipped_clis == []


def test_uninstall_unknown_cli_raises(fake_home: Path) -> None:
    """Unknown ``--cli NAME`` raises :class:`UnknownConnectorError`."""
    alpha = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )

    with pytest.raises(uninstaller.UnknownConnectorError) as excinfo:
        uninstaller.uninstall(
            connectors=[alpha],
            cli="nope",
            home_factory=lambda: fake_home,
        )

    assert excinfo.value.requested == "nope"
    assert "alpha" in excinfo.value.known
    # Connector must not have been touched.
    assert alpha.unregister_calls == 0


# ---------------------------------------------------------------------------
# --purge / --yes
# ---------------------------------------------------------------------------


def test_uninstall_purge_without_confirm_keeps_home(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``purge=True`` alone refuses to remove ``~/.crossmem``."""
    detected, _missing = _make_connectors(fake_home)
    crossmem_dir = fake_home / ".crossmem"
    crossmem_dir.mkdir()
    (crossmem_dir / "knowledge.db").write_text("data", encoding="utf-8")

    result = uninstaller.uninstall(
        connectors=[detected],
        purge=True,
        confirm=False,
        home_factory=lambda: fake_home,
    )

    out = capsys.readouterr().out
    assert crossmem_dir.exists()
    assert (crossmem_dir / "knowledge.db").exists()
    assert result.purged_home is False
    assert "--yes" in out


def test_uninstall_purge_with_confirm_removes_home(fake_home: Path) -> None:
    """``purge=True`` + ``confirm=True`` removes ``~/.crossmem``."""
    detected, _missing = _make_connectors(fake_home)
    crossmem_dir = fake_home / ".crossmem"
    crossmem_dir.mkdir()
    (crossmem_dir / "knowledge.db").write_text("data", encoding="utf-8")

    result = uninstaller.uninstall(
        connectors=[detected],
        purge=True,
        confirm=True,
        home_factory=lambda: fake_home,
    )

    assert not crossmem_dir.exists()
    assert result.purged_home is True
    assert result.home_path == crossmem_dir


def test_uninstall_purge_missing_home_is_noop(fake_home: Path) -> None:
    """``purge=True`` on a missing ``~/.crossmem`` is a clean no-op."""
    detected, _missing = _make_connectors(fake_home)
    # No ~/.crossmem created.

    result = uninstaller.uninstall(
        connectors=[detected],
        purge=True,
        confirm=True,
        home_factory=lambda: fake_home,
    )

    assert result.purged_home is False
    assert not (fake_home / ".crossmem").exists()


def test_uninstall_no_purge_keeps_home(fake_home: Path) -> None:
    """Without ``purge`` the ``~/.crossmem`` directory is preserved."""
    detected, _missing = _make_connectors(fake_home)
    crossmem_dir = fake_home / ".crossmem"
    crossmem_dir.mkdir()
    (crossmem_dir / "knowledge.db").write_text("data", encoding="utf-8")

    result = uninstaller.uninstall(
        connectors=[detected],
        home_factory=lambda: fake_home,
    )

    assert crossmem_dir.exists()
    assert result.purged_home is False


# ---------------------------------------------------------------------------
# Round-trip with real config_io
# ---------------------------------------------------------------------------


def test_register_then_uninstall_restores_config_state(fake_home: Path) -> None:
    """Round-trip: register adds crossmem entry, uninstall removes it.

    Starting state: config has other MCP servers, no crossmem entry.
    After register -> uninstall the ``mcpServers`` object is back to
    the original content (same keys, same values).
    """
    config_file = fake_home / ".alpha" / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    original = {
        "mcpServers": {
            "keep": {"command": "keep-bin", "args": [], "env": {}},
        },
        "extra": 1,
    }
    config_file.write_text(json.dumps(original), encoding="utf-8")

    connector = _RealConfigConnector(cli_name="alpha", config_file=config_file)
    # Register first to simulate a prior ``crossmem install`` run.
    connector.register("crossmem")
    after_register = json.loads(config_file.read_text(encoding="utf-8"))
    assert "crossmem" in after_register["mcpServers"]

    # Now uninstall.
    uninstaller.uninstall(connectors=[connector], home_factory=lambda: fake_home)

    after_uninstall = json.loads(config_file.read_text(encoding="utf-8"))
    assert "crossmem" not in after_uninstall["mcpServers"]
    # Original entries are preserved.
    assert after_uninstall["mcpServers"]["keep"] == original["mcpServers"]["keep"]
    assert after_uninstall["extra"] == 1


def test_cli_filter_leaves_other_configs_untouched(fake_home: Path) -> None:
    """``cli="alpha"`` does not call unregister on gamma's real config."""
    alpha_cfg = fake_home / ".alpha" / "config.json"
    gamma_cfg = fake_home / ".gamma" / "config.json"
    alpha_cfg.parent.mkdir(parents=True, exist_ok=True)
    gamma_cfg.parent.mkdir(parents=True, exist_ok=True)
    alpha_cfg.write_text("{}", encoding="utf-8")
    gamma_cfg.write_text("{}", encoding="utf-8")

    alpha = _RealConfigConnector(cli_name="alpha", config_file=alpha_cfg)
    gamma = _RealConfigConnector(cli_name="gamma", config_file=gamma_cfg)
    alpha.register("crossmem")
    gamma.register("crossmem")
    gamma_after_register = gamma_cfg.read_text(encoding="utf-8")

    uninstaller.uninstall(
        connectors=[alpha, gamma],
        cli="alpha",
        home_factory=lambda: fake_home,
    )

    # Alpha's entry is gone; gamma's file is byte-identical to before.
    alpha_data = json.loads(alpha_cfg.read_text(encoding="utf-8"))
    assert "crossmem" not in alpha_data.get("mcpServers", {})
    assert gamma_cfg.read_text(encoding="utf-8") == gamma_after_register
