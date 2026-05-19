"""Tests for the Kilo Code MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors import _vscode as vscode_module
from crossmem.connectors.kilocode import KiloCodeConnector

_RELATIVE = Path(
    "User",
    "globalStorage",
    "kilocode.kilo-code",
    "settings",
    "mcp_settings.json",
)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert KiloCodeConnector().name() == "kilocode"


def test_config_path_linux(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    expected = fake_home / ".config" / "Code" / _RELATIVE
    assert KiloCodeConnector().config_path() == expected


def test_config_path_macos(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "darwin")
    expected = fake_home / "Library" / "Application Support" / "Code" / _RELATIVE
    assert KiloCodeConnector().config_path() == expected


def test_config_path_windows_uses_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    appdata.mkdir(parents=True)
    monkeypatch.setattr(vscode_module.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))

    expected = appdata / "Code" / _RELATIVE
    assert KiloCodeConnector().config_path() == expected


def test_config_path_windows_appdata_missing_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(vscode_module.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = tmp_path / "AppData" / "Roaming" / "Code" / _RELATIVE
    assert KiloCodeConnector().config_path() == expected


def test_detect_false_when_missing(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    assert KiloCodeConnector().detect() is False


def test_detect_true_when_file_exists(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "Code" / _RELATIVE
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert KiloCodeConnector().detect() is True


def test_register_creates_nested_parents(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")

    KiloCodeConnector().register("crossmem")

    cfg = fake_home / ".config" / "Code" / _RELATIVE
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "Code" / _RELATIVE
    cfg.parent.mkdir(parents=True)
    original = {"mcpServers": {"keep": {"command": "keep-cmd", "args": [], "env": {}}}}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    KiloCodeConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["mcpServers"]
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"

    backups = list(cfg.parent.glob("mcp_settings.json.bak.*"))
    assert len(backups) == 1


def test_unregister_removes_only_crossmem(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "Code" / _RELATIVE
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "keep": {"command": "keep-cmd", "args": [], "env": {}},
                }
            }
        ),
        encoding="utf-8",
    )

    KiloCodeConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    KiloCodeConnector().unregister()
    cfg = fake_home / ".config" / "Code" / _RELATIVE
    assert not cfg.exists()


def test_roundtrip(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")
    connector = KiloCodeConnector()

    connector.register("crossmem")
    cfg = fake_home / ".config" / "Code" / _RELATIVE
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]


def test_config_path_env_override_wins_over_platform_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    portable = tmp_path / "portable-vscode" / "user-data"
    monkeypatch.setenv("CROSSMEM_VSCODE_USER_DIR", str(portable))
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")

    expected = portable / _RELATIVE
    assert KiloCodeConnector().config_path() == expected


def test_config_path_env_override_unset_uses_platform_default(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CROSSMEM_VSCODE_USER_DIR", raising=False)
    monkeypatch.setattr(vscode_module.sys, "platform", "linux")

    expected = fake_home / ".config" / "Code" / _RELATIVE
    assert KiloCodeConnector().config_path() == expected
