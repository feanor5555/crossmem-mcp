"""Tests for the OpenCode MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors import opencode as opencode_module
from crossmem.connectors.opencode import OpenCodeConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert OpenCodeConnector().name() == "opencode"


def test_config_path_linux(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    expected = fake_home / ".config" / "opencode" / "opencode.json"
    assert OpenCodeConnector().config_path() == expected


def test_config_path_macos(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "darwin")
    expected = fake_home / ".config" / "opencode" / "opencode.json"
    assert OpenCodeConnector().config_path() == expected


def test_config_path_windows_uses_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    appdata.mkdir(parents=True)
    monkeypatch.setattr(opencode_module.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))

    expected = appdata / "opencode" / "opencode.json"
    assert OpenCodeConnector().config_path() == expected


def test_config_path_windows_appdata_missing_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(opencode_module.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = tmp_path / "AppData" / "Roaming" / "opencode" / "opencode.json"
    assert OpenCodeConnector().config_path() == expected


def test_detect_false_when_missing(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    assert OpenCodeConnector().detect() is False


def test_detect_true_when_file_exists(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert OpenCodeConnector().detect() is True


def test_register_uses_mcp_key(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")

    OpenCodeConnector().register("crossmem")

    cfg = fake_home / ".config" / "opencode" / "opencode.json"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "mcpServers" not in data
    assert data["mcp"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    original = {"mcp": {"keep": {"command": "keep-cmd", "args": [], "env": {}}}}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    OpenCodeConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["mcp"]
    assert data["mcp"]["crossmem"]["command"] == "crossmem"

    backups = list(cfg.parent.glob("opencode.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_unregister_removes_only_crossmem(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "mcp": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "keep": {"command": "keep-cmd", "args": [], "env": {}},
                }
            }
        ),
        encoding="utf-8",
    )

    OpenCodeConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcp"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    OpenCodeConnector().unregister()
    assert not (fake_home / ".config" / "opencode" / "opencode.json").exists()


def test_roundtrip(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opencode_module.sys, "platform", "linux")
    connector = OpenCodeConnector()

    connector.register("crossmem")
    cfg = fake_home / ".config" / "opencode" / "opencode.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcp"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcp"]
