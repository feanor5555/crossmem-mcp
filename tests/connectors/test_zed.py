"""Tests for the Zed MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors import zed as zed_module
from crossmem.connectors.zed import ZedConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert ZedConnector().name() == "zed"


def test_config_path_linux(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    expected = fake_home / ".config" / "zed" / "settings.json"
    assert ZedConnector().config_path() == expected


def test_config_path_macos(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "darwin")
    expected = fake_home / ".config" / "zed" / "settings.json"
    assert ZedConnector().config_path() == expected


def test_config_path_windows_uses_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    appdata.mkdir(parents=True)
    monkeypatch.setattr(zed_module.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))

    expected = appdata / "Zed" / "settings.json"
    assert ZedConnector().config_path() == expected


def test_config_path_windows_appdata_missing_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(zed_module.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = tmp_path / "AppData" / "Roaming" / "Zed" / "settings.json"
    assert ZedConnector().config_path() == expected


def test_detect_false_when_missing(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    assert ZedConnector().detect() is False


def test_detect_true_when_file_exists(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "zed" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert ZedConnector().detect() is True


def test_register_uses_context_servers_key(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")

    ZedConnector().register("crossmem")

    cfg = fake_home / ".config" / "zed" / "settings.json"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "mcpServers" not in data
    assert data["context_servers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "zed" / "settings.json"
    cfg.parent.mkdir(parents=True)
    original = {
        "context_servers": {"keep": {"command": "keep-cmd", "args": [], "env": {}}},
        "theme": "One Dark",
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    ZedConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["context_servers"]
    assert data["context_servers"]["crossmem"]["command"] == "crossmem"
    assert data["theme"] == "One Dark"

    backups = list(cfg.parent.glob("settings.json.bak.*"))
    assert len(backups) == 1


def test_unregister_removes_only_crossmem(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "zed" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "context_servers": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "keep": {"command": "keep-cmd", "args": [], "env": {}},
                }
            }
        ),
        encoding="utf-8",
    )

    ZedConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["context_servers"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    ZedConnector().unregister()
    assert not (fake_home / ".config" / "zed" / "settings.json").exists()


def test_roundtrip(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zed_module.sys, "platform", "linux")
    connector = ZedConnector()

    connector.register("crossmem")
    cfg = fake_home / ".config" / "zed" / "settings.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["context_servers"]

    connector.unregister()
    assert (
        "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["context_servers"]
    )
