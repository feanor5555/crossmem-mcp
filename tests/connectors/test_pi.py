"""Tests for the Pi MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors.pi import PiConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert PiConnector().name() == "pi"


def test_config_path_is_home_relative(fake_home: Path) -> None:
    expected = fake_home / ".pi" / "agent" / "mcp.json"
    assert PiConnector().config_path() == expected


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert PiConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    cfg = fake_home / ".pi" / "agent" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert PiConnector().detect() is True


def test_register_uses_mcp_servers_key(fake_home: Path) -> None:
    PiConnector().register("crossmem")

    cfg = fake_home / ".pi" / "agent" / "mcp.json"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "mcp" not in data
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".pi" / "agent" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    original = {"mcpServers": {"keep": {"command": "keep-cmd", "args": [], "env": {}}}}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    PiConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["mcpServers"]
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"

    backups = list(cfg.parent.glob("mcp.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".pi" / "agent" / "mcp.json"
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

    PiConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    PiConnector().unregister()
    assert not (fake_home / ".pi" / "agent" / "mcp.json").exists()


def test_roundtrip(fake_home: Path) -> None:
    connector = PiConnector()

    connector.register("crossmem")
    cfg = fake_home / ".pi" / "agent" / "mcp.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
