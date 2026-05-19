"""Tests for the Windsurf MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors.windsurf import WindsurfConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert WindsurfConnector().name() == "windsurf"


def test_config_path(fake_home: Path) -> None:
    expected = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
    assert WindsurfConnector().config_path() == expected


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert WindsurfConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert WindsurfConnector().detect() is True


def test_register_creates_parents_and_entry(fake_home: Path) -> None:
    WindsurfConnector().register("crossmem")

    cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
    cfg.parent.mkdir(parents=True)
    original = {"mcpServers": {"keep": {"command": "keep-cmd", "args": [], "env": {}}}}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    WindsurfConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["mcpServers"]
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"

    backups = list(cfg.parent.glob("mcp_config.json.bak.*"))
    assert len(backups) == 1


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
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

    WindsurfConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    WindsurfConnector().unregister()
    assert not (fake_home / ".codeium" / "windsurf" / "mcp_config.json").exists()


def test_roundtrip(fake_home: Path) -> None:
    connector = WindsurfConnector()

    connector.register("crossmem")
    cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
