"""Tests for the Cursor MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors.cursor import CursorConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert CursorConnector().name() == "cursor"


def test_config_path(fake_home: Path) -> None:
    assert CursorConnector().config_path() == fake_home / ".cursor" / "mcp.json"


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert CursorConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    cfg = fake_home / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert CursorConnector().detect() is True


def test_register_creates_parents_and_entry(fake_home: Path) -> None:
    CursorConnector().register("crossmem")

    cfg = fake_home / ".cursor" / "mcp.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data == {
        "mcpServers": {
            "crossmem": {"command": "crossmem", "args": [], "env": {}},
        }
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    original = {"mcpServers": {"alpha": {"command": "a-cmd", "args": [], "env": {}}}}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    CursorConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert set(data["mcpServers"].keys()) == {"alpha", "crossmem"}

    backups = list(cfg.parent.glob("mcp.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "alpha": {"command": "a-cmd", "args": [], "env": {}},
                }
            }
        ),
        encoding="utf-8",
    )

    CursorConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["alpha"]

    backups = list(cfg.parent.glob("mcp.json.bak.*"))
    assert len(backups) == 1


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    CursorConnector().unregister()
    assert not (fake_home / ".cursor" / "mcp.json").exists()


def test_roundtrip(fake_home: Path) -> None:
    connector = CursorConnector()
    connector.register("crossmem")
    cfg = fake_home / ".cursor" / "mcp.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
