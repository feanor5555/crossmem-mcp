"""Tests for the Claude Code MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors.claude_code import ClaudeCodeConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch ``Path.home`` to point at ``tmp_path``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert ClaudeCodeConnector().name() == "claude_code"


def test_config_path(fake_home: Path) -> None:
    expected = fake_home / ".claude.json"
    assert ClaudeCodeConnector().config_path() == expected


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert ClaudeCodeConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    (fake_home / ".claude.json").write_text("{}", encoding="utf-8")
    assert ClaudeCodeConnector().detect() is True


def test_register_creates_file_and_entry(fake_home: Path) -> None:
    connector = ClaudeCodeConnector()
    connector.register("crossmem")

    cfg = fake_home / ".claude.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".claude.json"
    original = {
        "mcpServers": {"other": {"command": "other-cmd", "args": ["x"], "env": {}}},
        "unrelated": "keep",
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    ClaudeCodeConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    assert data["mcpServers"]["other"]["command"] == "other-cmd"
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"
    assert data["unrelated"] == "keep"

    backups = list(fake_home.glob(".claude.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_register_into_nonexistent_parents(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if the parent dir is missing, register should create it."""
    nested_home = fake_home / "nested" / "home"
    monkeypatch.setattr(Path, "home", lambda: nested_home)

    ClaudeCodeConnector().register("crossmem")

    cfg = nested_home / ".claude.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".claude.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "other": {"command": "other-cmd", "args": [], "env": {}},
                }
            }
        ),
        encoding="utf-8",
    )

    ClaudeCodeConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "crossmem" not in data["mcpServers"]
    assert "other" in data["mcpServers"]

    backups = list(fake_home.glob(".claude.json.bak.*"))
    assert len(backups) == 1


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    ClaudeCodeConnector().unregister()
    assert not (fake_home / ".claude.json").exists()
    assert list(fake_home.glob(".claude.json.bak.*")) == []


def test_unregister_missing_entry_is_noop(fake_home: Path) -> None:
    cfg = fake_home / ".claude.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"other": {"command": "x", "args": [], "env": {}}}}),
        encoding="utf-8",
    )

    ClaudeCodeConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["other"]
    # No backup taken when there is nothing to remove.
    assert list(fake_home.glob(".claude.json.bak.*")) == []


def test_roundtrip(fake_home: Path) -> None:
    connector = ClaudeCodeConnector()
    connector.register("crossmem")
    cfg = fake_home / ".claude.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
