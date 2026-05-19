"""Tests for the Gemini CLI MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem.connectors.gemini import GeminiConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert GeminiConnector().name() == "gemini"


def test_config_path(fake_home: Path) -> None:
    expected = fake_home / ".gemini" / "settings.json"
    assert GeminiConnector().config_path() == expected


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert GeminiConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    cfg = fake_home / ".gemini" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert GeminiConnector().detect() is True


def test_register_creates_parents_and_entry(fake_home: Path) -> None:
    GeminiConnector().register("crossmem")

    cfg = fake_home / ".gemini" / "settings.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".gemini" / "settings.json"
    cfg.parent.mkdir(parents=True)
    original = {
        "mcpServers": {"keep": {"command": "keep-cmd", "args": [], "env": {}}},
        "theme": "dark",
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    GeminiConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["mcpServers"]
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"
    assert data["theme"] == "dark"

    backups = list(cfg.parent.glob("settings.json.bak.*"))
    assert len(backups) == 1


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".gemini" / "settings.json"
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

    GeminiConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    GeminiConnector().unregister()
    assert not (fake_home / ".gemini" / "settings.json").exists()


def test_roundtrip(fake_home: Path) -> None:
    connector = GeminiConnector()

    connector.register("crossmem")
    cfg = fake_home / ".gemini" / "settings.json"
    assert "crossmem" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    connector.unregister()
    assert "crossmem" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
