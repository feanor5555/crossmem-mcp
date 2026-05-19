"""Tests for the Continue.dev MCP connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from crossmem.connectors.continuedev import ContinueDevConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert ContinueDevConnector().name() == "continuedev"


def test_config_path(fake_home: Path) -> None:
    expected = fake_home / ".continue" / "config.json"
    assert ContinueDevConnector().config_path() == expected


def test_yaml_config_path(fake_home: Path) -> None:
    expected = fake_home / ".continue" / "config.yaml"
    assert ContinueDevConnector().yaml_config_path() == expected


def test_detect_false_when_missing(fake_home: Path) -> None:
    assert ContinueDevConnector().detect() is False


def test_detect_true_when_file_exists(fake_home: Path) -> None:
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert ContinueDevConnector().detect() is True


def test_detect_true_when_yaml_exists(fake_home: Path) -> None:
    cfg = fake_home / ".continue" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}\n", encoding="utf-8")
    assert ContinueDevConnector().detect() is True


def test_register_uses_nested_key(fake_home: Path) -> None:
    ContinueDevConnector().register("crossmem")

    cfg = fake_home / ".continue" / "config.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    # The Continue.dev JSON MCP key is nested:
    #   experimental.modelContextProtocolServers
    assert "mcpServers" not in data
    assert data["experimental"]["modelContextProtocolServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }


def test_register_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    original = {
        "experimental": {
            "modelContextProtocolServers": {
                "keep": {"command": "keep-cmd", "args": [], "env": {}},
            },
        },
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    servers = data["experimental"]["modelContextProtocolServers"]
    assert "keep" in servers
    assert servers["crossmem"]["command"] == "crossmem"

    backups = list(cfg.parent.glob("config.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_register_preserves_other_experimental_keys(fake_home: Path) -> None:
    """``register`` must not wipe non-MCP keys inside the experimental block."""
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    original = {
        "models": [{"title": "GPT-4", "provider": "openai"}],
        "experimental": {
            "quickActions": [{"title": "Explain"}],
            "useChromiumForDocsCrawling": True,
        },
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    # Top-level keys preserved.
    assert data["models"] == original["models"]
    # Other experimental keys preserved.
    assert data["experimental"]["quickActions"] == [{"title": "Explain"}]
    assert data["experimental"]["useChromiumForDocsCrawling"] is True
    # Crossmem entry was added in the nested key.
    assert (
        data["experimental"]["modelContextProtocolServers"]["crossmem"]["command"]
        == "crossmem"
    )


def test_unregister_removes_only_crossmem(fake_home: Path) -> None:
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "experimental": {
                    "modelContextProtocolServers": {
                        "crossmem": {"command": "crossmem", "args": [], "env": {}},
                        "keep": {"command": "keep-cmd", "args": [], "env": {}},
                    },
                    "quickActions": [{"title": "Explain"}],
                },
            }
        ),
        encoding="utf-8",
    )

    ContinueDevConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    servers = data["experimental"]["modelContextProtocolServers"]
    assert list(servers.keys()) == ["keep"]
    # Sibling experimental keys untouched.
    assert data["experimental"]["quickActions"] == [{"title": "Explain"}]


def test_unregister_missing_file_is_noop(fake_home: Path) -> None:
    ContinueDevConnector().unregister()
    assert not (fake_home / ".continue" / "config.json").exists()


def test_unregister_missing_nested_key_is_noop(fake_home: Path) -> None:
    """If ``experimental`` or its child key is absent, unregister is a no-op."""
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"models": []}), encoding="utf-8")

    ContinueDevConnector().unregister()

    # File unchanged, no backup created.
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data == {"models": []}
    assert list(cfg.parent.glob("config.json.bak.*")) == []


def test_unregister_missing_crossmem_entry_is_noop(fake_home: Path) -> None:
    """If the nested servers dict has no ``crossmem`` key, do not modify the file."""
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    original = {
        "experimental": {
            "modelContextProtocolServers": {
                "other": {"command": "other-cmd", "args": [], "env": {}},
            },
        },
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    ContinueDevConnector().unregister()

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data == original
    assert list(cfg.parent.glob("config.json.bak.*")) == []


def test_register_replaces_empty_file(fake_home: Path) -> None:
    """An empty/whitespace-only config file is treated as empty JSON."""
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("   \n", encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert (
        data["experimental"]["modelContextProtocolServers"]["crossmem"]["command"]
        == "crossmem"
    )


def test_register_rejects_non_dict_json(fake_home: Path) -> None:
    """A JSON file whose top-level is a list (or scalar) raises ValueError."""
    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        ContinueDevConnector().register("crossmem")


def test_roundtrip(fake_home: Path) -> None:
    connector = ContinueDevConnector()

    connector.register("crossmem")
    cfg = fake_home / ".continue" / "config.json"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "crossmem" in data["experimental"]["modelContextProtocolServers"]

    connector.unregister()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "crossmem" not in data["experimental"]["modelContextProtocolServers"]


# --- YAML variant (Continue 2.x: top-level ``mcpServers`` in config.yaml) -------


def test_register_writes_yaml_when_yaml_exists(fake_home: Path) -> None:
    """If ``config.yaml`` exists, ``register`` writes the YAML variant.

    Continue.dev 2.x uses ``config.yaml`` with a flat top-level
    ``mcpServers`` mapping, mirroring most other MCP-aware CLIs.
    """
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    yaml_cfg.write_text("name: my-config\n", encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    # Top-level ``mcpServers``, not nested.
    assert data["mcpServers"]["crossmem"] == {
        "command": "crossmem",
        "args": [],
        "env": {},
    }
    # Other top-level keys preserved.
    assert data["name"] == "my-config"
    # JSON variant must NOT be created.
    assert not (fake_home / ".continue" / "config.json").exists()


def test_register_yaml_preserves_other_servers_and_backs_up(fake_home: Path) -> None:
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    original_text = (
        "mcpServers:\n  keep:\n    command: keep-cmd\n    args: []\n    env: {}\n"
    )
    yaml_cfg.write_text(original_text, encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert set(data["mcpServers"].keys()) == {"keep", "crossmem"}
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"

    backups = list(yaml_cfg.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    assert yaml.safe_load(backups[0].read_text(encoding="utf-8")) == yaml.safe_load(
        original_text
    )


def test_register_prefers_yaml_when_both_exist(fake_home: Path) -> None:
    """When both ``config.yaml`` and ``config.json`` exist, YAML wins."""
    cfg_dir = fake_home / ".continue"
    cfg_dir.mkdir(parents=True)
    json_cfg = cfg_dir / "config.json"
    yaml_cfg = cfg_dir / "config.yaml"
    json_cfg.write_text("{}", encoding="utf-8")
    yaml_cfg.write_text("name: my-config\n", encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    # YAML got the crossmem entry.
    yaml_data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert yaml_data["mcpServers"]["crossmem"]["command"] == "crossmem"
    # JSON was left untouched (no nested entry written).
    json_data = json.loads(json_cfg.read_text(encoding="utf-8"))
    assert json_data == {}
    # No JSON backup either — only the YAML file was modified.
    assert list(cfg_dir.glob("config.json.bak.*")) == []


def test_unregister_yaml_when_only_yaml_exists(fake_home: Path) -> None:
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    yaml_cfg.write_text(
        "mcpServers:\n"
        "  crossmem:\n"
        "    command: crossmem\n"
        "    args: []\n"
        "    env: {}\n"
        "  keep:\n"
        "    command: keep-cmd\n"
        "    args: []\n"
        "    env: {}\n",
        encoding="utf-8",
    )

    ContinueDevConnector().unregister()

    data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]


def test_unregister_prefers_yaml_when_both_exist(fake_home: Path) -> None:
    """When both files exist, ``unregister`` strips crossmem from the YAML."""
    cfg_dir = fake_home / ".continue"
    cfg_dir.mkdir(parents=True)
    json_cfg = cfg_dir / "config.json"
    yaml_cfg = cfg_dir / "config.yaml"
    json_original = {
        "experimental": {
            "modelContextProtocolServers": {
                "crossmem": {"command": "crossmem", "args": [], "env": {}},
            },
        },
    }
    json_cfg.write_text(json.dumps(json_original), encoding="utf-8")
    yaml_cfg.write_text(
        "mcpServers:\n  crossmem:\n    command: crossmem\n    args: []\n    env: {}\n",
        encoding="utf-8",
    )

    ContinueDevConnector().unregister()

    # YAML lost the crossmem entry.
    yaml_data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert "crossmem" not in (yaml_data.get("mcpServers") or {})
    # JSON file untouched.
    assert json.loads(json_cfg.read_text(encoding="utf-8")) == json_original
    assert list(cfg_dir.glob("config.json.bak.*")) == []


def test_register_yaml_rejects_non_mapping(fake_home: Path) -> None:
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    yaml_cfg.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a YAML mapping"):
        ContinueDevConnector().register("crossmem")


def test_register_yaml_treats_empty_file_as_empty_mapping(fake_home: Path) -> None:
    """A whitespace-only YAML file is loaded as an empty mapping."""
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    yaml_cfg.write_text("   \n", encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"


def test_register_yaml_handles_null_document(fake_home: Path) -> None:
    """A YAML doc that parses to ``None`` (just ``~``) is treated as empty."""
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    yaml_cfg.write_text("~\n", encoding="utf-8")

    ContinueDevConnector().register("crossmem")

    data = yaml.safe_load(yaml_cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"


def test_unregister_yaml_missing_servers_key_is_noop(fake_home: Path) -> None:
    """If the YAML file has no ``mcpServers`` mapping, do nothing."""
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    original_text = "name: my-config\n"
    yaml_cfg.write_text(original_text, encoding="utf-8")

    ContinueDevConnector().unregister()

    # File unchanged, no backup created.
    assert yaml_cfg.read_text(encoding="utf-8") == original_text
    assert list(yaml_cfg.parent.glob("config.yaml.bak.*")) == []


def test_unregister_yaml_missing_crossmem_entry_is_noop(fake_home: Path) -> None:
    """If ``mcpServers`` has no ``crossmem`` key, the file is left alone."""
    yaml_cfg = fake_home / ".continue" / "config.yaml"
    yaml_cfg.parent.mkdir(parents=True)
    original_text = (
        "mcpServers:\n  keep:\n    command: keep-cmd\n    args: []\n    env: {}\n"
    )
    yaml_cfg.write_text(original_text, encoding="utf-8")

    ContinueDevConnector().unregister()

    assert yaml_cfg.read_text(encoding="utf-8") == original_text
    assert list(yaml_cfg.parent.glob("config.yaml.bak.*")) == []
