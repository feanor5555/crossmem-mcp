"""Tests for the shared register/unregister helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from crossmem.connectors.config_io import (
    BACKUP_PREFIX,
    atomic_write_yaml,
    backup_config,
    load_yaml_config,
    register_mcp_server,
    unregister_mcp_server,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_register_into_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.json"
    cfg.write_text("   \n", encoding="utf-8")

    register_mcp_server(cfg, "crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["crossmem"]["command"] == "crossmem"


def test_register_repairs_non_dict_mcp_servers(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"mcpServers": "broken"}), encoding="utf-8")

    register_mcp_server(cfg, "crossmem")

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"] == {
        "crossmem": {"command": "crossmem", "args": [], "env": {}},
    }


def test_register_rejects_non_object_root(tmp_path: Path) -> None:
    cfg = tmp_path / "list.json"
    cfg.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        register_mcp_server(cfg, "crossmem")


def test_unregister_keeps_other_servers_intact(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "crossmem": {"command": "crossmem", "args": [], "env": {}},
                    "keep": {"command": "keep", "args": [], "env": {}},
                },
                "extra": 1,
            }
        ),
        encoding="utf-8",
    )

    unregister_mcp_server(cfg)

    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"].keys()) == ["keep"]
    assert data["extra"] == 1


def test_unregister_no_mcp_servers_key_is_noop(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"other": 1}), encoding="utf-8")

    unregister_mcp_server(cfg)

    assert json.loads(cfg.read_text(encoding="utf-8")) == {"other": 1}
    assert list(tmp_path.glob("cfg.json.bak.*")) == []


def test_unregister_non_dict_mcp_servers_is_noop(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": "not-a-dict"}), encoding="utf-8")

    unregister_mcp_server(cfg)

    assert json.loads(cfg.read_text(encoding="utf-8")) == {
        "mcpServers": "not-a-dict",
    }


def test_register_creates_unique_backups(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")

    register_mcp_server(cfg, "crossmem")
    register_mcp_server(cfg, "crossmem")

    backups = list(tmp_path.glob("cfg.json.bak.*"))
    assert len(backups) == 2


# --- YAML helpers ------------------------------------------------------------


def test_load_yaml_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_yaml_config(tmp_path / "missing.yaml") == {}


def test_load_yaml_empty_text_returns_empty(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("   \n", encoding="utf-8")
    assert load_yaml_config(cfg) == {}


def test_load_yaml_null_document_returns_empty(tmp_path: Path) -> None:
    cfg = tmp_path / "null.yaml"
    cfg.write_text("null\n", encoding="utf-8")
    assert load_yaml_config(cfg) == {}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        load_yaml_config(cfg)


def test_load_yaml_parses_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    assert load_yaml_config(cfg) == {"a": 1, "b": {"c": 2}}


def test_atomic_write_yaml_creates_parents(tmp_path: Path) -> None:
    cfg = tmp_path / "nested" / "deeper" / "config.yaml"
    atomic_write_yaml(cfg, {"k": "v", "nested": {"x": 1}})

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data == {"k": "v", "nested": {"x": 1}}


def test_atomic_write_yaml_preserves_unicode(tmp_path: Path) -> None:
    cfg = tmp_path / "uni.yaml"
    atomic_write_yaml(cfg, {"greeting": "Grueezi"})

    raw = cfg.read_text(encoding="utf-8")
    assert "Grueezi" in raw


def test_atomic_write_yaml_overwrites_atomically(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    atomic_write_yaml(cfg, {"a": 1})
    atomic_write_yaml(cfg, {"a": 2})

    assert list(tmp_path.glob("cfg.yaml.tmp")) == []
    assert yaml.safe_load(cfg.read_text(encoding="utf-8")) == {"a": 2}


# --- BACKUP_PREFIX constant --------------------------------------------------


def test_backup_prefix_constant_matches_disk_format(tmp_path: Path) -> None:
    """The constant is the exact infix that ``backup_config`` writes.

    Pinning the on-disk name to the constant guarantees that
    :mod:`crossmem.connectors_status` and :mod:`crossmem.doctor`, both
    of which count files via ``startswith(name + BACKUP_PREFIX)``, stay
    in lockstep with the writer.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")

    backup = backup_config(cfg)

    assert backup is not None
    assert backup.name.startswith(f"{cfg.name}{BACKUP_PREFIX}")
