"""Tests for the Goose MCP connector."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crossmem.connectors import goose as goose_module
from crossmem.connectors.goose import GooseConnector


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_name() -> None:
    assert GooseConnector().name() == "goose"


def test_config_path_linux(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    expected = fake_home / ".config" / "goose" / "config.yaml"
    assert GooseConnector().config_path() == expected


def test_config_path_macos(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "darwin")
    expected = fake_home / ".config" / "goose" / "config.yaml"
    assert GooseConnector().config_path() == expected


def test_config_path_windows_uses_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    appdata.mkdir(parents=True)
    monkeypatch.setattr(goose_module.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))

    expected = appdata / "goose" / "config.yaml"
    assert GooseConnector().config_path() == expected


def test_config_path_windows_appdata_missing_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(goose_module.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = tmp_path / "AppData" / "Roaming" / "goose" / "config.yaml"
    assert GooseConnector().config_path() == expected


def test_detect_false_when_missing(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    assert GooseConnector().detect() is False


def test_detect_true_when_file_exists(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("extensions: {}\n", encoding="utf-8")
    assert GooseConnector().detect() is True


def test_register_uses_extensions_key(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")

    GooseConnector().register("crossmem")

    cfg = fake_home / ".config" / "goose" / "config.yaml"
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert "mcpServers" not in data
    assert data["extensions"]["crossmem"] == {
        "type": "stdio",
        "cmd": "crossmem",
        "args": [],
        "enabled": True,
    }


def test_register_preserves_other_extensions_and_backs_up(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    original = {
        "extensions": {
            "keep": {
                "type": "stdio",
                "cmd": "keep-cmd",
                "args": [],
                "enabled": True,
            }
        },
        "GOOSE_PROVIDER": "openai",
    }
    cfg.write_text(yaml.safe_dump(original), encoding="utf-8")

    GooseConnector().register("crossmem")

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert "keep" in data["extensions"]
    assert data["extensions"]["crossmem"]["cmd"] == "crossmem"
    assert data["GOOSE_PROVIDER"] == "openai"

    backups = list(cfg.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    assert yaml.safe_load(backups[0].read_text(encoding="utf-8")) == original


def test_unregister_removes_only_crossmem(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "extensions": {
                    "crossmem": {
                        "type": "stdio",
                        "cmd": "crossmem",
                        "args": [],
                        "enabled": True,
                    },
                    "keep": {
                        "type": "stdio",
                        "cmd": "keep-cmd",
                        "args": [],
                        "enabled": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    GooseConnector().unregister()

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert list(data["extensions"].keys()) == ["keep"]


def test_unregister_missing_file_is_noop(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    GooseConnector().unregister()
    assert not (fake_home / ".config" / "goose" / "config.yaml").exists()


def test_register_handles_empty_file(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank/whitespace-only config file should be treated as empty mapping."""
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("   \n", encoding="utf-8")

    GooseConnector().register("crossmem")

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["extensions"]["crossmem"]["cmd"] == "crossmem"


def test_register_handles_null_yaml(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML file whose content parses to ``None`` is treated as empty."""
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("# only a comment\n", encoding="utf-8")

    GooseConnector().register("crossmem")

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["extensions"]["crossmem"]["cmd"] == "crossmem"


def test_register_rejects_non_mapping_yaml(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML file holding a list (not a mapping) is a hard error."""
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a YAML mapping"):
        GooseConnector().register("crossmem")


def test_unregister_no_crossmem_entry_is_noop(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the extensions block has no crossmem key, do not rewrite the file."""
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    original = "extensions:\n  keep:\n    type: stdio\n    cmd: keep-cmd\n"
    cfg.write_text(original, encoding="utf-8")

    GooseConnector().unregister()

    # File content untouched and no backup created (early-return path).
    assert cfg.read_text(encoding="utf-8") == original
    assert list(cfg.parent.glob("config.yaml.bak.*")) == []


def test_register_splits_server_cmd_into_cmd_and_args(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``server_cmd`` with spaces splits into Goose ``cmd`` + ``args``."""
    monkeypatch.setattr(goose_module.sys, "platform", "linux")

    GooseConnector().register("python -m crossmem.server")

    cfg = fake_home / ".config" / "goose" / "config.yaml"
    entry = yaml.safe_load(cfg.read_text(encoding="utf-8"))["extensions"]["crossmem"]
    assert entry["cmd"] == "python"
    assert entry["args"] == ["-m", "crossmem.server"]
    assert entry["type"] == "stdio"
    assert entry["enabled"] is True


def test_roundtrip(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(goose_module.sys, "platform", "linux")
    connector = GooseConnector()

    connector.register("crossmem")
    cfg = fake_home / ".config" / "goose" / "config.yaml"
    assert "crossmem" in yaml.safe_load(cfg.read_text(encoding="utf-8"))["extensions"]

    connector.unregister()
    assert (
        "crossmem" not in yaml.safe_load(cfg.read_text(encoding="utf-8"))["extensions"]
    )
