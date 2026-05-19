"""Tests for ``crossmem status --connectors``.

The subcommand prints one row per registered :class:`CLIConnector`:

* ``name``           — the connector's ``name()`` value
* ``detected``       — ``yes`` / ``no`` (``connector.detect()``)
* ``registered``     — ``yes`` / ``no`` / ``-`` for "not applicable" if the
  config file does not exist; ``connector.is_registered()`` otherwise
* ``backups``        — count of ``<config_path>.bak.*`` siblings
* ``config_path``    — full absolute path to the connector's config file

Tests cover the three groups required by the spec (registered, not-registered,
not-detected) with real connector classes against fixture configs in a
``tmp_path`` fake home.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from crossmem import cli
from crossmem.connectors.claude_code import ClaudeCodeConnector
from crossmem.connectors.cursor import CursorConnector
from crossmem.connectors.opencode import OpenCodeConnector
from crossmem.connectors.zed import ZedConnector
from crossmem.connectors_status import (
    ConnectorStatus,
    format_status_table,
    gather_connector_status,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from crossmem.connectors.base import CLIConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch ``Path.home`` so connectors see ``tmp_path``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # APPDATA matters on Windows for some connectors; pin it to fake home so
    # tests stay stable regardless of host platform.
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    return tmp_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-connector is_registered (base + overrides)
# ---------------------------------------------------------------------------


def test_is_registered_false_when_config_missing(fake_home: Path) -> None:
    """Missing config file -> not registered, no crash."""
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_true_for_default_servers_key(fake_home: Path) -> None:
    """Default JSON+``mcpServers`` reader picks up the entry."""
    cfg = fake_home / ".claude.json"
    _write_json(
        cfg,
        {
            "mcpServers": {
                "crossmem": {"command": "python", "args": [], "env": {}},
            }
        },
    )
    assert ClaudeCodeConnector().is_registered() is True


def test_is_registered_false_when_only_other_entries(fake_home: Path) -> None:
    """Other MCP servers present but no crossmem -> not registered."""
    cfg = fake_home / ".claude.json"
    _write_json(
        cfg,
        {"mcpServers": {"other": {"command": "x", "args": [], "env": {}}}},
    )
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_false_for_empty_file(fake_home: Path) -> None:
    """An empty config file is not a parse error — just not registered."""
    cfg = fake_home / ".claude.json"
    cfg.write_text("", encoding="utf-8")
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_false_for_malformed_json(fake_home: Path) -> None:
    """Malformed JSON must not crash status — treat as not registered."""
    cfg = fake_home / ".claude.json"
    cfg.write_text("{not valid json", encoding="utf-8")
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_false_for_non_object_root(fake_home: Path) -> None:
    """A JSON array (or any non-object) at the root means no entry."""
    cfg = fake_home / ".claude.json"
    cfg.write_text('["mcpServers"]', encoding="utf-8")
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_false_when_servers_key_is_not_dict(fake_home: Path) -> None:
    """``mcpServers`` set to a list (instead of object) -> not registered."""
    cfg = fake_home / ".claude.json"
    _write_json(cfg, {"mcpServers": ["not", "a", "dict"]})  # type: ignore[arg-type]
    assert ClaudeCodeConnector().is_registered() is False


def test_is_registered_opencode_uses_mcp_key(fake_home: Path) -> None:
    """OpenCode uses ``mcp`` not ``mcpServers``; status must follow."""
    cfg = OpenCodeConnector().config_path()
    _write_json(cfg, {"mcp": {"crossmem": {"command": "x", "args": []}}})
    assert OpenCodeConnector().is_registered() is True


def test_is_registered_zed_uses_context_servers_key(fake_home: Path) -> None:
    """Zed uses ``context_servers`` not ``mcpServers``."""
    cfg = ZedConnector().config_path()
    _write_json(cfg, {"context_servers": {"crossmem": {"command": "x"}}})
    assert ZedConnector().is_registered() is True


def test_is_registered_goose_yaml_extensions(fake_home: Path) -> None:
    """Goose uses YAML + ``extensions`` key; status reads that schema."""
    from crossmem.connectors.goose import GooseConnector

    connector = GooseConnector()
    connector.register("python -m crossmem.server")
    assert connector.is_registered() is True


def test_is_registered_continuedev_json_nested(fake_home: Path) -> None:
    """Continue.dev legacy JSON nests servers under experimental.<key>."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.json"
    _write_json(
        cfg,
        {
            "experimental": {
                "modelContextProtocolServers": {
                    "crossmem": {"command": "x", "args": []}
                }
            }
        },
    )
    assert ContinueDevConnector().is_registered() is True


def test_is_registered_continuedev_yaml_variant(fake_home: Path) -> None:
    """Continue.dev 2.x YAML variant uses flat top-level ``mcpServers``."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "mcpServers:\n  crossmem:\n    command: x\n    args: []\n    env: {}\n",
        encoding="utf-8",
    )
    assert ContinueDevConnector().is_registered() is True


def test_is_registered_continuedev_no_files(fake_home: Path) -> None:
    """Neither JSON nor YAML present -> not registered."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    assert ContinueDevConnector().is_registered() is False


def test_is_registered_continuedev_json_empty(fake_home: Path) -> None:
    """Empty JSON file -> not registered, no crash."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("", encoding="utf-8")
    assert ContinueDevConnector().is_registered() is False


def test_is_registered_continuedev_json_malformed(fake_home: Path) -> None:
    """Malformed JSON -> not registered, no crash."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{not json", encoding="utf-8")
    assert ContinueDevConnector().is_registered() is False


def test_is_registered_continuedev_json_non_dict_root(fake_home: Path) -> None:
    """JSON array at root -> not registered."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[]", encoding="utf-8")
    assert ContinueDevConnector().is_registered() is False


def test_is_registered_continuedev_json_no_experimental(fake_home: Path) -> None:
    """JSON without ``experimental`` key -> not registered."""
    from crossmem.connectors.continuedev import ContinueDevConnector

    cfg = fake_home / ".continue" / "config.json"
    _write_json(cfg, {"unrelated": True})
    assert ContinueDevConnector().is_registered() is False


# ---------------------------------------------------------------------------
# gather_connector_status: the three required spec groups
# ---------------------------------------------------------------------------


def test_gather_status_three_groups(fake_home: Path) -> None:
    """One registered, one not-registered, one not-detected.

    Spec: ``--connectors`` must cover all three cases. We use real
    connector classes (no stubs) so the per-connector schema dispatch
    is exercised end to end.
    """
    # Registered: write a claude config with the crossmem entry.
    cc = ClaudeCodeConnector()
    _write_json(
        cc.config_path(),
        {"mcpServers": {"crossmem": {"command": "x", "args": [], "env": {}}}},
    )

    # Not-registered: cursor config exists but no crossmem entry.
    cu = CursorConnector()
    _write_json(
        cu.config_path(),
        {"mcpServers": {"other": {"command": "y", "args": [], "env": {}}}},
    )

    # Not-detected: OpenCode config does NOT exist.
    oc = OpenCodeConnector()
    assert not oc.config_path().exists()

    statuses = gather_connector_status([cc, cu, oc])
    by_name = {s.name: s for s in statuses}

    assert by_name["claude_code"].detected is True
    assert by_name["claude_code"].registered is True
    assert by_name["claude_code"].config_path == cc.config_path()

    assert by_name["cursor"].detected is True
    assert by_name["cursor"].registered is False

    assert by_name["opencode"].detected is False
    assert by_name["opencode"].registered is False
    assert by_name["opencode"].config_path == oc.config_path()


def test_gather_status_counts_backups(fake_home: Path) -> None:
    """``backups`` counts ``<config>.bak.*`` siblings (the timestamped format)."""
    cc = ClaudeCodeConnector()
    cfg = cc.config_path()
    _write_json(cfg, {"mcpServers": {}})
    # Drop a couple of timestamped backups next to the config file.
    (cfg.parent / f"{cfg.name}.bak.2026-01-01T00-00-00.000000Z").write_text(
        "{}", encoding="utf-8"
    )
    (cfg.parent / f"{cfg.name}.bak.2026-02-01T00-00-00.000000Z").write_text(
        "{}", encoding="utf-8"
    )
    # Unrelated sibling that should NOT be counted.
    (cfg.parent / f"{cfg.name}.tmp").write_text("{}", encoding="utf-8")

    [status] = gather_connector_status([cc])
    assert status.backups == 2


def test_gather_status_zero_backups_when_none(fake_home: Path) -> None:
    cc = ClaudeCodeConnector()
    _write_json(cc.config_path(), {"mcpServers": {}})
    [status] = gather_connector_status([cc])
    assert status.backups == 0


def test_gather_status_not_detected_returns_zero_backups(fake_home: Path) -> None:
    """A missing config dir reports 0 backups (not an error)."""
    oc = OpenCodeConnector()
    assert not oc.config_path().exists()
    [status] = gather_connector_status([oc])
    assert status.backups == 0


# ---------------------------------------------------------------------------
# format_status_table: stdlib-only table renderer
# ---------------------------------------------------------------------------


def _make_status(**overrides: object) -> ConnectorStatus:
    base = ConnectorStatus(
        name="claude_code",
        detected=True,
        registered=True,
        backups=2,
        config_path=Path("/home/u/.claude.json"),
    )
    fields = {**base.__dict__, **overrides}
    return ConnectorStatus(**fields)  # type: ignore[arg-type]


def test_format_table_has_header_and_rows() -> None:
    rows = [
        _make_status(name="claude_code", detected=True, registered=True, backups=2),
        _make_status(name="cursor", detected=True, registered=False, backups=0),
        _make_status(
            name="opencode",
            detected=False,
            registered=False,
            backups=0,
            config_path=Path("/x/opencode.json"),
        ),
    ]
    table = format_status_table(rows)
    lines = table.splitlines()
    # First line is header, rows follow. We do not pin column widths
    # (those are computed from the data) but every column must appear.
    assert "name" in lines[0]
    assert "detected" in lines[0]
    assert "registered" in lines[0]
    assert "backups" in lines[0]
    assert "config_path" in lines[0]
    # Three data rows after the header.
    name_rows = [line for line in lines[1:] if line.strip()]
    assert any("claude_code" in line for line in name_rows)
    assert any("cursor" in line for line in name_rows)
    assert any("opencode" in line for line in name_rows)


def test_format_table_renders_booleans_as_yes_no() -> None:
    table = format_status_table(
        [
            _make_status(detected=True, registered=True),
            _make_status(name="other", detected=False, registered=False),
        ]
    )
    # ``yes`` for True, ``no`` for False — keeps the output greppable.
    assert "yes" in table
    assert "no" in table
    # Capital-True / capital-False would mean we forgot to translate.
    assert "True" not in table
    assert "False" not in table


def test_format_table_not_detected_row_marks_registered_as_dash() -> None:
    """When the config file is missing, registered is shown as ``-``.

    Reasoning: ``registered=no`` for a not-detected CLI is misleading
    (the file does not exist, so the question does not apply). ``-``
    signals "not applicable" without inventing a third boolean state
    in the dataclass.
    """
    table = format_status_table(
        [
            _make_status(
                name="opencode",
                detected=False,
                registered=False,
                backups=0,
            )
        ]
    )
    # Find the opencode row.
    [row] = [line for line in table.splitlines() if "opencode" in line]
    # ``registered`` column should contain ``-`` for not-detected.
    assert " - " in row or row.rstrip().endswith("-") or "  -  " in row


# ---------------------------------------------------------------------------
# CLI wiring: crossmem status --connectors
# ---------------------------------------------------------------------------


def _stub_connectors(connectors: Iterable[CLIConnector]) -> object:
    """Return a value suitable for monkeypatching ``cli._status_connectors``.

    The CLI delegates to a factory that returns the active connectors so
    tests can inject a controlled set without touching real CLI configs.
    """
    materialised = list(connectors)
    return lambda: materialised


def test_cli_status_connectors_prints_table(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    fake_home: Path,
) -> None:
    """End-to-end: ``crossmem status --connectors`` prints all three groups."""
    cc = ClaudeCodeConnector()
    cu = CursorConnector()
    oc = OpenCodeConnector()

    _write_json(
        cc.config_path(),
        {"mcpServers": {"crossmem": {"command": "x", "args": [], "env": {}}}},
    )
    _write_json(
        cu.config_path(),
        {"mcpServers": {"other": {"command": "y", "args": [], "env": {}}}},
    )

    monkeypatch.setattr(
        cli, "_status_connectors_factory", _stub_connectors([cc, cu, oc])
    )
    exit_code = cli.main(["status", "--connectors"])
    assert exit_code == 0
    out = capsys.readouterr().out
    # Header + three rows.
    assert "name" in out
    assert "detected" in out
    assert "registered" in out
    assert "backups" in out
    assert "config_path" in out
    assert "claude_code" in out
    assert "cursor" in out
    assert "opencode" in out
    # Group markers: registered=yes for claude_code, no for cursor, - for opencode.
    claude_line = next(line for line in out.splitlines() if "claude_code" in line)
    cursor_line = next(line for line in out.splitlines() if "cursor" in line)
    opencode_line = next(line for line in out.splitlines() if "opencode" in line)
    assert " yes " in claude_line or claude_line.split().count("yes") >= 1
    # cursor: detected=yes but registered=no.
    tokens = cursor_line.split()
    assert "yes" in tokens
    assert "no" in tokens
    # opencode: detected=no, registered=- (dash).
    assert " no " in opencode_line or "no" in opencode_line.split()
    assert "-" in opencode_line


def test_cli_status_requires_connectors_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crossmem status`` (no flag) is a usage error (argparse exits 2).

    The status subcommand currently has only one report (``--connectors``);
    calling it without any flag is a usage error rather than an empty
    success so callers cannot mistake a no-op for "nothing to report".
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["status"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "connectors" in captured.err.lower() or "connectors" in captured.out.lower()


def test_cli_status_help_lists_connectors_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["status", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--connectors" in out
