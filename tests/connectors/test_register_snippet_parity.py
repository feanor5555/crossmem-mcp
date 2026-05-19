"""Parity between ``connector.register()`` and ``connector.mcp_snippet()``.

Task 15.1 ships ``mcp_snippet(server_cmd)`` so install docs can show the
exact JSON/YAML entry each CLI expects. For that promise to hold, the
entry actually written by :meth:`register` MUST match the snippet —
otherwise the install guide tells the LLM one shape while the connector
writes a different one (and, before this fix, an unstartable
``{"command": "python -m crossmem.server", "args": []}``).

These tests run a per-connector roundtrip for both a multi-token
``server_cmd`` ("python -m crossmem.server") and a single-token form
("crossmem"), reading back the file the connector just wrote and
comparing the crossmem entry to ``mcp_snippet(server_cmd)`` dict-equal.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from crossmem.connectors.amazonq import AmazonQConnector
from crossmem.connectors.claude_code import ClaudeCodeConnector
from crossmem.connectors.cline import ClineConnector
from crossmem.connectors.continuedev import ContinueDevConnector
from crossmem.connectors.cursor import CursorConnector
from crossmem.connectors.gemini import GeminiConnector
from crossmem.connectors.goose import GooseConnector
from crossmem.connectors.kilocode import KiloCodeConnector
from crossmem.connectors.opencode import OpenCodeConnector
from crossmem.connectors.pi import PiConnector
from crossmem.connectors.windsurf import WindsurfConnector
from crossmem.connectors.zed import ZedConnector

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from crossmem.connectors.base import CLIConnector


def _navigate(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Walk ``data`` along nested ``keys`` and return the final mapping."""
    node: Any = data
    for key in keys:
        assert isinstance(node, dict), f"expected dict at key path {keys!r}"
        assert key in node, f"missing key {key!r} along path {keys!r}"
        node = node[key]
    assert isinstance(node, dict), f"path {keys!r} did not lead to a dict"
    return node


# Each row pins how a connector lays out its on-disk config:
#   factory:    zero-arg constructor for the connector.
#   loader:     callable that parses the file the connector writes.
#   path:       nested key sequence under which ``crossmem`` lives.
#   filename:   the suffix the connector writes ("config.json" / "config.yaml")
#               — purely for clarity in test parametrize IDs.
CASES: list[
    tuple[Callable[[], CLIConnector], Callable[[str], Any], tuple[str, ...], str]
] = [
    (
        ClaudeCodeConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "claude_code.json",
    ),
    (
        CursorConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "cursor.json",
    ),
    (
        ClineConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "cline.json",
    ),
    (
        KiloCodeConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "kilocode.json",
    ),
    (
        WindsurfConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "windsurf.json",
    ),
    (
        GeminiConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "gemini.json",
    ),
    (
        AmazonQConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "amazonq.json",
    ),
    (
        PiConnector,
        lambda text: json.loads(text),
        ("mcpServers",),
        "pi.json",
    ),
    (
        OpenCodeConnector,
        lambda text: json.loads(text),
        ("mcp",),
        "opencode.json",
    ),
    (
        ZedConnector,
        lambda text: json.loads(text),
        ("context_servers",),
        "zed.json",
    ),
    (
        ContinueDevConnector,
        lambda text: json.loads(text),
        ("experimental", "modelContextProtocolServers"),
        "continuedev.json",
    ),
    (
        GooseConnector,
        lambda text: yaml.safe_load(text),
        ("extensions",),
        "goose.yaml",
    ),
]


@pytest.mark.parametrize(
    ("factory", "loader", "key_path", "filename"),
    CASES,
    ids=[case[0].__name__ for case in CASES],
)
@pytest.mark.parametrize(
    "server_cmd",
    ["python -m crossmem.server", "crossmem"],
    ids=["multi_token", "single_token"],
)
def test_register_writes_the_mcp_snippet(
    factory: Callable[[], CLIConnector],
    loader: Callable[[str], Any],
    key_path: tuple[str, ...],
    filename: str,
    server_cmd: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``register(cmd)`` writes exactly the dict ``mcp_snippet(cmd)`` returns.

    Routes the connector's writes through a tmp file via
    monkeypatching ``config_path``, so the test is independent of the
    real OS / XDG / APPDATA layout. The crossmem entry inside the file
    is then compared dict-equal to ``mcp_snippet(server_cmd)``.
    """
    connector = factory()
    target = tmp_path / filename
    monkeypatch.setattr(connector, "config_path", lambda: target)
    # Continue.dev now also probes a YAML companion path. Pin it to a
    # non-existent tmp file so the JSON branch is exercised here.
    if hasattr(connector, "yaml_config_path"):
        monkeypatch.setattr(
            connector,
            "yaml_config_path",
            lambda: tmp_path / "does-not-exist.yaml",
        )

    connector.register(server_cmd)

    written = loader(target.read_text(encoding="utf-8"))
    assert isinstance(written, dict), (
        f"{factory.__name__}: expected dict root, got {type(written).__name__}"
    )
    servers = _navigate(written, key_path)
    assert "crossmem" in servers, (
        f"{factory.__name__}: crossmem entry missing at {key_path!r}"
    )
    on_disk_entry = servers["crossmem"]
    expected = connector.mcp_snippet(server_cmd)
    assert on_disk_entry == expected, (
        f"{factory.__name__}: on-disk entry diverges from mcp_snippet()\n"
        f"  on disk: {on_disk_entry!r}\n"
        f"  snippet: {expected!r}"
    )


@pytest.mark.parametrize(
    "server_cmd",
    ["python -m crossmem.server", "crossmem"],
    ids=["multi_token", "single_token"],
)
def test_continuedev_yaml_register_matches_mcp_snippet(
    server_cmd: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML branch (Continue 2.x) must also write exactly ``mcp_snippet``.

    Mirrors :func:`test_register_writes_the_mcp_snippet` but exercises
    the YAML companion path: the file is pre-created (so the connector
    picks the YAML branch) and the crossmem entry under top-level
    ``mcpServers`` is compared dict-equal to ``mcp_snippet(server_cmd)``.
    """
    connector = ContinueDevConnector()
    yaml_target = tmp_path / "continuedev.yaml"
    json_target = tmp_path / "continuedev.json"
    yaml_target.write_text("name: pinned\n", encoding="utf-8")
    monkeypatch.setattr(connector, "config_path", lambda: json_target)
    monkeypatch.setattr(connector, "yaml_config_path", lambda: yaml_target)

    connector.register(server_cmd)

    written = yaml.safe_load(yaml_target.read_text(encoding="utf-8"))
    assert isinstance(written, dict)
    servers = _navigate(written, ("mcpServers",))
    assert "crossmem" in servers
    expected = connector.mcp_snippet(server_cmd)
    assert servers["crossmem"] == expected, (
        "Continue.dev YAML branch diverges from mcp_snippet()\n"
        f"  on disk: {servers['crossmem']!r}\n"
        f"  snippet: {expected!r}"
    )
    # JSON branch must not have been touched.
    assert not json_target.exists()
