"""Claude Code MCP-config connector.

The Claude Code CLI keeps user-level configuration in ``~/.claude.json``
(both Linux/macOS and Windows). The MCP server entries live under the
top-level ``mcpServers`` object.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "claude_code")

ClaudeCodeConnector = make_flat_json_connector(_SPEC)
"""Connector for the Claude Code CLI."""

__all__ = ["ClaudeCodeConnector"]
