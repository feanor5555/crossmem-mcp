"""Pi (earendil-works/pi) MCP-config connector.

Pi has no native MCP support, but the ``pi-mcp-adapter`` npm package
hooks in as a Pi extension and reads MCP server configs from JSON
files. The global Pi-specific override lives at
``~/.pi/agent/mcp.json`` on every platform (Pi uses HOME-relative
paths throughout — no XDG, no APPDATA). MCP servers live under the
top-level ``mcpServers`` object, in the same shape as Claude Code /
Cursor / Cline.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "pi")

PiConnector = make_flat_json_connector(_SPEC)
"""Connector for the Pi CLI via pi-mcp-adapter."""

__all__ = ["PiConnector"]
