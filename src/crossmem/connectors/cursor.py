"""Cursor MCP-config connector.

Cursor stores its MCP configuration in ``~/.cursor/mcp.json`` on every
platform. The server entries live under the top-level ``mcpServers``
object.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "cursor")

CursorConnector = make_flat_json_connector(_SPEC)
"""Connector for the Cursor IDE."""

__all__ = ["CursorConnector"]
