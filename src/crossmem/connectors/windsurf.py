"""Windsurf (Codeium) MCP-config connector.

Windsurf keeps its MCP configuration in
``~/.codeium/windsurf/mcp_config.json`` on every platform. MCP servers
live under the top-level ``mcpServers`` object.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "windsurf")

WindsurfConnector = make_flat_json_connector(_SPEC)
"""Connector for the Windsurf editor (Codeium)."""

__all__ = ["WindsurfConnector"]
