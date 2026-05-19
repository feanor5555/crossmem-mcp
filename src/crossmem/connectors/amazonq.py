"""Amazon Q CLI MCP-config connector.

The Amazon Q CLI keeps its MCP configuration in
``~/.aws/amazonq/mcp.json`` on every platform. MCP servers live under
the top-level ``mcpServers`` object.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "amazonq")

AmazonQConnector = make_flat_json_connector(_SPEC)
"""Connector for the Amazon Q CLI."""

__all__ = ["AmazonQConnector"]
