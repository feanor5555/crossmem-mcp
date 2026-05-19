"""Kilo Code MCP-config connector.

Kilo Code is a Cline fork. Its global MCP configuration lives inside the
VSCode user globalStorage directory of the ``kilocode.kilo-code``
extension as ``settings/mcp_settings.json``. The platform-specific paths
mirror the Cline connector and can be overridden via the
``CROSSMEM_VSCODE_USER_DIR`` environment variable (see
:mod:`crossmem.connectors._vscode`).

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "kilocode")

KiloCodeConnector = make_flat_json_connector(_SPEC)
"""Connector for the Kilo Code VSCode extension."""

__all__ = ["KiloCodeConnector"]
