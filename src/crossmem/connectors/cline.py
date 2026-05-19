"""Cline (VSCode extension) MCP-config connector.

Cline stores its MCP configuration inside the VSCode user globalStorage
directory of the ``saoudrizwan.claude-dev`` extension. The location is
platform-specific:

- Linux:   ``~/.config/Code/User/globalStorage/...``
- macOS:   ``~/Library/Application Support/Code/User/globalStorage/...``
- Windows: ``%APPDATA%/Code/User/globalStorage/...``

The base directory can be overridden via the
``CROSSMEM_VSCODE_USER_DIR`` environment variable to support portable
VSCode installs (see :mod:`crossmem.connectors._vscode`).

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation.
"""

from __future__ import annotations

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "cline")

ClineConnector = make_flat_json_connector(_SPEC)
"""Connector for the Cline VSCode extension."""

__all__ = ["ClineConnector"]
