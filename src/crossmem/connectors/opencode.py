"""OpenCode (sst/opencode) MCP-config connector.

OpenCode keeps user configuration in ``~/.config/opencode/opencode.json``
on Linux/macOS and ``%APPDATA%/opencode/opencode.json`` on Windows.
MCP servers live under the top-level ``mcp`` object.

The class is generated from the row in
:mod:`crossmem.connectors.registry` so all flat-JSON connectors share
one implementation. ``sys`` is intentionally re-exported at module
scope so existing tests can monkey-patch ``sys.platform`` via
``crossmem.connectors.opencode.sys`` to simulate the three host OSes.
"""

from __future__ import annotations

import sys  # noqa: F401 — re-exported so tests can patch sys.platform

from crossmem.connectors.registry import (
    FLAT_JSON_CONNECTORS,
    make_flat_json_connector,
)

_SPEC = next(spec for spec in FLAT_JSON_CONNECTORS if spec.name == "opencode")

OpenCodeConnector = make_flat_json_connector(_SPEC)
"""Connector for the OpenCode (sst/opencode) CLI."""

__all__ = ["OpenCodeConnector"]
