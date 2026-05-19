"""Goose MCP-config connector.

Goose (block/goose, >=0.5.6) keeps user configuration in
``~/.config/goose/config.yaml`` on Linux/macOS and
``%APPDATA%/goose/config.yaml`` on Windows. MCP servers live under the
top-level ``extensions`` mapping, with each entry shaped like::

    extensions:
      crossmem:
        type: stdio
        cmd: python
        args: ["-m", "crossmem.server"]
        enabled: true

YAML load/dump and timestamped backups come from the shared
:mod:`crossmem.connectors.config_io` helpers; only ``_build_entry``
remains local because Goose's stdio-extension schema
(``type``/``cmd``/``args``/``enabled``) is unique among supported CLIs.
The caller's ``server_cmd`` is split on whitespace into ``cmd`` + ``args``
so a typical ``"python -m crossmem.server"`` value lands in the right
slots.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

from crossmem.connectors.base import CLIConnector
from crossmem.connectors.config_io import (
    atomic_write_yaml,
    backup_config,
    load_yaml_config,
)

_FILENAME = "config.yaml"
_SERVERS_KEY = "extensions"
_CROSSMEM_KEY = "crossmem"


def _config_root_for_platform(platform: str, *, appdata: str | None) -> Path:
    """Pure variant of :func:`_config_root` for doc rendering.

    Takes the simulated ``platform`` and ``appdata`` value as arguments
    instead of reading ``sys.platform`` / ``os.environ``. Used by the
    install-doc renderer (Task 21.4) so it can produce the per-OS
    triplet without mutating process globals.
    """
    if platform == "win32":
        if appdata:
            return Path(appdata) / "goose"
        return Path.home() / "AppData" / "Roaming" / "goose"
    return Path.home() / ".config" / "goose"


def _config_root() -> Path:
    """Return the platform-specific Goose config directory."""
    return _config_root_for_platform(sys.platform, appdata=os.environ.get("APPDATA"))


def _build_entry(server_cmd: str) -> dict[str, Any]:
    """Split ``server_cmd`` into a Goose stdio extension entry."""
    parts = shlex.split(server_cmd)
    cmd = parts[0] if parts else server_cmd
    args = parts[1:] if parts else []
    return {
        "type": "stdio",
        "cmd": cmd,
        "args": args,
        "enabled": True,
    }


class GooseConnector(CLIConnector):
    """Connector for the Goose (block/goose) CLI."""

    def name(self) -> str:
        return "goose"

    def config_path(self) -> Path:
        return _config_root() / _FILENAME

    def paths_for_platform(self, platform: str, *, appdata: str | None = None) -> Path:
        """Return the Goose config path for the simulated ``platform``.

        Pure override of :meth:`CLIConnector.paths_for_platform` that
        consumes the ``platform`` and ``appdata`` arguments rather than
        reading ``sys.platform`` / ``os.environ``. Required by the
        install-doc renderer's thread-safe rendering path (Task 21.4).
        """
        return _config_root_for_platform(platform, appdata=appdata) / _FILENAME

    def detect(self) -> bool:
        return self.config_path().exists()

    def is_registered(self) -> bool:
        """Return True when ``extensions.crossmem`` exists in ``config.yaml``.

        Goose uses YAML (not JSON) and a non-standard top-level key
        (``extensions``, not ``mcpServers``), so the base-class default
        cannot answer this question.
        """
        path = self.config_path()
        if not path.exists():
            return False
        try:
            data = load_yaml_config(path)
        except (OSError, ValueError, yaml.YAMLError):
            return False
        extensions = data.get(_SERVERS_KEY)
        if not isinstance(extensions, dict):
            return False
        return _CROSSMEM_KEY in extensions

    def mcp_snippet(self, server_cmd: str) -> dict[str, Any]:
        """Return the Goose stdio extension entry written to ``config.yaml``.

        Goose does not use the common ``{"command", "args", "env"}``
        layout; instead each extension is a stdio entry of the form
        ``{"type": "stdio", "cmd": ..., "args": ..., "enabled": True}``.
        We delegate to :func:`_build_entry`, the same helper used by
        :meth:`register`, so the snippet shown to LLMs matches the YAML
        actually written on disk byte for byte.
        """
        return _build_entry(server_cmd)

    def current_entry(self) -> dict[str, Any] | None:
        """Return the ``crossmem`` extension currently stored in YAML.

        The base class default reads JSON; Goose stores its config as
        YAML under the top-level ``extensions`` mapping, so we override
        to parse it correctly. Malformed YAML, missing file, or absent
        key all yield ``None`` — the dry-run code interprets that as
        ``would add`` rather than aborting.
        """
        path = self.config_path()
        try:
            data = load_yaml_config(path)
        except (yaml.YAMLError, ValueError, OSError):
            return None
        extensions = data.get(_SERVERS_KEY)
        if not isinstance(extensions, dict):
            return None
        entry = extensions.get(_CROSSMEM_KEY)
        if not isinstance(entry, dict):
            return None
        return entry

    def register(self, server_cmd: str) -> None:
        path = self.config_path()
        backup_config(path)
        data = load_yaml_config(path)
        extensions = data.get(_SERVERS_KEY)
        if not isinstance(extensions, dict):
            extensions = {}
        extensions[_CROSSMEM_KEY] = _build_entry(server_cmd)
        data[_SERVERS_KEY] = extensions
        atomic_write_yaml(path, data)

    def unregister(self) -> None:
        path = self.config_path()
        if not path.exists():
            return
        data = load_yaml_config(path)
        extensions = data.get(_SERVERS_KEY)
        if not isinstance(extensions, dict) or _CROSSMEM_KEY not in extensions:
            return
        backup_config(path)
        del extensions[_CROSSMEM_KEY]
        data[_SERVERS_KEY] = extensions
        atomic_write_yaml(path, data)
