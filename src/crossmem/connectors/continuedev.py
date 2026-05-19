"""Continue.dev MCP-config connector.

Continue.dev keeps its user configuration in ``~/.continue/`` on every
platform (it deliberately uses HOME-relative, not XDG/AppData) and now
ships two on-disk variants in parallel:

* **JSON (legacy / 1.x)** — ``~/.continue/config.json`` with MCP servers
  under the nested key ``experimental.modelContextProtocolServers``.
* **YAML (2.x)** — ``~/.continue/config.yaml`` with MCP servers under a
  flat top-level ``mcpServers`` mapping, matching most other MCP-aware
  CLIs.

If only one of the two files exists this connector reads/writes that
variant. If both exist the **YAML file wins** — that is the format
Continue 2.x actively uses. If neither exists, the JSON variant is
created as the default fallback to preserve historical behaviour.

Both the JSON and YAML branches delegate to the shared
:mod:`crossmem.connectors.config_io` helpers so timestamped backups and
atomic writes follow the same contract; only ``_build_entry`` is local
because the on-disk shape (``command``/``args``/``env``) is shared with
the other flat-JSON connectors.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import yaml

from crossmem.connectors.base import CLIConnector
from crossmem.connectors.config_io import (
    atomic_write_json,
    atomic_write_yaml,
    backup_config,
    load_json_config,
    load_yaml_config,
)

_CROSSMEM_KEY = "crossmem"
_EXPERIMENTAL_KEY = "experimental"
_MCP_KEY = "modelContextProtocolServers"
_YAML_SERVERS_KEY = "mcpServers"
_JSON_FILENAME = "config.json"
_YAML_FILENAME = "config.yaml"


def _build_entry(server_cmd: str) -> dict[str, Any]:
    """Split ``server_cmd`` into the standard ``command/args/env`` entry."""
    tokens = shlex.split(server_cmd) if server_cmd else []
    command = tokens[0] if tokens else server_cmd
    args = tokens[1:] if tokens else []
    return {"command": command, "args": args, "env": {}}


class ContinueDevConnector(CLIConnector):
    """Connector for the Continue.dev IDE extension/CLI."""

    is_gui_app = True

    def name(self) -> str:
        return "continuedev"

    def config_path(self) -> Path:
        """Return the JSON config path (legacy / 1.x layout)."""
        return Path.home() / ".continue" / _JSON_FILENAME

    def yaml_config_path(self) -> Path:
        """Return the YAML config path (Continue 2.x layout)."""
        return Path.home() / ".continue" / _YAML_FILENAME

    def detect(self) -> bool:
        """Detected when either config variant exists on disk."""
        return self.config_path().exists() or self.yaml_config_path().exists()

    def is_registered(self) -> bool:
        """Return True when ``crossmem`` is present in the active variant.

        Uses the same selection rule as :meth:`register`: YAML wins if
        ``config.yaml`` exists, otherwise the legacy JSON variant. The
        YAML variant stores entries under flat top-level ``mcpServers``;
        the JSON variant nests them under
        ``experimental.modelContextProtocolServers``.
        """
        yaml_path = self.yaml_config_path()
        if yaml_path.exists():
            try:
                data = load_yaml_config(yaml_path)
            except (OSError, ValueError, yaml.YAMLError):
                return False
            servers = data.get(_YAML_SERVERS_KEY)
            return isinstance(servers, dict) and _CROSSMEM_KEY in servers

        json_path = self.config_path()
        if not json_path.exists():
            return False
        try:
            text = json_path.read_text(encoding="utf-8")
        except OSError:
            return False
        if not text.strip():
            return False
        try:
            data = json.loads(text)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        experimental = data.get(_EXPERIMENTAL_KEY)
        if not isinstance(experimental, dict):
            return False
        servers = experimental.get(_MCP_KEY)
        return isinstance(servers, dict) and _CROSSMEM_KEY in servers

    def register(self, server_cmd: str) -> None:
        """Add the crossmem MCP entry to the active config variant.

        Picks the YAML variant if ``config.yaml`` exists (Continue 2.x;
        YAML wins when both files are present), otherwise the legacy
        JSON variant. The YAML branch writes a flat top-level
        ``mcpServers`` mapping; the JSON branch keeps the nested
        ``experimental.modelContextProtocolServers`` key. Both branches
        back the original file up before writing.
        """
        if self.yaml_config_path().exists():
            self._register_yaml(server_cmd)
            return
        self._register_json(server_cmd)

    def unregister(self) -> None:
        """Remove the crossmem entry from the active config variant.

        Same selection logic as :meth:`register`: YAML if present,
        otherwise JSON. No-op when neither file contains a ``crossmem``
        entry.
        """
        if self.yaml_config_path().exists():
            self._unregister_yaml()
            return
        self._unregister_json()

    def current_entry(self) -> dict[str, Any] | None:
        """Return the ``crossmem`` entry from the active config variant.

        Mirrors the selection logic of :meth:`register`: YAML wins if
        ``config.yaml`` exists, otherwise the legacy JSON file's
        nested ``experimental.modelContextProtocolServers.crossmem``
        slot is consulted. Returns ``None`` for missing files, parse
        errors, or absent keys — the dry-run code treats that as
        ``would add``.
        """
        if self.yaml_config_path().exists():
            return self._current_entry_yaml()
        return self._current_entry_json()

    def _current_entry_yaml(self) -> dict[str, Any] | None:
        path = self.yaml_config_path()
        try:
            data = load_yaml_config(path)
        except (yaml.YAMLError, ValueError, OSError):
            return None
        servers = data.get(_YAML_SERVERS_KEY)
        if not isinstance(servers, dict):
            return None
        entry = servers.get(_CROSSMEM_KEY)
        if not isinstance(entry, dict):
            return None
        return entry

    def _current_entry_json(self) -> dict[str, Any] | None:
        path = self.config_path()
        try:
            data = load_json_config(path)
        except (ValueError, OSError):
            return None
        experimental = data.get(_EXPERIMENTAL_KEY)
        if not isinstance(experimental, dict):
            return None
        servers = experimental.get(_MCP_KEY)
        if not isinstance(servers, dict):
            return None
        entry = servers.get(_CROSSMEM_KEY)
        if not isinstance(entry, dict):
            return None
        return entry

    # --- JSON branch -------------------------------------------------------

    def _register_json(self, server_cmd: str) -> None:
        """Add the crossmem MCP entry under ``experimental.<MCP_KEY>``.

        Preserves all other top-level keys *and* any non-MCP keys nested
        inside ``experimental`` (e.g. ``quickActions``). Backs the file
        up before writing.
        """
        path = self.config_path()
        backup_config(path)
        data = load_json_config(path)

        experimental = data.get(_EXPERIMENTAL_KEY)
        if not isinstance(experimental, dict):
            experimental = {}
        servers = experimental.get(_MCP_KEY)
        if not isinstance(servers, dict):
            servers = {}

        servers[_CROSSMEM_KEY] = _build_entry(server_cmd)
        experimental[_MCP_KEY] = servers
        data[_EXPERIMENTAL_KEY] = experimental
        atomic_write_json(path, data)

    def _unregister_json(self) -> None:
        """Remove the crossmem entry. No-op if file or nested key missing."""
        path = self.config_path()
        if not path.exists():
            return
        data = load_json_config(path)
        experimental = data.get(_EXPERIMENTAL_KEY)
        if not isinstance(experimental, dict):
            return
        servers = experimental.get(_MCP_KEY)
        if not isinstance(servers, dict) or _CROSSMEM_KEY not in servers:
            return
        backup_config(path)
        del servers[_CROSSMEM_KEY]
        experimental[_MCP_KEY] = servers
        data[_EXPERIMENTAL_KEY] = experimental
        atomic_write_json(path, data)

    # --- YAML branch (Continue 2.x) ---------------------------------------

    def _register_yaml(self, server_cmd: str) -> None:
        """Add the crossmem MCP entry under top-level ``mcpServers``.

        Preserves every other top-level key in the YAML file. Backs the
        file up before writing.
        """
        path = self.yaml_config_path()
        backup_config(path)
        data = load_yaml_config(path)

        servers = data.get(_YAML_SERVERS_KEY)
        if not isinstance(servers, dict):
            servers = {}
        servers[_CROSSMEM_KEY] = _build_entry(server_cmd)
        data[_YAML_SERVERS_KEY] = servers
        atomic_write_yaml(path, data)

    def _unregister_yaml(self) -> None:
        """Remove the crossmem entry from the YAML config. No-op when absent."""
        path = self.yaml_config_path()
        if not path.exists():
            return
        data = load_yaml_config(path)
        servers = data.get(_YAML_SERVERS_KEY)
        if not isinstance(servers, dict) or _CROSSMEM_KEY not in servers:
            return
        backup_config(path)
        del servers[_CROSSMEM_KEY]
        data[_YAML_SERVERS_KEY] = servers
        atomic_write_yaml(path, data)
