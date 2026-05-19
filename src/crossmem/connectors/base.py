"""Abstract base class for CLI connectors."""

from __future__ import annotations

import json
import shlex
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

CROSSMEM_ENTRY_KEY = "crossmem"


class CLIConnector(ABC):
    """Abstract base class for CLI MCP-config connectors.

    Subclasses must implement :meth:`name`, :meth:`detect`,
    :meth:`config_path`, :meth:`register` and :meth:`unregister`.

    Four pieces of LLM-install metadata have default implementations
    here and may be overridden:

    * :attr:`is_gui_app` — ``True`` for editor/IDE-style apps that
      require a manual restart after editing the MCP config,
      ``False`` for CLIs that pick up changes on the next invocation.
    * :meth:`restart_hint` — short human/LLM-readable instruction on
      how to apply the new config. Default branches on
      :attr:`is_gui_app`.
    * :meth:`mcp_snippet` — the inner JSON/YAML fragment that gets
      written under the CLI's MCP-server key for ``crossmem``. Must
      mirror the real per-CLI schema produced by :meth:`register`.
      The default returns the common
      ``{"command", "args", "env"}`` layout used by Claude Code,
      Cursor, Cline et al. Connectors whose CLI uses a different
      schema (e.g. Goose's ``{"type", "cmd", "args", "enabled"}``)
      override this method.
    * :meth:`current_entry` — the entry currently stored under the
      ``crossmem`` key in the CLI's config, used by
      ``crossmem install --dry-run`` to decide between
      ``would add`` / ``would update`` / ``already present``. The
      default reads :meth:`config_path` as JSON and looks under
      :attr:`servers_key`; YAML-based connectors (Goose) and
      connectors with a nested key path (Continue.dev legacy JSON)
      override this method.
    """

    #: ``True`` for editor/IDE-style apps (Cursor, Cline, Kilo Code,
    #: Windsurf, Zed, Continue.dev), ``False`` for CLI-driven tools
    #: that re-read their config on the next invocation. Subclasses
    #: override by setting this class attribute.
    is_gui_app: bool = False

    #: Top-level key in the JSON config that holds MCP-server entries.
    #: The 10 connectors with a flat layout default to ``mcpServers``;
    #: OpenCode overrides to ``mcp`` and Zed to ``context_servers``.
    #: YAML-based connectors (Goose) and nested-key connectors
    #: (Continue.dev legacy JSON) bypass this attribute by overriding
    #: :meth:`current_entry` directly.
    servers_key: ClassVar[str] = "mcpServers"

    @abstractmethod
    def name(self) -> str:
        """Return the connector's unique CLI name (e.g. "claude_code")."""

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this CLI appears installed (config dir exists)."""

    @abstractmethod
    def config_path(self) -> Path:
        """Return the platform-specific path to the CLI's MCP config file."""

    def paths_for_platform(self, platform: str, *, appdata: str | None = None) -> Path:
        """Return the config path that would be used on ``platform``.

        Pure function: must not read or write ``sys.platform`` or
        ``os.environ['APPDATA']``. Connectors whose runtime
        :meth:`config_path` branches on ``sys.platform`` MUST override
        this method and consume the ``platform`` / ``appdata`` arguments
        instead.

        The install-doc renderer (Task 21.4) calls this three times per
        connector — once for ``linux``, ``darwin`` and ``win32`` — to
        render the per-OS path triplet shown to the LLM. Doing the
        branching in a pure function lets the renderer stay thread-safe
        under ``pytest -n auto`` (previously the renderer flipped
        ``sys.platform`` globally inside a contextmanager, which was
        racy when tests ran concurrently).

        Default implementation: return :meth:`config_path` unchanged.
        That is correct for the connectors whose config path is the
        same on every platform (Claude Code, Cursor, Gemini, Amazon Q,
        Pi, Windsurf, Continue.dev). The remaining five connectors
        (OpenCode, Zed, Goose, Cline, Kilo Code) override this method.
        """
        return self.config_path()

    @abstractmethod
    def register(self, server_cmd: str) -> None:
        """Add the crossmem MCP server to the CLI's config (with backup)."""

    @abstractmethod
    def unregister(self) -> None:
        """Remove the crossmem entry from the CLI's config."""

    def is_registered(self) -> bool:
        """Return True if the CLI's config currently lists ``crossmem``.

        Default implementation reads :meth:`config_path` as a JSON
        object and checks ``data[servers_key]['crossmem']``. Returns
        ``False`` when the file is missing, unreadable, not a JSON
        object, or simply does not contain the entry. Connectors with
        non-JSON or non-flat schemas (Goose YAML, Continue.dev nested)
        override this method.
        """
        path = self.config_path()
        if not path.exists():
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False
        if not text.strip():
            return False
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, dict):
            return False
        servers = data.get(self.servers_key)
        if not isinstance(servers, dict):
            return False
        return CROSSMEM_ENTRY_KEY in servers

    def restart_hint(self) -> str:
        """Return a short instruction on how to apply the new MCP config.

        Default branches on :attr:`is_gui_app`:

        * GUI apps: ask the user to fully quit and relaunch the app
          (an in-app reload-window often does not re-spawn MCP
          servers).
        * CLI tools: the new config is picked up on the next
          invocation; no manual restart is required.

        Subclasses may override with a tool-specific phrasing.
        """
        if self.is_gui_app:
            return (
                f"Fully quit and relaunch {self.name()} so it re-reads the MCP "
                "config and re-spawns the crossmem server."
            )
        return (
            f"The new config is picked up on the next {self.name()} invocation; "
            "no manual restart is required."
        )

    def mcp_snippet(self, server_cmd: str) -> dict[str, Any]:
        """Return the inner MCP-server entry written under ``crossmem``.

        The default splits ``server_cmd`` into a ``command`` + ``args``
        pair via :func:`shlex.split` and returns the common JSON shape
        used by Claude Code, Cursor, Cline, Windsurf, Kilo Code, Pi,
        Gemini, Amazon Q, OpenCode, Zed and Continue.dev::

            {"command": "python", "args": ["-m", "crossmem.server"],
             "env": {}}

        Connectors whose CLI uses a different schema override this
        method. Goose, for example, returns the stdio extension entry
        ``{"type": "stdio", "cmd": ..., "args": ..., "enabled": True}``
        that its ``config.yaml`` actually expects.

        Every snippet MUST be a dict with at least one of ``command``
        or ``cmd`` (str) plus an ``args`` key (``list[str]``) so the
        generated install docs can render the entry uniformly.
        """
        parts = shlex.split(server_cmd) if server_cmd else []
        command = parts[0] if parts else server_cmd
        args = parts[1:] if parts else []
        return {"command": command, "args": args, "env": {}}

    def current_entry(self) -> dict[str, Any] | None:
        """Return the existing ``crossmem`` entry from the CLI's config.

        Used by ``crossmem install --dry-run`` to compare the
        already-stored entry against :meth:`mcp_snippet` and produce a
        ``would add`` / ``would update`` / ``already present`` status.

        The default implementation handles the 10 connectors with a flat
        JSON layout: read :meth:`config_path` as JSON, look up
        :attr:`servers_key`, return its ``crossmem`` child if present.
        Missing file, empty file, parse error, or absent key all yield
        ``None`` (treated as ``would add`` upstream — the dry-run code
        never aborts the install on a malformed config, it just falls
        back to the safer status).

        Connectors with YAML configs (Goose) or a nested JSON path
        (Continue.dev legacy ``experimental.modelContextProtocolServers``)
        override this method directly.
        """
        path = self.config_path()
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not text.strip():
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        servers = data.get(self.servers_key)
        if not isinstance(servers, dict):
            return None
        entry = servers.get("crossmem")
        if not isinstance(entry, dict):
            return None
        return entry
