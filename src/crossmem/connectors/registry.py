"""Data-driven registry of flat-JSON CLI connectors.

Ten of the twelve shipped connectors (Claude Code, Cursor, Cline, Gemini,
Amazon Q, Pi, Windsurf, Kilo Code, OpenCode, Zed) share the same wiring:

* read/write a JSON config file at a CLI-specific path,
* place the ``crossmem`` entry under a flat top-level ``servers_key``
  (defaults to ``mcpServers``; OpenCode uses ``mcp`` and Zed uses
  ``context_servers``), and
* delegate the heavy lifting to
  :mod:`crossmem.connectors.config_io`.

Only two values vary across them — the on-disk path and a couple of
boolean/string knobs — so we capture each entry as a row in
:data:`FLAT_JSON_CONNECTORS` and synthesise the connector class via
:func:`make_flat_json_connector`. Adding a new flat-JSON CLI is a one-row
edit; no new file is required (though small per-CLI shim modules still
exist so ``import crossmem.connectors.cursor`` keeps working — and so
tests can monkey-patch each module's ``sys`` attribute to simulate
Linux/macOS/Windows).

Goose (YAML) and Continue.dev (nested JSON / YAML companion) deliberately
remain hand-written classes — their layouts deviate enough that forcing
them through the same shape would obscure rather than simplify the code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from crossmem.connectors.base import CLIConnector
from crossmem.connectors.config_io import (
    register_mcp_server,
    unregister_mcp_server,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FLAT_JSON_CONNECTORS",
    "FlatJsonConnectorSpec",
    "make_flat_json_connector",
]


# ---------------------------------------------------------------------------
# Path helpers used by individual connectors
# ---------------------------------------------------------------------------


def _claude_code_path() -> Path:
    """``~/.claude.json`` — same on every platform."""
    return Path.home() / ".claude.json"


def _cursor_path() -> Path:
    """``~/.cursor/mcp.json`` — same on every platform."""
    return Path.home() / ".cursor" / "mcp.json"


def _gemini_path() -> Path:
    """``~/.gemini/settings.json`` — same on every platform."""
    return Path.home() / ".gemini" / "settings.json"


def _amazonq_path() -> Path:
    """``~/.aws/amazonq/mcp.json`` — same on every platform."""
    return Path.home() / ".aws" / "amazonq" / "mcp.json"


def _pi_path() -> Path:
    """``~/.pi/agent/mcp.json`` — same on every platform (HOME-only)."""
    return Path.home() / ".pi" / "agent" / "mcp.json"


def _windsurf_path() -> Path:
    """``~/.codeium/windsurf/mcp_config.json`` — same on every platform."""
    return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"


# Cline + Kilo Code share the VSCode globalStorage root resolved by
# :func:`crossmem.connectors._vscode._vscode_user_root`. The extension
# subpath differs per connector.
_CLINE_RELATIVE = Path(
    "User",
    "globalStorage",
    "saoudrizwan.claude-dev",
    "settings",
    "cline_mcp_settings.json",
)

_KILOCODE_RELATIVE = Path(
    "User",
    "globalStorage",
    "kilocode.kilo-code",
    "settings",
    "mcp_settings.json",
)


def _cline_path() -> Path:
    """VSCode globalStorage path for the Cline extension."""
    from crossmem.connectors._vscode import _vscode_user_root

    return _vscode_user_root() / _CLINE_RELATIVE


def _kilocode_path() -> Path:
    """VSCode globalStorage path for the Kilo Code extension."""
    from crossmem.connectors._vscode import _vscode_user_root

    return _vscode_user_root() / _KILOCODE_RELATIVE


def _appdata_subdir_for_platform(folder: str, *, appdata: str | None) -> Path:
    """Return ``<appdata>/<folder>`` for the given ``appdata`` value.

    Pure variant of the runtime helper: takes the ``%APPDATA%`` value as
    an argument instead of reading ``os.environ``. Falls back to
    ``~/AppData/Roaming/<folder>`` when ``appdata`` is ``None`` (mirrors
    the historical CI fallback).
    """
    if appdata:
        return Path(appdata) / folder
    return Path.home() / "AppData" / "Roaming" / folder


def _appdata_subdir(folder: str) -> Path:
    """Return ``%APPDATA%/<folder>`` on Windows, falling back to the default.

    Mirrors the inline helpers previously duplicated in ``opencode.py`` and
    ``zed.py``: read ``APPDATA`` from the environment, falling back to the
    standard ``~/AppData/Roaming/<folder>`` layout when the variable is
    missing (rare/CI).
    """
    return _appdata_subdir_for_platform(folder, appdata=os.environ.get("APPDATA"))


def _opencode_path() -> Path:
    """Platform-specific OpenCode config path.

    * Windows: ``%APPDATA%/opencode/opencode.json``
    * Linux/macOS: ``~/.config/opencode/opencode.json``
    """
    if sys.platform == "win32":
        return _appdata_subdir("opencode") / "opencode.json"
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _opencode_path_for_platform(platform: str, *, appdata: str | None = None) -> Path:
    """Pure variant of :func:`_opencode_path` for doc rendering."""
    if platform == "win32":
        return _appdata_subdir_for_platform("opencode", appdata=appdata) / (
            "opencode.json"
        )
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _zed_path() -> Path:
    """Platform-specific Zed config path.

    * Windows: ``%APPDATA%/Zed/settings.json``
    * Linux/macOS: ``~/.config/zed/settings.json``
    """
    if sys.platform == "win32":
        return _appdata_subdir("Zed") / "settings.json"
    return Path.home() / ".config" / "zed" / "settings.json"


def _zed_path_for_platform(platform: str, *, appdata: str | None = None) -> Path:
    """Pure variant of :func:`_zed_path` for doc rendering."""
    if platform == "win32":
        return _appdata_subdir_for_platform("Zed", appdata=appdata) / ("settings.json")
    return Path.home() / ".config" / "zed" / "settings.json"


def _cline_path_for_platform(platform: str, *, appdata: str | None = None) -> Path:
    """Pure variant of :func:`_cline_path` for doc rendering."""
    from crossmem.connectors._vscode import vscode_user_root_for_platform

    return vscode_user_root_for_platform(platform, appdata=appdata) / _CLINE_RELATIVE


def _kilocode_path_for_platform(platform: str, *, appdata: str | None = None) -> Path:
    """Pure variant of :func:`_kilocode_path` for doc rendering."""
    from crossmem.connectors._vscode import vscode_user_root_for_platform

    return vscode_user_root_for_platform(platform, appdata=appdata) / _KILOCODE_RELATIVE


# ---------------------------------------------------------------------------
# Connector factory
# ---------------------------------------------------------------------------


class FlatJsonConnectorSpec:
    """Static description of a flat-JSON CLI connector.

    Captures the values that distinguish one flat-JSON connector from
    another:

    * ``name`` — value returned by :meth:`CLIConnector.name`.
    * ``config_path_fn`` — zero-arg callable returning the on-disk
      config :class:`~pathlib.Path` for the current host.
    * ``servers_key`` — top-level JSON key under which the ``crossmem``
      entry is written (``mcpServers`` for most, ``mcp`` for OpenCode,
      ``context_servers`` for Zed).
    * ``is_gui_app`` — ``True`` for editor/IDE-style apps that need a
      manual restart, ``False`` for CLI-driven tools.
    * ``paths_for_platform_fn`` — optional pure function used by the
      install-doc renderer (Task 21.4). Takes the simulated
      ``platform`` and an optional ``appdata`` override and returns
      the config path that the connector would produce there. When
      ``None`` (the default) the generated class falls back to
      :attr:`config_path_fn` for every platform; that is correct for
      the connectors whose path is identical on Linux/macOS/Windows
      (Claude Code, Cursor, Gemini, Amazon Q, Pi, Windsurf). The other
      flat-JSON connectors (OpenCode, Zed, Cline, Kilo Code) supply a
      pure variant of their runtime helper here.
    """

    __slots__ = (
        "config_path_fn",
        "is_gui_app",
        "name",
        "paths_for_platform_fn",
        "servers_key",
    )

    def __init__(
        self,
        *,
        name: str,
        config_path_fn: Callable[[], Path],
        servers_key: str = "mcpServers",
        is_gui_app: bool = False,
        paths_for_platform_fn: (Callable[..., Path] | None) = None,
    ) -> None:
        self.name = name
        self.config_path_fn = config_path_fn
        self.servers_key = servers_key
        self.is_gui_app = is_gui_app
        self.paths_for_platform_fn = paths_for_platform_fn


def make_flat_json_connector(spec: FlatJsonConnectorSpec) -> type[CLIConnector]:
    """Synthesise a :class:`CLIConnector` subclass from ``spec``.

    The returned class behaves exactly like the hand-written
    connectors used to: ``name()``/``config_path()`` come from the spec,
    ``detect()`` is a presence check on the config file, and
    ``register``/``unregister`` route through
    :mod:`crossmem.connectors.config_io` with the configured
    ``servers_key``.
    """
    spec_name = spec.name
    spec_path_fn = spec.config_path_fn
    spec_servers_key = spec.servers_key
    spec_is_gui_app = spec.is_gui_app
    spec_paths_for_platform_fn = spec.paths_for_platform_fn

    class _FlatJsonConnector(CLIConnector):
        """Generated flat-JSON connector (see :func:`make_flat_json_connector`)."""

        is_gui_app = spec_is_gui_app
        servers_key = spec_servers_key

        def name(self) -> str:
            return spec_name

        def config_path(self) -> Path:
            return spec_path_fn()

        def paths_for_platform(
            self, platform: str, *, appdata: str | None = None
        ) -> Path:
            if spec_paths_for_platform_fn is None:
                # Path is identical on every platform: bypass the
                # platform argument and return the runtime value.
                return spec_path_fn()
            return spec_paths_for_platform_fn(platform, appdata=appdata)

        def detect(self) -> bool:
            return self.config_path().exists()

        def register(self, server_cmd: str) -> None:
            register_mcp_server(
                self.config_path(),
                server_cmd,
                servers_key=spec_servers_key,
            )

        def unregister(self) -> None:
            unregister_mcp_server(
                self.config_path(),
                servers_key=spec_servers_key,
            )

    # Make repr/tracebacks readable; PEP 3155 qualname mirrors the public
    # alias each shim module re-exports (e.g. ``ClaudeCodeConnector``).
    class_name = "".join(part.capitalize() for part in spec_name.split("_")) + (
        "Connector"
    )
    _FlatJsonConnector.__name__ = class_name
    _FlatJsonConnector.__qualname__ = class_name
    return _FlatJsonConnector


# ---------------------------------------------------------------------------
# Registry table
# ---------------------------------------------------------------------------


# One row per shipped flat-JSON connector. Order matches
# :data:`crossmem.installer.ALL_CONNECTORS` for the connectors in this
# table; Goose and Continue.dev are not flat-JSON and live as
# hand-written modules.
FLAT_JSON_CONNECTORS: list[FlatJsonConnectorSpec] = [
    FlatJsonConnectorSpec(
        name="claude_code",
        config_path_fn=_claude_code_path,
    ),
    FlatJsonConnectorSpec(
        name="cursor",
        config_path_fn=_cursor_path,
        is_gui_app=True,
    ),
    FlatJsonConnectorSpec(
        name="cline",
        config_path_fn=_cline_path,
        paths_for_platform_fn=_cline_path_for_platform,
        is_gui_app=True,
    ),
    FlatJsonConnectorSpec(
        name="pi",
        config_path_fn=_pi_path,
    ),
    FlatJsonConnectorSpec(
        name="opencode",
        config_path_fn=_opencode_path,
        paths_for_platform_fn=_opencode_path_for_platform,
        servers_key="mcp",
    ),
    FlatJsonConnectorSpec(
        name="kilocode",
        config_path_fn=_kilocode_path,
        paths_for_platform_fn=_kilocode_path_for_platform,
        is_gui_app=True,
    ),
    FlatJsonConnectorSpec(
        name="gemini",
        config_path_fn=_gemini_path,
    ),
    FlatJsonConnectorSpec(
        name="windsurf",
        config_path_fn=_windsurf_path,
        is_gui_app=True,
    ),
    FlatJsonConnectorSpec(
        name="amazonq",
        config_path_fn=_amazonq_path,
    ),
    FlatJsonConnectorSpec(
        name="zed",
        config_path_fn=_zed_path,
        paths_for_platform_fn=_zed_path_for_platform,
        servers_key="context_servers",
        is_gui_app=True,
    ),
]
