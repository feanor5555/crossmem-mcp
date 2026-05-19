"""Shared VSCode user-directory resolution for VSCode-extension connectors.

Used by the Cline and Kilo Code connectors, which both store their MCP
configuration inside the VSCode user ``globalStorage`` tree. The location
is platform-specific by default but can be overridden via the
``CROSSMEM_VSCODE_USER_DIR`` environment variable to support portable
VSCode installs.

Two entry points are exposed:

* :func:`_vscode_user_root` — reads ``sys.platform`` / ``os.environ``
  itself and is used at runtime when actually editing the host's
  config file.
* :func:`vscode_user_root_for_platform` — pure function that takes the
  target ``platform`` (and optional ``appdata`` override) as arguments.
  Used by the install-doc renderer so it can produce the Linux/macOS/
  Windows triplet from a single host without mutating ``sys.platform``
  globally (Task 21.4).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_OVERRIDE_ENV = "CROSSMEM_VSCODE_USER_DIR"


def vscode_user_root_for_platform(platform: str, *, appdata: str | None = None) -> Path:
    """Return the VSCode user root for a given target ``platform``.

    Pure function: never reads or writes ``sys.platform``. The
    ``CROSSMEM_VSCODE_USER_DIR`` override is still honoured because the
    install-doc renderer wants the same precedence the runtime uses.

    Resolution order:

    1. ``CROSSMEM_VSCODE_USER_DIR`` environment variable, if set
       (returned as-is without an existence check).
    2. Platform default:

       - ``linux``:  ``~/.config/Code``
       - ``darwin``: ``~/Library/Application Support/Code``
       - ``win32``:  ``<appdata>/Code`` (fallback:
         ``~/AppData/Roaming/Code`` when ``appdata`` is ``None``).
    """
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        return Path(override)
    if platform == "win32":
        if appdata:
            return Path(appdata) / "Code"
        return Path.home() / "AppData" / "Roaming" / "Code"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code"
    return Path.home() / ".config" / "Code"


def _vscode_user_root() -> Path:
    """Return the VSCode user root directory for the current host.

    Thin wrapper that captures the live ``sys.platform`` and
    ``%APPDATA%`` and forwards to :func:`vscode_user_root_for_platform`.
    """
    appdata = os.environ.get("APPDATA")
    return vscode_user_root_for_platform(sys.platform, appdata=appdata)
