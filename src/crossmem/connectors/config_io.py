"""Shared helpers to (un)register the crossmem MCP server in a JSON config.

Every supported CLI keeps its MCP servers under a top-level ``mcpServers``
object. The helpers in this module load such a file (or create a new one),
mutate the ``mcpServers`` object, write it back atomically and create a
timestamped backup of the previous content first.

The lower-level building blocks — :func:`timestamp`, :func:`backup_config`,
:func:`load_json_config`, :func:`atomic_write_json`, :func:`load_yaml_config`
and :func:`atomic_write_yaml` — are public so connectors with non-standard
layouts (e.g. Continue.dev's nested key or Goose's YAML config) can reuse
them instead of duplicating the logic.
"""

from __future__ import annotations

import json
import shlex
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

CROSSMEM_KEY = "crossmem"

#: Suffix infix used for every timestamped config backup: a backup file
#: is named ``<config.name><BACKUP_PREFIX><timestamp>`` (e.g.
#: ``config.json.bak.2026-01-01T00-00-00.000000Z``). Centralised here so
#: the prefix lives in one place — :mod:`crossmem.connectors_status`
#: and :mod:`crossmem.doctor` import this constant for their
#: backup-counting checks instead of re-spelling the literal.
BACKUP_PREFIX = ".bak."

#: Maximum number of ``<config>.bak.<timestamp>`` siblings kept per config
#: file. Every successful :func:`backup_config` call prunes the oldest
#: entries above this cap. Five generations is enough to roll back a
#: handful of recent register/unregister mistakes without letting the
#: backups grow without bound — every ``crossmem install`` run produces
#: one backup per registered CLI, so an unbounded count means "as many
#: stale config copies as the user has ever installed crossmem".
BACKUP_RETENTION = 5


def timestamp() -> str:
    """Return a filesystem-safe ISO-8601 UTC timestamp."""
    # Filesystem-safe: replace ':' (forbidden on Windows) with '-'.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")


def _prune_backups(path: Path, *, keep: int = BACKUP_RETENTION) -> None:
    """Delete ``<path>.bak.*`` siblings beyond the newest ``keep`` entries.

    Timestamps embedded in the filename (see :func:`timestamp`) sort
    chronologically when sorted lexicographically, so the newest files
    are the last ``keep`` entries of the sorted list. Missing parent
    directory or fewer than ``keep`` files: no-op.

    Best-effort: an :class:`OSError` while unlinking is swallowed
    silently. The doctor backup-retention check runs on every
    ``crossmem doctor`` call and will surface a lingering excess as a
    warning, so a missed unlink does not stay invisible.
    """
    parent = path.parent
    if not parent.is_dir():
        return
    prefix = f"{path.name}{BACKUP_PREFIX}"
    backups = sorted(p for p in parent.iterdir() if p.name.startswith(prefix))
    if len(backups) <= keep:
        return
    for old in backups[: len(backups) - keep]:
        try:
            old.unlink()
        except OSError:
            # Filesystem hiccup (lock, transient permission error) — the
            # next call will retry on the same surviving files.
            continue


def backup_config(path: Path) -> Path | None:
    """Copy ``path`` to ``<path>.bak.<timestamp>`` if it exists.

    After the new backup is in place, prune the directory down to
    :data:`BACKUP_RETENTION` newest generations (deletes the oldest
    surplus). The fresh backup is always among the survivors because it
    carries the most recent timestamp.
    """
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + f"{BACKUP_PREFIX}{timestamp()}")
    shutil.copy2(path, backup)
    _prune_backups(path)
    return backup


def load_json_config(path: Path) -> dict[str, Any]:
    """Return parsed JSON from ``path`` or an empty dict if missing/empty."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a JSON object")
    return data


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically using ``Path.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    tmp.write_text(serialized + "\n", encoding="utf-8")
    tmp.replace(path)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Return parsed YAML from ``path`` or an empty dict if missing/empty.

    Mirrors :func:`load_json_config` for connectors whose CLIs (Goose,
    Continue.dev 2.x) store their MCP configuration in YAML. A null
    top-level document (``null``) is treated the same as an empty file
    and yields ``{}``.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping")
    return data


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically using ``Path.replace``.

    Mirrors :func:`atomic_write_json` but emits YAML (``safe_dump`` with
    ``sort_keys=False`` to keep the on-disk key order stable and
    ``allow_unicode=True`` so non-ASCII strings stay readable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    tmp.write_text(serialized, encoding="utf-8")
    tmp.replace(path)


def register_mcp_server(
    path: Path,
    server_cmd: str,
    *,
    servers_key: str = "mcpServers",
) -> None:
    """Add the crossmem entry to ``path``'s ``servers_key`` object.

    Creates parent directories and the file itself if they don't exist.
    Backs up the original file (if any) before writing. ``servers_key``
    defaults to ``mcpServers``; pass another key (e.g. ``context_servers``
    for Zed or ``mcp`` for OpenCode) to target a different top-level slot.

    ``server_cmd`` is split via :func:`shlex.split` so the resulting
    ``command`` is a single executable name and ``args`` carries the
    trailing tokens — the shape MCP clients actually spawn
    (``Popen([command, *args])``). A single-token ``server_cmd`` (e.g.
    ``"crossmem"``) yields ``command=server_cmd`` and ``args=[]``.
    """
    backup_config(path)
    data = load_json_config(path)
    servers = data.get(servers_key)
    if not isinstance(servers, dict):
        servers = {}
    tokens = shlex.split(server_cmd) if server_cmd else []
    command = tokens[0] if tokens else server_cmd
    args = tokens[1:] if tokens else []
    servers[CROSSMEM_KEY] = {
        "command": command,
        "args": args,
        "env": {},
    }
    data[servers_key] = servers
    atomic_write_json(path, data)


def unregister_mcp_server(
    path: Path,
    *,
    servers_key: str = "mcpServers",
) -> None:
    """Remove the crossmem entry from ``path``. No-op if missing."""
    if not path.exists():
        return
    data = load_json_config(path)
    servers = data.get(servers_key)
    if not isinstance(servers, dict) or CROSSMEM_KEY not in servers:
        return
    backup_config(path)
    del servers[CROSSMEM_KEY]
    data[servers_key] = servers
    atomic_write_json(path, data)
