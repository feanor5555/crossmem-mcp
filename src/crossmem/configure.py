"""Backend configuration — ``crossmem configure``.

Writes the active backend choice and optional connection details to
``~/.crossmem/config.toml``. Supported backends:

* ``sqlite`` (default, bundled)
* ``chroma`` (requires ``crossmem[chroma]``)
* ``qdrant`` (requires ``crossmem[qdrant]``)

When the active backend changes, the user is offered a one-shot migration
via export/import. The migration itself is delegated to :func:`_run_migration`
which today raises :class:`NotImplementedError` — users perform the
migration manually with ``crossmem export | configure | import``. Tests
monkeypatch :func:`_run_migration` to verify the wiring.

**Order of operations**: when a migration is requested, the migration
helper is called *before* ``config.toml`` is written. If the helper
raises (e.g. the default :class:`NotImplementedError`), the config file
stays untouched — so a user who follows the error hint ("export, then
configure, then import") exports against the *current* backend that
still holds the data, not against a freshly switched, empty target.

This module also exposes :func:`build_backend`, the central factory that
turns a loaded config into a concrete :class:`VectorStoreBase`. Both
``server.main()`` and ``cli.build_default_store()`` route through it so the
``[backend]`` section of ``config.toml`` actually drives runtime behaviour.

The config is intentionally flat and small enough that we serialize by hand
rather than pulling in ``tomli-w``. Reading uses the stdlib :mod:`tomllib`
(Python 3.11+).

**Variant (a) — typed rejection over runtime dep** (Task 26.16): The hand-
rolled writer :func:`_dump_toml` only supports string values inside one-level
tables. Earlier the code relied on a duck-typed ``value.replace(...)`` call,
which raised :class:`AttributeError` for non-strings — a misleading error
message that named ``replace`` instead of the actual contract violation.
The hardened writer now raises :class:`TypeError` with a clear message
(``configure: only string values supported, got <type>``) for any non-string
section, key, or value. We chose variant (a) over pulling ``tomli-w`` into
the runtime dependency graph because: (1) the writer handles a single flat
table with three known string fields, so a 1-line precondition is cheaper
than an extra wheel; (2) the optional-backend story already adds enough
install surface (``crossmem[chroma]``, ``crossmem[qdrant]``); (3) any future
schema growth that needs TOML types beyond strings should switch to
``tomli-w`` wholesale, not patch this writer further.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomllib

if TYPE_CHECKING:
    from crossmem.backends.base import VectorStoreBase

__all__ = [
    "SUPPORTED_BACKENDS",
    "ConfigureResult",
    "build_backend",
    "config_path",
    "configure",
    "load_config",
]

SUPPORTED_BACKENDS: tuple[str, ...] = ("sqlite", "chroma", "qdrant")
_DEFAULT_BACKEND = "sqlite"


@dataclass(frozen=True)
class ConfigureResult:
    """What ``configure()`` reports back to its caller (the CLI)."""

    path: Path
    previous_backend: str
    new_backend: str
    migrated: bool


def config_path() -> Path:
    """Return the path to ``config.toml`` (without creating it)."""
    return Path.home() / ".crossmem" / "config.toml"


def load_config() -> dict[str, Any]:
    """Load the current config, or return the default sqlite config."""
    path = config_path()
    if not path.exists():
        return {"backend": {"name": _DEFAULT_BACKEND}}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def configure(
    backend: str,
    url: str | None = None,
    api_key: str | None = None,
    migrate: bool = False,
) -> ConfigureResult:
    """Set the active backend.

    Args:
        backend: One of :data:`SUPPORTED_BACKENDS`.
        url: Backend URL (chroma/qdrant only). ``None`` -> omitted.
        api_key: API key for the remote backend. ``None`` -> omitted.
        migrate: If ``True`` and the backend actually changes, run the
            migration without prompting. If ``False`` and the backend
            changes and a previous config exists, the user is asked
            interactively (``y`` runs the migration, anything else
            skips it).

    Returns:
        A :class:`ConfigureResult` describing the previous and new
        backend plus whether a migration was performed.

    Raises:
        ValueError: If ``backend`` is not in :data:`SUPPORTED_BACKENDS`.
    """
    if backend not in SUPPORTED_BACKENDS:
        msg = (
            f"unsupported backend {backend!r}; "
            f"choose one of {', '.join(SUPPORTED_BACKENDS)}"
        )
        raise ValueError(msg)

    path = config_path()
    had_previous_file = path.exists()
    previous = load_config()["backend"].get("name", _DEFAULT_BACKEND)

    # Run the migration FIRST so a failure (the default
    # NotImplementedError) leaves config.toml untouched. Otherwise the
    # backend pointer would already point at the new, empty backend by
    # the time the user follows the "export, then configure, then
    # import" hint — and the export would run against the wrong side.
    migrated = False
    backend_changed = previous != backend
    should_migrate = (
        backend_changed
        and had_previous_file
        and (migrate or _prompt_migration(previous, backend))
    )
    if should_migrate:
        _run_migration(previous, backend)
        migrated = True

    section: dict[str, str] = {"name": backend}
    if url is not None:
        section["url"] = url
    if api_key is not None:
        section["api_key"] = api_key

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml({"backend": section}), encoding="utf-8")
    # The config may carry an ``api_key`` for the active remote backend.
    # On Linux/Mac restrict to owner read/write only (cross-platform rule #4
    # in CLAUDE.md). On Windows POSIX modes are a no-op — skip the chmod
    # entirely rather than relying on it being silently ignored.
    if sys.platform != "win32":
        os.chmod(path, 0o600)

    return ConfigureResult(
        path=path,
        previous_backend=previous,
        new_backend=backend,
        migrated=migrated,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _prompt_migration(previous: str, new: str) -> bool:
    """Ask the user whether to migrate existing data."""
    answer = (
        input(f"Migrate existing data from {previous} to {new}? [y/N]: ")
        .strip()
        .lower()
    )
    return answer == "y"


def _run_migration(previous: str, new: str) -> None:
    """Migrate data between backends via export/import.

    Automatic migration is intentionally unimplemented: a real round-trip
    needs both the source and destination backend live at once (which forces
    both optional dependencies into the install graph) and the export
    format already supports the manual workflow. Because the surrounding
    :func:`configure` calls this helper *before* writing ``config.toml``,
    a raised ``NotImplementedError`` leaves the previous backend active —
    the export in the hint below runs against the backend that still has
    the data, not against the new empty target. Users perform the migration
    explicitly with::

        crossmem export --path backup.zip           # exports from <previous>
        crossmem configure --backend <new>          # flip the pointer
        crossmem import --path backup.zip           # imports into <new>

    Tests monkeypatch this function to verify the wiring of the surrounding
    code path; production callers see :class:`NotImplementedError` with the
    same guidance.
    """
    msg = (
        f"automatic migration from {previous} to {new} is not implemented; "
        f"run `crossmem export --path backup.zip` (exports from {previous}), "
        f"then `crossmem configure --backend {new}` to flip the pointer, "
        "then `crossmem import --path backup.zip` instead"
    )
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class BackendConfigError(RuntimeError):
    """Raised when ``config.toml`` selects a backend that cannot be built.

    Carries a user-facing message that names the missing optional extra
    (``crossmem[chroma]`` / ``crossmem[qdrant]``) so the CLI can surface it
    without inspecting :class:`ImportError` internals.
    """


def build_backend(
    config: dict[str, Any] | None = None,
    *,
    sqlite_path: Path | None = None,
) -> VectorStoreBase:
    """Instantiate the active backend described by ``config``.

    ``config`` is the dict returned by :func:`load_config`. When ``None``
    (the common case for the MCP server boot path) the on-disk config is
    loaded here so callers don't have to repeat themselves.

    ``sqlite_path`` is only consulted when the active backend is SQLite —
    it lets the server and CLI route the DB to ``CROSSMEM_DB_PATH`` or a
    test-controlled tmp dir without leaking that detail into TOML.

    Raises:
        BackendConfigError: When the active backend is ``chroma``/``qdrant``
            and the optional client dependency is not installed, or when
            the config names an unknown backend.
    """
    cfg = config if config is not None else load_config()
    section = cfg.get("backend", {}) or {}
    name = section.get("name", _DEFAULT_BACKEND)

    if name == "sqlite":
        from crossmem.backends.sqlite_backend import SQLiteBackend

        path = sqlite_path if sqlite_path is not None else _default_sqlite_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteBackend(path)

    if name == "chroma":
        try:
            from crossmem.backends.chroma_backend import ChromaBackend
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in tests via monkeypatch
            msg = (
                "backend 'chroma' is configured but chromadb is not installed. "
                "Install it with: pip install crossmem[chroma]"
            )
            raise BackendConfigError(msg) from exc
        url = section.get("url")
        return ChromaBackend(path=url)

    if name == "qdrant":
        try:
            from crossmem.backends.qdrant_backend import QdrantBackend
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in tests via monkeypatch
            msg = (
                "backend 'qdrant' is configured but qdrant-client is not installed. "
                "Install it with: pip install crossmem[qdrant]"
            )
            raise BackendConfigError(msg) from exc
        return QdrantBackend(url=section.get("url"), api_key=section.get("api_key"))

    msg = (
        f"unknown backend {name!r} in {config_path()}; "
        f"expected one of {', '.join(SUPPORTED_BACKENDS)}"
    )
    raise BackendConfigError(msg)


def _default_sqlite_path() -> Path:
    """Return the default SQLite DB path, honouring ``CROSSMEM_DB_PATH``.

    Kept in this module (instead of importing from ``server``) so the
    factory has no dependency on the MCP layer.
    """
    override = os.environ.get("CROSSMEM_DB_PATH")
    if override:
        return Path(override)
    return Path.home() / ".crossmem" / "knowledge.db"


def _dump_toml(data: dict[str, dict[str, str]]) -> str:
    """Tiny TOML writer for our flat ``[section] key = "value"`` schema.

    Only string section names, string keys, and string values inside
    one-level-deep tables are supported. That is exactly what we write:
    ``[backend]`` with ``name``/``url``/``api_key``. Double quotes inside
    strings are escaped (``\\"``); backslashes are too.

    Raises:
        TypeError: When a section name, key, or value is not a :class:`str`.
            Without this guard, non-string values fell through to
            ``str.replace`` and surfaced as a misleading ``AttributeError``
            naming ``replace`` instead of the contract violation.
    """
    lines: list[str] = []
    for section, fields in data.items():
        if not isinstance(section, str):
            msg = (
                "configure: only string values supported, "
                f"got {type(section).__name__} for section name"
            )
            raise TypeError(msg)
        lines.append(f"[{section}]")
        for key, value in fields.items():
            if not isinstance(key, str):
                msg = (
                    "configure: only string values supported, "
                    f"got {type(key).__name__} for key"
                )
                raise TypeError(msg)
            if not isinstance(value, str):
                msg = (
                    "configure: only string values supported, "
                    f"got {type(value).__name__} for {section}.{key}"
                )
                raise TypeError(msg)
            lines.append(f"{key} = {_quote(value)}")
        lines.append("")
    return "\n".join(lines)


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
