"""``crossmem uninstall`` — rollback wiring done by ``crossmem install``.

Three responsibilities, mirroring :mod:`crossmem.installer` in reverse:

1. **Filter** the connector list. With ``--cli NAME`` only the matching
   connector is considered; without it every shipped connector is
   inspected. Unknown ``--cli`` values raise :class:`UnknownConnectorError`
   so the CLI can surface the list of valid names.
2. **Unregister** every detected connector. ``unregister()`` is a no-op
   when the entry is already missing — see ``connectors/config_io.py``
   — so the operation is idempotent.
3. **Purge** the on-disk ``~/.crossmem`` directory only when both
   ``purge=True`` and ``confirm=True`` are passed. The double-flag
   gate exists so an accidental ``crossmem uninstall --purge`` (without
   the explicit ``--yes`` confirmation) does NOT wipe the user's DB.

Like the installer, the connector list and home directory are
dependency-injected so tests can exercise the flow without touching
real CLI configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

from crossmem import installer

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossmem.connectors.base import CLIConnector

__all__ = [
    "UninstallResult",
    "UnknownConnectorError",
    "uninstall",
]


class UnknownConnectorError(ValueError):
    """Raised when ``--cli NAME`` does not match any shipped connector.

    Carries ``requested`` (the user input) and ``known`` (the sorted
    list of valid connector names) so the CLI can render a helpful
    error without re-walking the connector registry itself.
    """

    def __init__(self, requested: str, known: list[str]) -> None:
        self.requested = requested
        self.known = known
        super().__init__(
            f"unknown --cli {requested!r}. Known CLIs: {', '.join(known)}."
        )


@dataclass(frozen=True)
class UninstallResult:
    """Summary of what :func:`uninstall` did.

    * ``unregistered_clis`` — CLI ``name()`` values that had their
      ``crossmem`` entry removed (i.e. ``detect()`` returned True and
      ``unregister()`` was called).
    * ``skipped_clis`` — CLI ``name()`` values whose ``detect()``
      returned False; nothing was touched.
    * ``purged_home`` — ``True`` when ``~/.crossmem`` was actually
      removed during this call. ``False`` when purge was not requested,
      not confirmed, or the directory was already missing.
    * ``home_path`` — the ``~/.crossmem`` path considered for purge.
    """

    unregistered_clis: list[str]
    skipped_clis: list[str]
    purged_home: bool
    home_path: Path


def _print(msg: str) -> None:
    """Single funnel for progress output (keeps tests targeting one channel)."""
    print(msg)


def _materialise_connectors(
    connectors: list[CLIConnector] | None,
) -> list[CLIConnector]:
    """Return ``connectors`` if supplied, else call the canonical factory.

    Delegates to :func:`crossmem.installer.instantiate_connectors` so the
    uninstaller's default connector set never drifts from the installer.
    """
    if connectors is not None:
        return connectors
    return installer.instantiate_connectors()


def _filter_by_name(
    connectors: list[CLIConnector],
    cli: str | None,
) -> list[CLIConnector]:
    """Return only the connector whose ``name()`` matches ``cli`` (or all)."""
    if cli is None:
        return connectors
    matches = [c for c in connectors if c.name() == cli]
    if not matches:
        known = sorted(c.name() for c in connectors)
        raise UnknownConnectorError(cli, known)
    return matches


def _unregister_detected(
    connectors: list[CLIConnector],
) -> tuple[list[str], list[str]]:
    """Run ``unregister()`` on every detected connector.

    Returns ``(unregistered, skipped)`` — the first lists CLI names
    that had the crossmem entry removed, the second lists CLIs whose
    ``detect()`` returned False. Skipped connectors are also printed
    so the user sees the full audit trail.
    """
    unregistered: list[str] = []
    skipped: list[str] = []
    for connector in connectors:
        name = connector.name()
        if not connector.detect():
            _print(f"  - {name}: not detected, skipping")
            skipped.append(name)
            continue
        config_path = connector.config_path()
        connector.unregister()
        _print(f"  - {name}: removed crossmem entry from {config_path}")
        unregistered.append(name)
    return unregistered, skipped


def _maybe_purge_home(home: Path, *, purge: bool, confirm: bool) -> bool:
    """Remove ``home/.crossmem`` only when both ``purge`` and ``confirm``.

    Returns ``True`` if the directory was actually removed. Prints a
    one-line trace either way so the user sees what happened.
    """
    target = home / ".crossmem"
    if not purge:
        return False
    if not confirm:
        _print(f"  - purge: refused (re-run with --yes to remove {target})")
        return False
    if not target.exists():
        _print(f"  - purge: {target} already absent, nothing to remove")
        return False
    rmtree(target)
    _print(f"  - purge: removed {target}")
    return True


def uninstall(
    *,
    connectors: list[CLIConnector] | None = None,
    cli: str | None = None,
    purge: bool = False,
    confirm: bool = False,
    home_factory: Callable[[], Path] | None = None,
) -> UninstallResult:
    """Run the ``crossmem uninstall`` flow and return an :class:`UninstallResult`.

    Parameters are dependency-injected for testability:

    * ``connectors`` — explicit list of :class:`CLIConnector` instances.
      ``None`` (the default) uses :data:`crossmem.installer.ALL_CONNECTORS`.
    * ``cli`` — optional connector ``name()`` filter. Unknown values
      raise :class:`UnknownConnectorError`.
    * ``purge`` — when ``True``, attempt to remove ``~/.crossmem``.
      Always combined with ``confirm``: without ``confirm`` the purge
      is refused and a hint is printed.
    * ``confirm`` — must be ``True`` together with ``purge`` for the
      directory to be removed.
    * ``home_factory`` — zero-arg callable returning the home directory
      to operate on. ``None`` uses :func:`pathlib.Path.home`.
    """
    home = home_factory() if home_factory is not None else Path.home()
    cli_objects = _materialise_connectors(connectors)
    selected = _filter_by_name(cli_objects, cli)
    _print("Removing crossmem MCP wiring...")
    unregistered, skipped = _unregister_detected(selected)
    purged = _maybe_purge_home(home, purge=purge, confirm=confirm)
    _print(f"Done. Removed from {len(unregistered)} CLI(s), skipped {len(skipped)}.")
    return UninstallResult(
        unregistered_clis=unregistered,
        skipped_clis=skipped,
        purged_home=purged,
        home_path=home / ".crossmem",
    )
