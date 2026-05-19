"""Inspect and render :class:`CLIConnector` status for ``crossmem status``.

Exposes two pure functions plus a small dataclass:

* :func:`gather_connector_status` collects per-connector facts
  (detected, registered, backup count, config path) into
  :class:`ConnectorStatus` rows.
* :func:`format_status_table` renders the rows as a plain ASCII table
  using only the standard library (no ``tabulate`` dependency).

The two are split so callers (the CLI dispatcher, tests, future JSON
exporters) can consume the structured data without going through the
table renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crossmem.connectors.config_io import BACKUP_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from crossmem.connectors.base import CLIConnector

__all__ = [
    "ConnectorStatus",
    "format_status_table",
    "gather_connector_status",
]


@dataclass(frozen=True)
class ConnectorStatus:
    """One row in the ``crossmem status --connectors`` report.

    ``detected`` and ``registered`` are kept as plain booleans (rather
    than encoding "not applicable" as a third state on ``registered``)
    so the dataclass stays trivially serialisable. The table renderer
    is responsible for showing ``-`` when ``detected`` is ``False``.
    """

    name: str
    detected: bool
    registered: bool
    backups: int
    config_path: Path


def _count_backups(path: Path) -> int:
    """Count ``<path>.bak.*`` siblings.

    Backups created by :mod:`crossmem.connectors.config_io` (and the
    bespoke YAML routines in :mod:`crossmem.connectors.goose` and
    :mod:`crossmem.connectors.continuedev`) follow the suffix pattern
    ``.bak.<timestamp>``. We count those — and only those — so unrelated
    siblings like ``.tmp`` or ``.swp`` do not inflate the number.
    """
    parent = path.parent
    if not parent.is_dir():
        return 0
    prefix = f"{path.name}{BACKUP_PREFIX}"
    return sum(1 for entry in parent.iterdir() if entry.name.startswith(prefix))


def gather_connector_status(
    connectors: Iterable[CLIConnector],
) -> list[ConnectorStatus]:
    """Return :class:`ConnectorStatus` rows for ``connectors`` (input order).

    Each connector is inspected independently — a failure in one (e.g.
    a malformed config file raising deep inside :meth:`is_registered`)
    must not poison the rest of the report. We deliberately do not
    swallow exceptions here: connectors are required to make
    :meth:`is_registered` and :meth:`detect` total (no raises on
    malformed input). If a connector still raises, the bug surfaces.
    """
    rows: list[ConnectorStatus] = []
    for connector in connectors:
        path = connector.config_path()
        rows.append(
            ConnectorStatus(
                name=connector.name(),
                detected=connector.detect(),
                registered=connector.is_registered(),
                backups=_count_backups(path),
                config_path=path,
            )
        )
    return rows


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


# Column ordering for the table renderer. ``name`` first (sorting handle),
# then the two booleans, then the count, and the always-long path last so
# variable widths in ``config_path`` do not push narrow columns around.
_COLUMNS: tuple[str, ...] = (
    "name",
    "detected",
    "registered",
    "backups",
    "config_path",
)


def _row_cells(status: ConnectorStatus) -> tuple[str, ...]:
    return (
        status.name,
        _yes_no(status.detected),
        # ``registered`` is meaningless without a config file on disk —
        # render it as ``-`` so callers (and humans) cannot confuse a
        # missing CLI with a CLI that simply has no crossmem entry yet.
        _yes_no(status.registered) if status.detected else "-",
        str(status.backups),
        str(status.config_path),
    )


def format_status_table(statuses: Sequence[ConnectorStatus]) -> str:
    """Render ``statuses`` as a fixed-width plain ASCII table.

    Column widths are the max of the header label and every value in
    that column. The result is suitable for piping to a file or a less
    pager — no ANSI codes, no borders, just two-space separators which
    keeps the output greppable.
    """
    header = _COLUMNS
    rows = [_row_cells(s) for s in statuses]

    widths = [len(col) for col in header]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _format_row(cells: Sequence[str]) -> str:
        # Two-space separator is enough to keep adjacent columns visually
        # distinct without bloating the row width on a typical terminal.
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_format_row(header)]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines) + "\n"
