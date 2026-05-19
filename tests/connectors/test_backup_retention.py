"""Tests for ``*.bak.<ts>`` retention (task 21.7).

``backup_config`` creates a timestamped backup on every register/unregister
call. Without retention the directory accumulates one extra file per
install run — over the lifetime of a CLI config that adds up. We cap the
on-disk count at :data:`crossmem.connectors.config_io.BACKUP_RETENTION`
generations and prune the oldest entries on each new backup.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from crossmem.connectors.config_io import (
    BACKUP_RETENTION,
    backup_config,
    register_mcp_server,
    unregister_mcp_server,
)

if TYPE_CHECKING:
    from pathlib import Path


def _list_backups(path: Path) -> list[Path]:
    """Return ``<path>.bak.*`` siblings sorted by filename (timestamp)."""
    return sorted(path.parent.glob(f"{path.name}.bak.*"))


def test_backup_retention_default_is_five() -> None:
    """The module-level constant is the documented default of 5."""
    assert BACKUP_RETENTION == 5


def test_backup_config_prunes_oldest_when_exceeding_retention(
    tmp_path: Path,
) -> None:
    """After the 8th backup only the newest 5 remain on disk."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")

    # 7 pre-existing backups + one fresh one from backup_config = 8 total
    # before pruning. After pruning only BACKUP_RETENTION (=5) remain.
    for i in range(7):
        stamp = f"2026-01-01T00-00-{i:02d}.000000Z"
        (tmp_path / f"cfg.json.bak.{stamp}").write_text("{}", encoding="utf-8")

    # Sanity: the 7 pre-existing files are visible before we call
    # backup_config; the helper must observe them when deciding which
    # ones to prune.
    assert len(_list_backups(cfg)) == 7

    backup_config(cfg)

    backups = _list_backups(cfg)
    assert len(backups) == BACKUP_RETENTION
    # The pruned set must be the OLDEST ones — i.e. the surviving files
    # are the lexicographically largest names (timestamps sort
    # chronologically by design).
    surviving_stamps = [b.name for b in backups]
    assert surviving_stamps == sorted(surviving_stamps)[-BACKUP_RETENTION:]


def test_backup_config_keeps_all_when_below_retention(tmp_path: Path) -> None:
    """No pruning happens while the count stays at or under the cap."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")

    # 3 pre-existing backups; backup_config adds a 4th -> still under 5.
    for i in range(3):
        stamp = f"2026-01-01T00-00-{i:02d}.000000Z"
        (tmp_path / f"cfg.json.bak.{stamp}").write_text("{}", encoding="utf-8")

    backup_config(cfg)

    assert len(_list_backups(cfg)) == 4


def test_backup_config_no_prune_when_path_missing(tmp_path: Path) -> None:
    """``backup_config`` on a missing path is a no-op; nothing to prune."""
    cfg = tmp_path / "missing.json"

    result = backup_config(cfg)

    assert result is None
    assert _list_backups(cfg) == []


def test_register_mcp_server_enforces_retention(tmp_path: Path) -> None:
    """End-to-end via ``register_mcp_server`` (the public write path).

    Eight register calls in a row must leave exactly ``BACKUP_RETENTION``
    backup files behind — the first call creates no backup (the config
    did not exist before that), each subsequent call adds one.
    """
    cfg = tmp_path / "cfg.json"

    for _ in range(8):
        register_mcp_server(cfg, "crossmem")
        # Backup timestamps include microseconds but rapid-fire calls in
        # the same process can still collide. Sleep a millisecond between
        # writes so each backup gets a fresh filename.
        time.sleep(0.001)

    assert len(_list_backups(cfg)) == BACKUP_RETENTION


def test_unregister_mcp_server_enforces_retention(tmp_path: Path) -> None:
    """``unregister_mcp_server`` shares the same backup path; same cap."""
    cfg = tmp_path / "cfg.json"
    register_mcp_server(cfg, "crossmem")

    for _ in range(7):
        # Re-register so there is something to unregister each time.
        register_mcp_server(cfg, "crossmem")
        unregister_mcp_server(cfg)
        time.sleep(0.001)

    assert len(_list_backups(cfg)) <= BACKUP_RETENTION
