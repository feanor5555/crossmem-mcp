"""Regression guard: the sync stack must stay removed (Task 23.1).

Both an import-level check (``crossmem.sync`` does not exist) and a grep
over ``src/`` and ``tests/`` for the public sync identifiers. The grep
prevents accidental re-introduction via copy-paste; the import probe
keeps ``import crossmem.sync`` and ``SyncEngine`` from sneaking back in.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"

# Identifiers that named the removed sync stack. If any of these reappear in
# ``src/`` or ``tests/`` it is almost certainly a regression — the negative
# test in this file fails so the offending change cannot land.
_FORBIDDEN = (
    "crossmem.sync",
    "SyncEngine",
    "_build_sync_engine",
)


def test_crossmem_sync_module_does_not_import() -> None:
    """``import crossmem.sync`` must raise ``ModuleNotFoundError``."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("crossmem.sync")


@pytest.mark.parametrize("identifier", _FORBIDDEN)
def test_no_sync_identifier_remains(identifier: str) -> None:
    """Grep ``src/`` and ``tests/`` for the removed sync identifier."""
    pattern = re.compile(re.escape(identifier))
    offenders: list[str] = []
    for root in (_SRC_DIR, _TESTS_DIR):
        for path in root.rglob("*.py"):
            # The regression guard itself legitimately mentions the names.
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(_PROJECT_ROOT)))
    assert not offenders, f"{identifier} still appears in: {offenders}"
