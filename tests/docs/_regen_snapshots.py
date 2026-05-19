"""Regenerate ``tests/docs/snapshots/<cli>.md`` for every connector.

Run with::

    PYTHONPATH=src python -m tests.docs._regen_snapshots

Only invoke this when an intentional renderer change has happened and
the snapshot test now fails. Commits that touch the snapshots must
include the renderer change in the same patch so reviewers can spot
unintended drift.
"""

from __future__ import annotations

from pathlib import Path

from crossmem.docs.install_template import CONNECTOR_REGISTRY, render_install_doc

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for cli_name in sorted(CONNECTOR_REGISTRY):
        target = SNAPSHOT_DIR / f"{cli_name}.md"
        rendered = render_install_doc(cli_name)
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":  # pragma: no cover - utility script
    main()
