"""Verify that ``import crossmem`` resolves to this repo's ``src/``.

Background: a previous run of ``pip install -e .`` inside a worktree
overwrote the global ``site-packages`` ``.pth`` so that ``import crossmem``
on the host pointed at the worktree instead of the main repo. The pre-push
pytest hook then loaded stale code, coverage was wrong, push was blocked.

This guard is wired into ``.pre-commit-config.yaml`` at ``stages: [pre-push]``
just before pytest. It exits non-zero when the imported package does not live
under ``<repo_root>/src`` and prints the exact fix command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def main() -> int:
    repo_root = _repo_root()
    expected_src = (repo_root / "src").resolve()

    import crossmem

    crossmem_path = Path(crossmem.__file__).resolve()

    if expected_src in crossmem_path.parents:
        return 0

    print(
        f"crossmem resolves to {crossmem_path}, not {expected_src}",
        file=sys.stderr,
    )
    print(f"Fix: pip install -e {repo_root}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
