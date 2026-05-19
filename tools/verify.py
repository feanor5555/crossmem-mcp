"""On-demand mini-CI for CrossMem.

The pre-push hook is intentionally slim — it only runs the
``editable-install-guard`` and a quick ``ruff check``. The heavy checks
(full pytest with coverage gate, ``pip-audit``, ``check_install_docs``,
install/skills test slice) used to block every push but have moved here
so developers can run them on demand without paying the cost on every
``git push``.

Run everything::

    python tools/verify.py

Skip individual steps when iterating (e.g. while a fix is still in
progress)::

    python tools/verify.py --skip-coverage --skip-audit

Exit code is non-zero as soon as any step fails. By default later steps
are skipped after the first failure; pass ``--keep-going`` to run them
all and see every failure in one shot.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Step:
    """A single verification step.

    ``name`` is used both for the ``--skip-<name>`` CLI flag and in the
    human-readable progress output.
    """

    name: str
    description: str
    command: list[str]


# Order matters: cheap/fast checks first so failures surface quickly.
STEPS: list[Step] = [
    Step(
        name="format",
        description="ruff format --check",
        command=[sys.executable, "-m", "ruff", "format", "--check", "."],
    ),
    Step(
        name="lint",
        description="ruff check",
        command=[sys.executable, "-m", "ruff", "check", "."],
    ),
    Step(
        name="coverage",
        description=(
            "pytest with coverage gate (>=90 percent, "
            "excludes network/benchmark/workflow)"
        ),
        command=[
            sys.executable,
            "-m",
            "pytest",
            "--cov=crossmem",
            "--cov-config=.coveragerc",
            "--cov-fail-under=90",
            "-m",
            "not network and not benchmark and not workflow",
        ],
    ),
    Step(
        name="audit",
        description="pip-audit --strict (declared dependencies)",
        command=[
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--progress-spinner",
            "off",
            ".",
        ],
    ),
    Step(
        name="install-docs",
        description="check install docs (parity / schema / snippet)",
        command=[
            sys.executable,
            "tools/check_install_docs.py",
            "--repo-root",
            ".",
        ],
    ),
    Step(
        name="install-skills-tests",
        description="install-doc and skill tests",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/install",
            "tests/skills",
            "-m",
            "not network and not benchmark",
        ],
    ),
]


def _build_parser(steps: list[Step]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/verify.py",
        description=(
            "On-demand mini-CI: ruff format/check, pytest with coverage, "
            "pip-audit, install-docs check, install/skills test slice."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Run every step even after a failure (default: stop at first failure).",
    )
    for step in steps:
        parser.add_argument(
            f"--skip-{step.name}",
            action="store_true",
            help=f"Skip the {step.name!r} step ({step.description}).",
        )
    return parser


def _run_step(step: Step, runner: Callable[[list[str]], int]) -> tuple[bool, float]:
    start = time.monotonic()
    print(f"==> {step.name}: {step.description}", flush=True)
    rc = runner(step.command)
    duration = time.monotonic() - start
    ok = rc == 0
    status = "OK" if ok else f"FAIL (exit {rc})"
    print(f"<== {step.name}: {status} in {duration:.1f}s", flush=True)
    return ok, duration


def _default_runner(command: list[str]) -> int:
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def run(
    args: argparse.Namespace,
    steps: list[Step],
    runner: Callable[[list[str]], int] = _default_runner,
) -> int:
    """Execute the selected steps and return the overall exit code.

    Exposed for tests so they can plug in a fake runner instead of
    spawning subprocesses.
    """

    failures: list[str] = []
    skipped: list[str] = []
    for step in steps:
        if getattr(args, f"skip_{step.name.replace('-', '_')}", False):
            skipped.append(step.name)
            print(f"-- {step.name}: skipped via --skip-{step.name}", flush=True)
            continue
        ok, _ = _run_step(step, runner)
        if not ok:
            failures.append(step.name)
            if not args.keep_going:
                break

    print()
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        return 1
    print("OK: all selected steps passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser(STEPS)
    args = parser.parse_args(argv)
    return run(args, STEPS)


if __name__ == "__main__":
    raise SystemExit(main())
