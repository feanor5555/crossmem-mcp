"""Behaviour tests for ``tools/verify.py`` (on-demand mini-CI)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_PATH = REPO_ROOT / "tools" / "verify.py"


def _load_verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_oncemore", VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before ``exec_module`` so ``@dataclass`` can resolve
    # ``cls.__module__`` via ``sys.modules`` (Python 3.12+ behaviour).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_all_steps_pass_returns_zero() -> None:
    verify = _load_verify()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    args = verify._build_parser(verify.STEPS).parse_args([])
    rc = verify.run(args, verify.STEPS, runner=runner)
    assert rc == 0
    assert len(calls) == len(verify.STEPS), (
        f"runner must be called once per step, got {len(calls)} for "
        f"{len(verify.STEPS)} steps"
    )


def test_failure_stops_at_first_step_by_default() -> None:
    verify = _load_verify()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 1  # every step fails

    args = verify._build_parser(verify.STEPS).parse_args([])
    rc = verify.run(args, verify.STEPS, runner=runner)
    assert rc == 1
    assert len(calls) == 1, (
        f"default mode must stop at the first failure, ran {len(calls)} steps"
    )


def test_keep_going_runs_every_step_after_failure() -> None:
    verify = _load_verify()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 1

    args = verify._build_parser(verify.STEPS).parse_args(["--keep-going"])
    rc = verify.run(args, verify.STEPS, runner=runner)
    assert rc == 1
    assert len(calls) == len(verify.STEPS), (
        f"--keep-going must run every step, ran {len(calls)} of {len(verify.STEPS)}"
    )


def test_skip_flag_skips_named_step() -> None:
    verify = _load_verify()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    args = verify._build_parser(verify.STEPS).parse_args(["--skip-coverage"])
    rc = verify.run(args, verify.STEPS, runner=runner)
    assert rc == 0
    # Every step except ``coverage`` should have been invoked.
    assert len(calls) == len(verify.STEPS) - 1
    for cmd in calls:
        assert "--cov=crossmem" not in cmd, (
            f"--skip-coverage must skip the coverage pytest step, got {cmd!r}"
        )


def test_every_step_has_a_skip_flag() -> None:
    """Each step in ``STEPS`` must be skippable via ``--skip-<name>``."""
    verify = _load_verify()
    parser = verify._build_parser(verify.STEPS)
    for step in verify.STEPS:
        # ``parse_args`` will SystemExit if the flag is unknown.
        ns = parser.parse_args([f"--skip-{step.name}"])
        attr = f"skip_{step.name.replace('-', '_')}"
        assert getattr(ns, attr) is True, (
            f"--skip-{step.name} must set namespace attribute {attr!r} to True"
        )
