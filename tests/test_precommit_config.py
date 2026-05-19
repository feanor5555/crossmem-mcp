"""Regression tests for `.pre-commit-config.yaml` hook-stage layout.

The pre-push hook is intentionally slim — it must only run the cheap
checks (``editable-install-guard`` and ``ruff check``). The heavy
checks (full pytest with coverage, ``pip-audit``, ``check-install-docs``,
install/skills test slice) live in ``tools/verify.py`` and are run on
demand instead of blocking every push.

The split is:

* ruff hooks run on pre-commit (sub-second formatting/linting).
* ``editable-install-guard`` and ``ruff check`` run on pre-push.
* ``conventional-commit`` runs on commit-msg.

This test catches future regressions in either direction:

* Re-adding pytest / pip-audit / check-install-docs /
  install-skills-tests to the pre-push stage (slows every push down).
* Dropping those checks from ``tools/verify.py`` (silently disables the
  on-demand mini-CI).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"
VERIFY_PATH = REPO_ROOT / "tools" / "verify.py"


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_hooks(config: dict):
    for repo in config.get("repos", []):
        for hook in repo.get("hooks", []):
            yield repo.get("repo", "local"), hook


def _find_hook(config: dict, hook_id: str) -> dict:
    for _repo, hook in _iter_hooks(config):
        if hook.get("id") == hook_id:
            return hook
    raise AssertionError(f"hook {hook_id!r} not found in .pre-commit-config.yaml")


def _hook_ids_for_stage(config: dict, stage: str) -> set[str]:
    return {
        hook["id"]
        for _repo, hook in _iter_hooks(config)
        if stage in (hook.get("stages") or [])
    }


def test_default_install_hook_types_contains_all_three_stages() -> None:
    """`pre-commit install` must auto-register pre-commit, pre-push, commit-msg."""
    config = _load_config()
    types = config.get("default_install_hook_types")
    assert types is not None, "default_install_hook_types missing from config"
    assert set(types) >= {"pre-commit", "pre-push", "commit-msg"}, (
        f"default_install_hook_types must include all three stages, got {types!r}"
    )


def test_ruff_hooks_have_no_explicit_stage_or_include_pre_commit() -> None:
    """ruff hooks must fire on pre-commit (either by default or explicitly)."""
    config = _load_config()
    ruff_ids = {"ruff", "ruff-format"}
    found: set[str] = set()
    for _repo, hook in _iter_hooks(config):
        hook_id = hook.get("id")
        if hook_id not in ruff_ids:
            continue
        found.add(hook_id)
        stages = hook.get("stages")
        if stages is not None:
            assert "pre-commit" in stages, (
                f"ruff hook {hook_id!r} has explicit stages {stages!r} "
                "without pre-commit"
            )
    assert found == ruff_ids, f"missing ruff hooks in config: {ruff_ids - found}"


def test_conventional_commit_hook_runs_on_commit_msg() -> None:
    """conventional-commit must validate commit messages, not gate code."""
    hook = _find_hook(_load_config(), "conventional-commit")
    assert hook.get("stages") == ["commit-msg"], (
        "conventional-commit hook stages must be ['commit-msg'], "
        f"got {hook.get('stages')!r}"
    )


# ---------------------------------------------------------------------------
# Slim pre-push: only the cheap guards run here.
# ---------------------------------------------------------------------------


def test_editable_install_guard_runs_on_pre_push() -> None:
    hook = _find_hook(_load_config(), "editable-install-guard")
    assert hook.get("stages") == ["pre-push"], (
        "editable-install-guard hook stages must be ['pre-push'], "
        f"got {hook.get('stages')!r}"
    )


def test_ruff_check_runs_on_pre_push() -> None:
    """A lightweight ``ruff check`` gates every push (sub-second)."""
    hook = _find_hook(_load_config(), "ruff-check")
    assert hook.get("stages") == ["pre-push"], (
        f"ruff-check hook stages must be ['pre-push'], got {hook.get('stages')!r}"
    )
    entry = str(hook.get("entry", ""))
    assert "ruff" in entry and "check" in entry, (
        f"ruff-check hook entry must invoke ruff check, got {entry!r}"
    )


def test_heavy_checks_are_not_in_pre_push() -> None:
    """Heavy checks must NOT block every push — they live in tools/verify.py.

    ``pytest`` (full suite), ``pip-audit``, ``check-install-docs`` and
    ``install-skills-tests`` used to run on pre-push but were moved to
    on-demand mini-CI.
    """
    pre_push_ids = _hook_ids_for_stage(_load_config(), "pre-push")
    forbidden = {
        "pytest",
        "pip-audit",
        "check-install-docs",
        "install-skills-tests",
    }
    leaked = forbidden & pre_push_ids
    assert not leaked, (
        f"these hooks must run on demand via tools/verify.py, not pre-push: "
        f"{sorted(leaked)}"
    )


# ---------------------------------------------------------------------------
# tools/verify.py — the heavy checks moved here.
# ---------------------------------------------------------------------------


def test_verify_script_exists_and_covers_all_moved_checks() -> None:
    """tools/verify.py must still invoke every heavy check that left pre-push.

    The contract is "moved, not deleted": each check needs to be reachable
    on demand. The assertion is intentionally loose (substring match in
    the file body) so the test does not over-constrain the implementation
    of ``verify.py`` itself.
    """
    assert VERIFY_PATH.is_file(), f"missing on-demand mini-CI script: {VERIFY_PATH}"
    body = VERIFY_PATH.read_text(encoding="utf-8")
    required_markers = [
        # full pytest with coverage gate
        "--cov=crossmem",
        "--cov-fail-under=90",
        # pip-audit
        "pip_audit",
        "--strict",
        # check_install_docs
        "tools/check_install_docs.py",
        # install/skills test slice
        "tests/install",
        "tests/skills",
        # ruff format/check
        "ruff",
        "format",
        "check",
    ]
    missing = [marker for marker in required_markers if marker not in body]
    assert not missing, (
        f"tools/verify.py is missing markers for on-demand checks: {missing}"
    )
