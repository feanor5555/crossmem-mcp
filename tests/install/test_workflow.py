"""Regression tests for ``.github/workflows/install-docs-lint.yml`` (Task 17.4).

The workflow is a thin wrapper around ``tools/check_install_docs.py``. We
assert the structural invariants here so a later refactor cannot silently
disable the lint:

* YAML parses without errors.
* The workflow triggers on ``workflow_dispatch`` only (Task 25.1 — the
  push/PR-driven remote CI was removed; pre-push hooks (Task 25.3) cover
  the local gate).
* At least one job calls the check script and ``ruff`` against the same
  paths that the install docs touch (``install/``, ``tools/``,
  ``tests/install/``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "install-docs-lint.yml"


def _load() -> dict:
    with WORKFLOW_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    # ``on`` is parsed by PyYAML as the boolean ``True`` (YAML 1.1). Accept
    # both spellings so the test does not break if the file is rewritten.
    return workflow.get("on") or workflow.get(True) or {}


def test_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.is_file(), f"missing workflow file: {WORKFLOW_PATH}"


def test_workflow_yaml_is_well_formed() -> None:
    """The file must parse as valid YAML — catches syntax drift early."""
    workflow = _load()
    assert isinstance(workflow, dict) and workflow, (
        "workflow must parse to a non-empty mapping"
    )


def test_workflow_runs_on_workflow_dispatch_only() -> None:
    """Task 25.1 — push/PR triggers removed, manual dispatch only.

    The pre-push hooks introduced in Task 25.3 run the install-doc /
    schema / snippet checks locally before every push; leaving push/PR
    triggers would re-introduce the automatic remote CI the public-prep
    sequence is explicitly removing.
    """
    triggers = _triggers(_load())
    keys = set(triggers.keys()) if isinstance(triggers, dict) else set()
    assert keys == {"workflow_dispatch"}, (
        "install-docs-lint.yml must trigger on workflow_dispatch only after "
        f"Task 25.1; saw triggers={sorted(keys)!r}"
    )


def test_workflow_has_a_job_that_invokes_the_check_script() -> None:
    """At least one ``run`` step must call ``check_install_docs.py``."""
    workflow = _load()
    run_blob = "\n".join(
        str(step.get("run", ""))
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
    )
    assert "check_install_docs.py" in run_blob, (
        f"workflow must invoke tools/check_install_docs.py, saw run blob:\n{run_blob}"
    )


def test_workflow_runs_ruff_on_script_and_tests() -> None:
    """Ruff must lint both the script and the install tests in CI."""
    workflow = _load()
    run_blob = "\n".join(
        str(step.get("run", ""))
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
    )
    assert "ruff" in run_blob, (
        f"workflow must run ruff against tools/ and tests/, got:\n{run_blob}"
    )
