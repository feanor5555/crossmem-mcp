"""Regression tests for GitHub Actions workflow configuration.

Four workflows are mandated by ``CLAUDE.md`` (section ``## Testbarkeit`` ->
``### CI``) and the public-prep contract (Task 25.1):

* ``ci.yml`` — matrix lint + test + pip-audit. After Task 25.1 the workflow
  no longer runs on ``push``/``pull_request``; it is ``workflow_dispatch``
  only. Pre-push hooks (Task 25.3) cover the local gate; CI fires manually
  when needed.
* ``install-docs-lint.yml`` — install-doc parity / schema / snippet lint.
  Also ``workflow_dispatch`` only after Task 25.1; the same pre-push hooks
  exercise its checks locally.
* ``nightly.yml`` — full integration run against ChromaDB and Qdrant served
  as Docker services (ubuntu + py3.12). Unchanged by Task 25.1.
* ``release.yml`` — on ``v*`` tags, build the distribution and publish to PyPI
  via Trusted Publishers (OIDC) with Sigstore signatures. Unchanged by
  Task 25.1.

These tests assert the structural invariants. Any drift (missing job, wrong
trigger, missing OIDC permission, missing service container) breaks the
contract that the CI pipeline gives us and must fail loudly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Task 22.8 — these tests assert static shape of ``.github/workflows/*.yml``
# files. They exercise only PyYAML + Path I/O, so the ``crossmem`` modules
# they "touch" (none, in fact) get measured by every other test. Counting
# them toward ``--cov-fail-under=90`` therefore inflates the coverage
# signal without adding real protection — and the gate occasionally
# tripped when an unrelated refactor moved lines around. Mark the whole
# module ``workflow`` so the default ``pytest -m "not workflow"`` run
# omits them; CI invokes ``pytest -m workflow`` as its own job.
pytestmark = pytest.mark.workflow

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CI_PATH = WORKFLOWS_DIR / "ci.yml"
INSTALL_DOCS_LINT_PATH = WORKFLOWS_DIR / "install-docs-lint.yml"
NIGHTLY_PATH = WORKFLOWS_DIR / "nightly.yml"
RELEASE_PATH = WORKFLOWS_DIR / "release.yml"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    # ``on`` is parsed by PyYAML as the boolean True (YAML 1.1 spec). Accept
    # both spellings so the test does not break if the file is rewritten.
    return workflow.get("on") or workflow.get(True) or {}


def _trigger_keys(workflow: dict) -> set[str]:
    """Return the set of trigger names declared under ``on:``.

    ``workflow_dispatch:`` may appear either as a mapping value (with inputs)
    or as a bare key (no inputs). Both forms collapse to the same key here.
    """
    triggers = _triggers(workflow)
    if isinstance(triggers, dict):
        return set(triggers.keys())
    if isinstance(triggers, list):
        return set(triggers)
    if isinstance(triggers, str):
        return {triggers}
    return set()


# ---------------------------------------------------------------------------
# ci.yml — workflow_dispatch only (Task 25.1)
# ---------------------------------------------------------------------------


def test_ci_workflow_exists() -> None:
    assert CI_PATH.is_file(), f"missing workflow file: {CI_PATH}"


def test_ci_uses_workflow_dispatch_only() -> None:
    """Task 25.1 — ci.yml runs on manual dispatch only, no push/PR triggers.

    The public-prep contract moves the regular CI gate to local pre-push
    hooks (Task 25.3). Leaving push/PR triggers would re-introduce the
    automatic remote CI we are explicitly removing.
    """
    keys = _trigger_keys(_load(CI_PATH))
    assert keys == {"workflow_dispatch"}, (
        "ci.yml must trigger on workflow_dispatch only after Task 25.1; "
        f"saw triggers={sorted(keys)!r}"
    )


# ---------------------------------------------------------------------------
# install-docs-lint.yml — workflow_dispatch only (Task 25.1)
# ---------------------------------------------------------------------------


def test_install_docs_lint_workflow_exists() -> None:
    assert INSTALL_DOCS_LINT_PATH.is_file(), (
        f"missing workflow file: {INSTALL_DOCS_LINT_PATH}"
    )


def test_install_docs_lint_uses_workflow_dispatch_only() -> None:
    """Task 25.1 — install-docs-lint.yml drops push/PR triggers.

    Same rationale as ci.yml: the install-doc / skill / schema checks move
    to pre-push hooks (Task 25.3). Remote CI fires manually when needed.
    """
    keys = _trigger_keys(_load(INSTALL_DOCS_LINT_PATH))
    assert keys == {"workflow_dispatch"}, (
        "install-docs-lint.yml must trigger on workflow_dispatch only after "
        f"Task 25.1; saw triggers={sorted(keys)!r}"
    )


# ---------------------------------------------------------------------------
# nightly.yml
# ---------------------------------------------------------------------------


def test_nightly_workflow_exists() -> None:
    assert NIGHTLY_PATH.is_file(), f"missing workflow file: {NIGHTLY_PATH}"


def test_nightly_runs_on_schedule_and_manual() -> None:
    """Nightly must run on a daily cron and allow manual dispatch."""
    triggers = _triggers(_load(NIGHTLY_PATH))
    assert "schedule" in triggers, "nightly must define a cron schedule"
    schedules = triggers["schedule"]
    assert schedules and isinstance(schedules, list)
    assert all("cron" in entry for entry in schedules), (
        f"each schedule entry needs a 'cron' key, got {schedules!r}"
    )
    assert "workflow_dispatch" in triggers, (
        "nightly must allow manual runs via workflow_dispatch"
    )


def test_nightly_runs_on_ubuntu_with_python_312() -> None:
    workflow = _load(NIGHTLY_PATH)
    jobs = workflow.get("jobs", {})
    assert jobs, "nightly workflow has no jobs"
    for name, job in jobs.items():
        runs_on = job.get("runs-on", "")
        assert "ubuntu" in runs_on, (
            f"job {name!r} must run on ubuntu (per CLAUDE.md), got {runs_on!r}"
        )
    # At least one job must explicitly request Python 3.12.
    found_py312 = False
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/setup-python"):
                version = str(step.get("with", {}).get("python-version", ""))
                if version == "3.12":
                    found_py312 = True
    assert found_py312, "nightly must pin python-version to 3.12"


def test_nightly_provides_chroma_and_qdrant_services() -> None:
    """Both optional backends must be available as Docker service containers."""
    workflow = _load(NIGHTLY_PATH)
    images: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for service in (job.get("services") or {}).values():
            image = service.get("image", "")
            images.append(image)
    blob = " ".join(images).lower()
    assert "chroma" in blob, f"nightly is missing a ChromaDB service, got {images!r}"
    assert "qdrant" in blob, f"nightly is missing a Qdrant service, got {images!r}"


def test_nightly_installs_optional_backend_extras() -> None:
    """The integration job must install both [chroma] and [qdrant] extras."""
    workflow = _load(NIGHTLY_PATH)
    run_commands: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                run_commands.append(str(step["run"]))
    blob = "\n".join(run_commands)
    # Accept any form of extras declaration: ``[chroma,qdrant]``, ``[chroma]``
    # + ``[qdrant]``, or separate ``pip install`` calls. Intent: both backend
    # clients reach the runner.
    assert "chroma" in blob and "qdrant" in blob, (
        "nightly must install crossmem with both [chroma] and [qdrant] extras, "
        f"saw run blocks: {run_commands!r}"
    )
    assert 'pip install ".[' in blob or "pip install .[" in blob, (
        "nightly must install the project with extras via 'pip install .[...]', "
        f"saw run blocks: {run_commands!r}"
    )


# ---------------------------------------------------------------------------
# nightly.yml — Docker end-to-end smoke job (Task 13.3)
# ---------------------------------------------------------------------------


def _docker_smoke_job(workflow: dict) -> dict:
    """Return the docker-smoke job from ``nightly.yml``.

    The job name is conventional but not load-bearing — we search for any
    job whose steps build a wheel and run ``pytest`` against
    ``tests/smoke``. If multiple match, the first one wins (there should
    only ever be one).
    """
    for name, job in workflow.get("jobs", {}).items():
        run_blob = "\n".join(str(step.get("run", "")) for step in job.get("steps", []))
        if "python -m build" in run_blob and "tests/smoke" in run_blob:
            return {"name": name, **job}
    raise AssertionError(
        "nightly is missing a docker-smoke job that builds a wheel and "
        "runs tests/smoke (Task 13.3)"
    )


def test_nightly_has_docker_smoke_job() -> None:
    """Nightly must include a job that exercises a freshly built wheel.

    Task 13.3 demands a clean-room install: build a wheel, install it
    inside ``ubuntu:24.04``, run ``crossmem doctor`` and a store->query
    round-trip. This test asserts the structural shape of that job so a
    later refactor cannot quietly drop the validation.
    """
    workflow = _load(NIGHTLY_PATH)
    job = _docker_smoke_job(workflow)
    assert "ubuntu" in job.get("runs-on", ""), (
        f"docker-smoke job must run on ubuntu, got {job.get('runs-on')!r}"
    )


def test_nightly_docker_smoke_builds_wheel_before_pytest() -> None:
    """``python -m build`` must run before ``pytest tests/smoke``.

    A green run depends on the wheel actually existing when the smoke
    test fires; reversing the order would make the test skip silently
    (the wheel-build step inside the test would still succeed, but the
    intent of "validate the artefact produced by *this* job" would be
    lost). We assert ordering by step index.
    """
    workflow = _load(NIGHTLY_PATH)
    job = _docker_smoke_job(workflow)
    steps = job.get("steps", [])
    build_idx = next(
        (i for i, s in enumerate(steps) if "python -m build" in str(s.get("run", ""))),
        None,
    )
    pytest_idx = next(
        (i for i, s in enumerate(steps) if "tests/smoke" in str(s.get("run", ""))),
        None,
    )
    assert build_idx is not None, "docker-smoke must call 'python -m build'"
    assert pytest_idx is not None, "docker-smoke must invoke pytest on tests/smoke"
    assert build_idx < pytest_idx, (
        "docker-smoke must build the wheel before running tests/smoke"
    )


def test_nightly_docker_smoke_enables_run_env() -> None:
    """The smoke pytest step must set ``CROSSMEM_RUN_SMOKE=1``.

    Without that opt-in env var, ``tests/smoke/test_docker_smoke.py``
    skips itself (the test is gated to keep dev-machine runs from
    pulling a 80MB ubuntu image on every ``pytest``). Forgetting it in
    CI would turn the job into a silent green no-op.
    """
    workflow = _load(NIGHTLY_PATH)
    job = _docker_smoke_job(workflow)
    saw_env = False
    for step in job.get("steps", []):
        if "tests/smoke" not in str(step.get("run", "")):
            continue
        env = step.get("env") or job.get("env") or {}
        if str(env.get("CROSSMEM_RUN_SMOKE")) == "1":
            saw_env = True
            break
    assert saw_env, (
        "the pytest step for tests/smoke must set CROSSMEM_RUN_SMOKE=1 "
        "(otherwise the smoke test skips itself)"
    )


# ---------------------------------------------------------------------------
# release.yml
# ---------------------------------------------------------------------------


def test_release_workflow_exists() -> None:
    assert RELEASE_PATH.is_file(), f"missing workflow file: {RELEASE_PATH}"


def test_release_triggers_only_on_version_tags() -> None:
    """Release must fire on ``v*`` tag pushes (Trusted Publishers contract)."""
    triggers = _triggers(_load(RELEASE_PATH))
    push = triggers.get("push") or {}
    tags = push.get("tags") or []
    assert any(pattern.startswith("v") for pattern in tags), (
        f"release must trigger on v* tags, got tags={tags!r}"
    )


def test_release_uses_pypi_trusted_publishers() -> None:
    """No long-lived API token: rely on OIDC via PyPA's publish action."""
    workflow = _load(RELEASE_PATH)
    publish_steps = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if "pypi-publish" in uses:
                publish_steps.append(step)
    assert publish_steps, (
        "release must include pypa/gh-action-pypi-publish step for Trusted Publishers"
    )
    # No password / API token in publish step — Trusted Publishers uses OIDC.
    for step in publish_steps:
        with_block = step.get("with") or {}
        assert "password" not in with_block, (
            "Trusted Publishers must not pass an API token via 'password'"
        )


def test_release_requests_oidc_id_token_permission() -> None:
    """Trusted Publishers and Sigstore both need ``id-token: write``."""
    workflow = _load(RELEASE_PATH)
    publish_jobs = [
        job
        for job in workflow.get("jobs", {}).values()
        if any(
            "pypi-publish" in step.get("uses", "")
            or "sigstore" in step.get("uses", "").lower()
            for step in job.get("steps", [])
        )
    ]
    assert publish_jobs, "no release job found that publishes or signs"
    for job in publish_jobs:
        permissions = job.get("permissions") or workflow.get("permissions") or {}
        assert permissions.get("id-token") == "write", (
            "publish/sign job must declare 'id-token: write' (OIDC)"
        )


def test_release_signs_artifacts_with_sigstore() -> None:
    """Distribution artifacts must be signed with Sigstore before publishing."""
    workflow = _load(RELEASE_PATH)
    sigstore_steps = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses", "").lower()
            if "sigstore" in uses:
                sigstore_steps.append(step)
    assert sigstore_steps, (
        "release must sign artifacts with sigstore/gh-action-sigstore-python"
    )


def test_release_builds_distribution_before_publishing() -> None:
    """Sanity: ``python -m build`` (or equivalent) must precede the publish."""
    workflow = _load(RELEASE_PATH)
    saw_build = False
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if "python -m build" in run or "pyproject-build" in run:
                saw_build = True
    assert saw_build, "release must build sdist + wheel via 'python -m build'"
