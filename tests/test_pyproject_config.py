"""Regression tests for ``pyproject.toml`` pytest/coverage configuration.

Task 19.2: ``--cov`` and ``--cov-fail-under`` must NOT live in
``[tool.pytest.ini_options].addopts``. If they do, any CLI invocation that
adds its own ``--cov=...`` measures coverage twice — the second pass starts
from zero and silently fails ``--cov-fail-under``. Several
Implementation-Subagent runs were tripped up by this red-but-not-really
signal.

Contract enforced here:

* Local ``pytest`` (no flags) runs WITHOUT coverage — fast.
* CI explicitly passes ``--cov=crossmem --cov-fail-under=90`` in the test
  step.
* The pre-push pytest hook stays a plain ``python -m pytest`` (no coverage
  measurement; the CI coverage job is the gate, not the developer's push).
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
NIGHTLY_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
PRECOMMIT_PATH = REPO_ROOT / ".pre-commit-config.yaml"
COVERAGERC_DEFAULT_PATH = REPO_ROOT / ".coveragerc"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_addopts_does_not_pin_coverage_flags() -> None:
    """``addopts`` must not contain ``--cov`` or ``--cov-fail-under``.

    Pinning coverage in ``addopts`` makes every invocation that adds its own
    ``--cov=...`` measure twice — the redundant pass invariably fails
    ``--cov-fail-under`` even when the real run is at 100%.
    """
    config = _load_pyproject()
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--cov" not in addopts, (
        "pyproject.toml [tool.pytest.ini_options].addopts must not pin "
        "--cov / --cov=... (task 19.2). Got: "
        f"{addopts!r}"
    )
    assert "--cov-fail-under" not in addopts, (
        "pyproject.toml [tool.pytest.ini_options].addopts must not pin "
        "--cov-fail-under (task 19.2). Got: "
        f"{addopts!r}"
    )


def test_addopts_keeps_non_coverage_defaults() -> None:
    """Stripping coverage must not lose the other addopts (``--tb=short``)."""
    config = _load_pyproject()
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--tb=short" in addopts, (
        f"--tb=short must remain in addopts after coverage strip, got {addopts!r}"
    )


def test_ci_test_step_enforces_coverage_threshold() -> None:
    """CI test step must explicitly run with ``--cov-fail-under=90``.

    Coverage moved from ``addopts`` (everywhere) to the CI step (one place).
    The threshold must therefore appear in ``ci.yml``.
    """
    workflow = _load_yaml(CI_WORKFLOW_PATH)
    test_job = workflow["jobs"]["test"]
    steps = test_job["steps"]
    pytest_steps = [
        step
        for step in steps
        if isinstance(step.get("run"), str) and "pytest" in step["run"]
    ]
    assert pytest_steps, "ci.yml test job is missing a pytest invocation"
    joined = "\n".join(step["run"] for step in pytest_steps)
    assert "--cov=crossmem" in joined, (
        "ci.yml test step must invoke pytest with --cov=crossmem after the "
        "addopts strip (task 19.2)."
    )
    assert "--cov-fail-under=90" in joined, (
        "ci.yml test step must enforce --cov-fail-under=90 explicitly after "
        "the addopts strip (task 19.2)."
    )


def test_precommit_has_no_pytest_hook() -> None:
    """Pre-push must not run pytest — slim hook delegates to tools/verify.py.

    Earlier revisions ran the full pytest suite on every push. That made
    ``git push`` painfully slow and was not even the real gate (CI is).
    The hook is now slim (``editable-install-guard`` + ``ruff check``)
    and pytest with coverage moved to ``tools/verify.py`` for on-demand
    runs. This regression test catches a future re-add.
    """
    config = _load_yaml(PRECOMMIT_PATH)
    hooks: list[dict] = []
    for repo in config.get("repos", []):
        hooks.extend(repo.get("hooks", []))
    pytest_hook = next((h for h in hooks if h.get("id") == "pytest"), None)
    assert pytest_hook is None, (
        "pre-push pytest hook must not exist — heavy tests run on demand "
        "via tools/verify.py, not on every push"
    )


# Task 21.10 — Coverage profile per job. The optional chroma/qdrant backends
# are skipped by the default CI matrix (their extras are not installed), so
# measuring them there would force a permanent 0% drag. The nightly job
# installs ``[chroma,qdrant]`` and runs the real backend tests, so it must
# include those modules in the coverage report. The configuration shape is:
#
# * ``pyproject.toml`` defines NO ``omit`` — it is the universal baseline.
# * ``.coveragerc`` at repo root carries the default-CI omit; ``ci.yml`` opts
#   into it via ``--cov-config=.coveragerc``.
# * ``nightly.yml`` runs pytest with ``--cov=crossmem`` and NO ``--cov-config``
#   override, so the optional backends are measured.


def test_pyproject_does_not_omit_optional_backends() -> None:
    """pyproject.toml must not omit chroma/qdrant from coverage (task 21.10).

    The omit moved to ``.coveragerc`` (default-CI profile) so the nightly
    coverage job — which installs the optional extras — measures the
    optional backends. Keeping the omit in pyproject.toml would silently
    drop them from every coverage report regardless of which profile a
    caller selects.
    """
    config = _load_pyproject()
    omit = config.get("tool", {}).get("coverage", {}).get("run", {}).get("omit", [])
    for path in omit:
        assert "chroma_backend" not in path, (
            "pyproject.toml [tool.coverage.run].omit must not exclude "
            "chroma_backend (task 21.10 — moved to .coveragerc). "
            f"Got omit={omit!r}"
        )
        assert "qdrant_backend" not in path, (
            "pyproject.toml [tool.coverage.run].omit must not exclude "
            "qdrant_backend (task 21.10 — moved to .coveragerc). "
            f"Got omit={omit!r}"
        )


def test_default_coveragerc_omits_optional_backends() -> None:
    """``.coveragerc`` (default-CI profile) must omit optional backends.

    Default CI does not install ``[chroma,qdrant]``; measuring those
    modules there would produce a permanent 0% drag that buries the
    real coverage signal. The nightly job overrides this by NOT passing
    ``--cov-config=.coveragerc`` so the default omit is dropped.
    """
    assert COVERAGERC_DEFAULT_PATH.is_file(), (
        f".coveragerc must exist at repo root (task 21.10), expected "
        f"{COVERAGERC_DEFAULT_PATH}"
    )
    text = COVERAGERC_DEFAULT_PATH.read_text(encoding="utf-8")
    assert "chroma_backend.py" in text, (
        ".coveragerc must omit src/crossmem/backends/chroma_backend.py "
        f"(task 21.10). Got: {text!r}"
    )
    assert "qdrant_backend.py" in text, (
        ".coveragerc must omit src/crossmem/backends/qdrant_backend.py "
        f"(task 21.10). Got: {text!r}"
    )


def test_ci_test_step_uses_default_coveragerc() -> None:
    """Default CI must opt into ``.coveragerc`` via ``--cov-config``.

    Without ``--cov-config``, ``coverage`` falls back to ``pyproject.toml``,
    which (after task 21.10) carries no omit — and the default CI job lacks
    the optional extras, so chroma/qdrant would land in the report as 0%.
    """
    workflow = _load_yaml(CI_WORKFLOW_PATH)
    test_job = workflow["jobs"]["test"]
    pytest_steps = [
        step
        for step in test_job["steps"]
        if isinstance(step.get("run"), str) and "pytest" in step["run"]
    ]
    joined = "\n".join(step["run"] for step in pytest_steps)
    assert "--cov-config=.coveragerc" in joined, (
        "ci.yml test step must pass --cov-config=.coveragerc so the "
        "default-CI omit applies (task 21.10). Got step content: "
        f"{joined!r}"
    )


def test_nightly_optional_backends_job_measures_coverage() -> None:
    """Nightly optional-backends job must run pytest with coverage flags.

    Task 21.10 — once ``[chroma,qdrant]`` extras are installed, the optional
    backends must be measured. The job invokes pytest with ``--cov=crossmem``
    and ``--cov-fail-under=90`` and MUST NOT pass ``--cov-config=.coveragerc``
    (that profile would re-apply the default omit and defeat the point).
    """
    workflow = _load_yaml(NIGHTLY_WORKFLOW_PATH)
    job = workflow["jobs"]["optional-backends"]
    pytest_steps = [
        step
        for step in job["steps"]
        if isinstance(step.get("run"), str) and "pytest" in step["run"]
    ]
    joined = "\n".join(step["run"] for step in pytest_steps)
    assert "--cov=crossmem" in joined, (
        "nightly.yml optional-backends job must invoke pytest with "
        f"--cov=crossmem (task 21.10). Got: {joined!r}"
    )
    assert "--cov-fail-under=90" in joined, (
        "nightly.yml optional-backends job must enforce --cov-fail-under=90 "
        f"(task 21.10). Got: {joined!r}"
    )
    assert "--cov-config=.coveragerc" not in joined, (
        "nightly.yml optional-backends job must NOT pass "
        "--cov-config=.coveragerc — that would re-apply the default omit "
        f"and defeat the point of task 21.10. Got: {joined!r}"
    )


def test_nightly_optional_backends_installs_extras() -> None:
    """Sanity: nightly must install ``[chroma,qdrant]`` to measure them.

    Without the extras, the backend modules import but the tests skip —
    coverage of those modules would still be ~0%. This assertion pins the
    install line as the precondition for task 21.10's coverage promise.
    """
    workflow = _load_yaml(NIGHTLY_WORKFLOW_PATH)
    job = workflow["jobs"]["optional-backends"]
    install_steps = [
        step
        for step in job["steps"]
        if isinstance(step.get("run"), str) and "pip install" in step["run"]
    ]
    joined = "\n".join(step["run"] for step in install_steps)
    assert "chroma" in joined and "qdrant" in joined, (
        "nightly.yml optional-backends job must install [chroma,qdrant] "
        f"extras (task 21.10 precondition). Got: {joined!r}"
    )
