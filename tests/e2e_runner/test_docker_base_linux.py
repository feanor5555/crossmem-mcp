"""Static validation for the Linux E2E runner artefacts (task 27.1).

The Docker-based E2E suite is intentionally run by humans — building
images and spawning containers in CI for every PR would be both
expensive and architecturally orthogonal to the per-component unit
tests that already gate ``main``. The DoD for task 27.1 therefore
defines the *manual* command (``bash tests/e2e/run_all.sh``) as the
end-to-end check.

What this module guards instead is the **shape** of the artefacts the
human runs. Every regression that would silently turn the manual
command into a no-op — missing scenario script, dropped executable
bit, JSON schema drift, hard-coded Windows path, forgotten
``set -euo pipefail`` — trips one of the assertions below. The cost
is microseconds per CI run; the payoff is that the manual command
either works or fails loudly the next time someone bumps the runner.

These tests are deliberately stricter than "file exists":

* The bash entry-points must start with a ``#!/usr/bin/env bash``
  shebang and use ``set -euo pipefail``. Without those the report
  could be written on a failed build because intermediate non-zero
  exits would be swallowed.
* ``run_all.sh`` and ``scenarios/_smoke/hello.sh`` must be tracked
  with ``100755`` in the git index — Windows hosts default to
  ``core.filemode=false`` so a contributor cloning on Linux would
  otherwise hit ``Permission denied``.
* The JSON-report sample in the README must satisfy the schema the
  spec mandates (``runner``, ``scenarios[].{name,status,duration_s,
  log_path}``, ``started_at``, ``finished_at``). Drifting the docs
  away from the script is how downstream tasks (27.2 windows mirror,
  27.4+ scenario reports) end up with incompatible report formats.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_ROOT = REPO_ROOT / "tests" / "e2e"
DOCKER_DIR = E2E_ROOT / "docker"
SCENARIOS_DIR = DOCKER_DIR / "scenarios"
REPORTS_DIR = E2E_ROOT / "reports"
DOCKERFILE = DOCKER_DIR / "Dockerfile.linux"
RUN_ALL = E2E_ROOT / "run_all.sh"
SMOKE_SCENARIO = SCENARIOS_DIR / "_smoke" / "hello.sh"
README = E2E_ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
    """Return the git-tracked file mode (e.g. ``100755``) or ``""``.

    Uses ``git ls-files -s`` which prints ``<mode> <hash> <stage>\t<path>``.
    Returns an empty string when the file is not yet tracked — the test
    that calls this then triggers an explicit assertion failure with a
    helpful message instead of a confusing ``IndexError``.
    """
    proc = subprocess.run(
        ["git", "ls-files", "-s", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return ""
    return proc.stdout.split()[0]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_e2e_layout_exists() -> None:
    """Every path listed in TODO 27.1 is present in the worktree.

    Splitting the existence check from the content checks keeps the
    failure message specific — a missing file fails here once, instead
    of cascading through every content assertion below.
    """
    assert E2E_ROOT.is_dir(), f"missing {E2E_ROOT}"
    assert DOCKER_DIR.is_dir(), f"missing {DOCKER_DIR}"
    assert REPORTS_DIR.is_dir(), f"missing {REPORTS_DIR}"
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"
    assert RUN_ALL.is_file(), f"missing {RUN_ALL}"
    assert SMOKE_SCENARIO.is_file(), f"missing {SMOKE_SCENARIO}"
    assert README.is_file(), f"missing {README}"


def test_reports_dir_is_gitkept() -> None:
    """``reports/`` must survive a fresh clone even when empty.

    Git does not track empty directories, so a ``.gitkeep`` (or any
    placeholder) is required. Without it, ``run_all.sh`` would have
    to ``mkdir -p`` defensively; with it the script can fail loudly
    if the directory is missing, which is the better failure mode.
    """
    placeholders = list(REPORTS_DIR.iterdir())
    assert placeholders, (
        f"{REPORTS_DIR} has no placeholder; add a .gitkeep so the empty "
        "dir survives a fresh clone"
    )


# ---------------------------------------------------------------------------
# Dockerfile.linux
# ---------------------------------------------------------------------------


def test_dockerfile_uses_debian_slim_base() -> None:
    """The base image must be a Debian/Ubuntu *slim* variant.

    Slim images keep the build context under ~80MB which matters for
    contributors on low-bandwidth connections re-running the suite. A
    non-slim ``debian:bookworm`` is ~120MB; ``ubuntu:24.04`` is closer
    to 80MB but lacks the ``slim`` qualifier so we accept both.
    """
    content = _read(DOCKERFILE)
    base = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
    assert base, f"no FROM directive in {DOCKERFILE}"
    image = base.group(1).lower()
    assert "slim" in image or image.startswith("ubuntu:"), (
        f"base image {image!r} is neither *-slim nor ubuntu:* — bloated CI"
    )


def test_dockerfile_installs_python_pipx_and_crossmem() -> None:
    """The image must contain Python, pipx and an installed crossmem.

    Subsequent scenario tasks (27.4-27.15) call ``crossmem install``
    inside the container; without pipx + crossmem the scenarios cannot
    run. Pre-installing them in the base layer keeps individual scenario
    runs fast (no per-run ``pipx install``).
    """
    content = _read(DOCKERFILE)
    assert re.search(r"python3?[\b\s]", content), (
        "Dockerfile.linux does not install python3"
    )
    assert "pipx" in content, "Dockerfile.linux does not install pipx"
    # ``pipx install /repo`` (or equivalent) copies the working-copy in.
    # Editable installs are forbidden by CLAUDE.md, so we positively
    # assert a non-editable install command is present.
    assert re.search(r"pipx\s+install\s+(?!-e\b)", content), (
        "Dockerfile.linux must `pipx install` the working-copy (non-editable)"
    )
    assert "-e " not in content.replace("--no-cache", ""), (
        "Dockerfile.linux must not use editable installs "
        "(CLAUDE.md editable-install-guard)"
    )


def test_dockerfile_copies_working_copy() -> None:
    """The Dockerfile must consume the working-copy via ``COPY``.

    Spec says "installiert ... crossmem (aus dem Working-Copy)" — so
    the build context is the repo root and the image bakes the local
    source instead of pulling from PyPI. The README documents how to
    invoke ``docker build`` with that context.
    """
    content = _read(DOCKERFILE)
    assert re.search(r"^COPY\s+", content, re.MULTILINE), (
        "Dockerfile.linux has no COPY directive — working-copy not consumed"
    )


def test_dockerfile_no_apt_cache_left_behind() -> None:
    """``apt-get install`` must clean its cache to keep the image lean.

    Without ``rm -rf /var/lib/apt/lists/*`` (or ``--no-install-recommends``
    combined with a cache prune) the image grows by ~40MB of metadata
    that is never read again. This is the standard Debian-slim idiom.
    """
    content = _read(DOCKERFILE)
    if "apt-get" in content:
        assert (
            "rm -rf /var/lib/apt/lists" in content
            or "--no-install-recommends" in content
        ), "apt-get layer leaks the apt cache into the final image"


# ---------------------------------------------------------------------------
# run_all.sh
# ---------------------------------------------------------------------------


def test_run_all_has_bash_shebang() -> None:
    """The script must declare bash explicitly via ``#!/usr/bin/env bash``.

    A ``#!/bin/sh`` shebang would run under dash on Debian, breaking
    bash-isms like ``[[ ... ]]`` and arrays that the script relies on.
    """
    head = _read(RUN_ALL).splitlines()[0]
    assert head.startswith("#!"), f"run_all.sh missing shebang: {head!r}"
    assert "bash" in head, f"run_all.sh shebang is not bash: {head!r}"


def test_run_all_uses_strict_mode() -> None:
    """``set -euo pipefail`` is mandatory.

    Without ``-e``, a failed ``docker build`` would still let the
    report be written as if everything succeeded. ``pipefail`` is
    required because the script pipes scenario stdout into ``tee``.
    """
    content = _read(RUN_ALL)
    assert "set -euo pipefail" in content, "run_all.sh missing `set -euo pipefail`"


def test_run_all_builds_image_and_runs_scenario() -> None:
    """The script must build the image and invoke the smoke scenario.

    We deliberately do not pin the image tag here — the script is free
    to compute one (``crossmem-e2e:linux`` or a sha-based tag) as long
    as a ``docker build`` and a ``docker run`` happen in that order.
    """
    content = _read(RUN_ALL)
    assert re.search(r"docker\s+build\b", content), (
        "run_all.sh does not call `docker build`"
    )
    assert re.search(r"docker\s+run\b", content), (
        "run_all.sh does not call `docker run`"
    )
    assert "scenarios/_smoke/hello.sh" in content, (
        "run_all.sh does not invoke the smoke scenario"
    )


def test_run_all_writes_timestamped_report() -> None:
    """The script must write ``reports/<timestamp>.json``.

    Timestamped filenames let a developer keep historical runs around
    without overwriting. ISO-8601-ish basic format (no colons, safe on
    Windows filesystems) is required.
    """
    content = _read(RUN_ALL)
    # ``date +...`` or printf-based timestamp — both spelled out so that
    # a switch from one to the other is reviewable.
    assert re.search(r"date\s+[-+]u?\s*\+", content), (
        "run_all.sh does not compute a timestamp"
    )
    assert "reports/" in content, "run_all.sh does not target the reports/ dir"
    assert ".json" in content, "run_all.sh does not write a .json report"


def test_run_all_emits_valid_schema_keys() -> None:
    """The report payload must mention every required schema key.

    Grepping for the key names in the script is a poor man's schema
    check but it catches the common refactor mistake of dropping a
    field (``log_path`` is the usual victim). The README test below
    runs the strict ``json.loads`` round-trip.
    """
    content = _read(RUN_ALL)
    for key in (
        '"runner"',
        '"scenarios"',
        '"name"',
        '"status"',
        '"duration_s"',
        '"log_path"',
        '"started_at"',
        '"finished_at"',
    ):
        assert key in content, f"run_all.sh report missing key {key}"
    assert '"linux"' in content, 'run_all.sh runner field is not "linux"'


# ---------------------------------------------------------------------------
# Smoke scenario
# ---------------------------------------------------------------------------


def test_smoke_scenario_exits_zero() -> None:
    """The smoke scenario must be a trivially-passing bash script.

    Spec: ``scenarios/_smoke/hello.sh`` does ``exit 0``. We check that
    an ``exit 0`` is present rather than re-executing the script —
    keeping the test platform-agnostic (no bash required on the test
    runner).
    """
    content = _read(SMOKE_SCENARIO)
    assert content.startswith("#!"), "smoke scenario missing shebang"
    assert "bash" in content.splitlines()[0], "smoke scenario shebang not bash"
    assert re.search(r"^\s*exit\s+0\b", content, re.MULTILINE), (
        "smoke scenario does not end with `exit 0`"
    )


# ---------------------------------------------------------------------------
# Executable bits in the git index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "tests/e2e/run_all.sh",
        "tests/e2e/docker/scenarios/_smoke/hello.sh",
    ],
)
def test_shell_scripts_are_tracked_executable(rel_path: str) -> None:
    """Bash entry-points must be tracked as ``100755`` in the index.

    Windows hosts have ``core.filemode=false`` by default — without
    an explicit ``git update-index --chmod=+x`` the script is
    committed as ``100644`` and a Linux clone hits ``Permission
    denied`` when it tries to run it.
    """
    mode = _git_ls_files_mode(rel_path)
    if not mode:
        pytest.skip(f"{rel_path} not yet staged; will be enforced on commit")
    assert mode == "100755", (
        f"{rel_path} tracked as {mode}; run "
        f"`git update-index --add --chmod=+x {rel_path}`"
    )


# ---------------------------------------------------------------------------
# README documents the manual test
# ---------------------------------------------------------------------------


def test_readme_documents_manual_run() -> None:
    """README must teach a human how to invoke the suite.

    Spec mandates the manual command and the expected outcome. Without
    a concrete invocation the README would be aspirational prose —
    keeping the test specific lets future scenario tasks (27.2, 27.4+)
    extend the README without breaking the contract.
    """
    content = _read(README)
    assert "bash tests/e2e/run_all.sh" in content, (
        "README does not show `bash tests/e2e/run_all.sh`"
    )
    assert "reports/" in content, (
        "README does not mention the reports/ output directory"
    )


def test_readme_contains_valid_report_example() -> None:
    """The README's example JSON must satisfy the spec schema.

    Catches the documentation/code drift where the script is updated
    to add a field but the README sample falls behind. We extract the
    first fenced ``json`` block and parse it.
    """
    content = _read(README)
    match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    assert match, "README has no fenced ```json``` example block"
    payload = json.loads(match.group(1))
    assert payload["runner"] == "linux", payload
    assert isinstance(payload["scenarios"], list) and payload["scenarios"]
    for scn in payload["scenarios"]:
        assert set(scn) >= {"name", "status", "duration_s", "log_path"}, scn
        assert scn["status"] in {"pass", "fail"}, scn["status"]
        assert isinstance(scn["duration_s"], int | float), scn["duration_s"]
    assert "started_at" in payload and "finished_at" in payload
