"""Static validation for the Windows E2E runner artefacts (task 27.2).

This module mirrors ``test_docker_base_linux.py`` for the Windows side
of the suite. The Windows runner is also a manual end-to-end check
(``pwsh tests/e2e/run_all.ps1``); CI on Linux hosts cannot exercise it
directly because ``mcr.microsoft.com/windows/servercore`` images only
run on Windows containers. What we *can* gate from any host is the
**shape** of the artefacts a Windows contributor will end up running.

Every regression that would silently turn the manual command into a
no-op trips one of the assertions below:

* The Dockerfile must target ``windows/servercore:ltsc2022`` — the
  spec pins the base explicitly so a contributor cannot drift to a
  newer LTS that has not been validated against the scenarios.
* ``run_all.ps1`` must use ``Set-StrictMode`` + ``$ErrorActionPreference
  = 'Stop'`` so a failed ``docker build`` does not silently let the
  report be written as if everything succeeded — the PowerShell
  analogue of ``set -euo pipefail``.
* The JSON-report schema is identical to the Linux runner's, with
  ``runner: "windows"``. Drift between the two would force every
  downstream consumer (tasks 27.4+) to special-case the schema by
  runner.
* The smoke scenario must be a trivially-passing ``.ps1`` (``exit 0``)
  so the runner self-test does not depend on crossmem itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_ROOT = REPO_ROOT / "tests" / "e2e"
DOCKER_DIR = E2E_ROOT / "docker"
SCENARIOS_DIR = DOCKER_DIR / "scenarios"
REPORTS_DIR = E2E_ROOT / "reports"
DOCKERFILE = DOCKER_DIR / "Dockerfile.windows"
RUN_ALL = E2E_ROOT / "run_all.ps1"
SMOKE_SCENARIO = SCENARIOS_DIR / "_smoke" / "hello.ps1"
README = E2E_ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_windows_layout_exists() -> None:
    """Every path listed in TODO 27.2 is present in the worktree.

    Splitting the existence check from content checks keeps failures
    specific — a missing file fails here once instead of cascading
    through every content assertion below.
    """
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"
    assert RUN_ALL.is_file(), f"missing {RUN_ALL}"
    assert SMOKE_SCENARIO.is_file(), f"missing {SMOKE_SCENARIO}"


# ---------------------------------------------------------------------------
# Dockerfile.windows
# ---------------------------------------------------------------------------


def test_dockerfile_windows_uses_servercore_ltsc2022() -> None:
    """The base image is pinned to ``windows/servercore:ltsc2022``.

    Spec mandates the exact tag. ``servercore`` (not ``nanoserver``)
    is required because pipx + Python need a full Win32 surface;
    ``ltsc2022`` is the current LTS and matches the
    ``windows-2022`` GitHub-hosted runners byte-for-byte. A drift to
    a newer or rolling tag would invalidate the scenario reproductions
    captured in the JSON reports.
    """
    content = _read(DOCKERFILE)
    match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
    assert match, f"no FROM directive in {DOCKERFILE}"
    image = match.group(1).lower()
    assert "mcr.microsoft.com/windows/servercore" in image, (
        f"base image {image!r} is not mcr.microsoft.com/windows/servercore — "
        f"spec pins servercore:ltsc2022 for parity with windows-2022 runners"
    )
    assert "ltsc2022" in image, (
        f"base image {image!r} is not pinned to the ltsc2022 tag"
    )


def test_dockerfile_windows_installs_python_pipx_and_crossmem() -> None:
    """The image must contain Python, pipx and an installed crossmem.

    Downstream scenarios (27.4-27.15 PowerShell mirrors) call
    ``crossmem install`` inside the container; without pipx + crossmem
    the scenarios cannot run. Pre-installing them in the base layer
    keeps individual scenario runs fast (no per-run install cost).
    Editable installs are forbidden by CLAUDE.md, so we positively
    assert a non-editable ``pipx install`` command is present.
    """
    content = _read(DOCKERFILE)
    assert re.search(r"python", content, re.IGNORECASE), (
        "Dockerfile.windows does not install Python"
    )
    assert "pipx" in content, "Dockerfile.windows does not install pipx"
    assert re.search(r"pipx\s+install\s+(?!-e\b)", content), (
        "Dockerfile.windows must `pipx install` the working-copy (non-editable)"
    )
    # No editable install — the editable-install-guard in CLAUDE.md
    # forbids `pip install -e` for crossmem source trees.
    assert " -e " not in content, (
        "Dockerfile.windows must not use editable installs "
        "(CLAUDE.md editable-install-guard)"
    )


def test_dockerfile_windows_copies_working_copy() -> None:
    """The Dockerfile must consume the working-copy via ``COPY``.

    Spec says "installiert ... crossmem (aus dem Working-Copy)" — so
    the build context is the repo root and the image bakes the local
    source instead of pulling from PyPI. The README documents how to
    invoke ``docker build`` with that context.
    """
    content = _read(DOCKERFILE)
    assert re.search(r"^COPY\s+", content, re.MULTILINE), (
        "Dockerfile.windows has no COPY directive — working-copy not consumed"
    )


def test_dockerfile_windows_uses_powershell_shell() -> None:
    """Windows Dockerfiles must declare a PowerShell SHELL directive.

    Without ``SHELL ["powershell", ...]`` (or ``pwsh``), every ``RUN``
    instruction defaults to ``cmd /S /C``, which makes multi-line
    PowerShell installers impossible to express cleanly and silently
    swallows non-zero exits unless ``$ErrorActionPreference`` is set
    inline on every line. Declaring SHELL once at the top is the
    standard Windows-container idiom and matches what the spec's
    "PowerShell-Dev-Tools" implies.
    """
    content = _read(DOCKERFILE)
    assert re.search(r"^SHELL\s+\[", content, re.MULTILINE), (
        "Dockerfile.windows missing SHELL directive — multi-line RUNs would "
        "default to cmd /S /C and swallow PowerShell errors"
    )
    assert "powershell" in content.lower() or "pwsh" in content.lower(), (
        "Dockerfile.windows SHELL directive must point at PowerShell"
    )


# ---------------------------------------------------------------------------
# run_all.ps1
# ---------------------------------------------------------------------------


def test_run_all_ps1_uses_strict_mode() -> None:
    """``$ErrorActionPreference = 'Stop'`` is the PowerShell ``set -e``.

    Without it, a failed ``docker build`` would still let the report
    be written as if everything succeeded — exactly the failure mode
    ``set -euo pipefail`` prevents in the bash sibling. We also require
    ``Set-StrictMode -Version Latest`` so that a typo in a variable
    name fails loudly instead of silently coercing to ``$null``.
    """
    content = _read(RUN_ALL)
    assert "$ErrorActionPreference" in content and "'Stop'" in content, (
        "run_all.ps1 missing `$ErrorActionPreference = 'Stop'` "
        "(PowerShell analogue of `set -e`)"
    )
    assert re.search(r"Set-StrictMode\s+-Version\s+Latest", content), (
        "run_all.ps1 missing `Set-StrictMode -Version Latest`"
    )


def test_run_all_ps1_builds_image_and_runs_scenario() -> None:
    """The script must build the image and invoke the smoke scenario.

    We deliberately do not pin the image tag — the script is free to
    compute one (``crossmem-e2e:windows`` or a sha-based tag) as long
    as a ``docker build`` and a ``docker run`` happen in that order.
    """
    content = _read(RUN_ALL)
    assert re.search(r"docker\s+build\b", content), (
        "run_all.ps1 does not call `docker build`"
    )
    assert re.search(r"docker\s+run\b", content), (
        "run_all.ps1 does not call `docker run`"
    )
    assert "scenarios/_smoke/hello.ps1" in content, (
        "run_all.ps1 does not invoke the smoke scenario (hello.ps1)"
    )


def test_run_all_ps1_writes_timestamped_report() -> None:
    """The script must write ``reports/<timestamp>.json``.

    Timestamped filenames let a developer keep historical runs around
    without overwriting. The ISO-8601 basic format (no colons) is
    mandatory on Windows: NTFS forbids ``:`` in filenames, so a
    Linux-style ``2026-05-12T17:30:45Z`` would crash ``New-Item``.
    """
    content = _read(RUN_ALL)
    # PowerShell's ``Get-Date -Format`` is the canonical way; accept
    # ``[DateTime]::UtcNow.ToString(...)`` as well to keep style
    # decisions in the script's hands.
    assert "Get-Date" in content or "UtcNow" in content, (
        "run_all.ps1 does not compute a timestamp"
    )
    assert "reports" in content, "run_all.ps1 does not target the reports/ dir"
    assert ".json" in content, "run_all.ps1 does not write a .json report"
    # No raw colons in timestamp format strings — those would crash
    # ``New-Item`` on NTFS. The Linux script uses ``%Y%m%dT%H%M%SZ``
    # for the filename; the PowerShell equivalent is
    # ``yyyyMMddTHHmmssZ``.
    assert "yyyyMMddTHHmmss" in content or "HHmmss" in content, (
        "run_all.ps1 timestamp format must avoid colons — NTFS forbids "
        "them in filenames"
    )


def test_run_all_ps1_emits_valid_schema_keys() -> None:
    """The report payload must mention every required schema key.

    Grepping for key names in the script is a poor man's schema check
    but it catches the common refactor mistake of dropping a field
    (``log_path`` is the usual victim). The README test below runs the
    strict ``json.loads`` round-trip on the documented sample.
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
        assert key in content, f"run_all.ps1 report missing key {key}"
    assert '"windows"' in content, 'run_all.ps1 runner field is not "windows"'


# ---------------------------------------------------------------------------
# Smoke scenario
# ---------------------------------------------------------------------------


def test_smoke_scenario_ps1_exits_zero() -> None:
    """The smoke scenario must be a trivially-passing PowerShell script.

    Spec: ``scenarios/_smoke/hello.ps1`` does ``exit 0``. We check for
    a literal ``exit 0`` rather than re-executing the script —
    keeping the test platform-agnostic (no pwsh required on the test
    runner). The presence of an ``exit 0`` plus the absence of any
    other ``exit`` line guarantees a deterministic ``status: "pass"``.
    """
    content = _read(SMOKE_SCENARIO)
    assert re.search(r"^\s*exit\s+0\b", content, re.MULTILINE), (
        "smoke scenario does not end with `exit 0`"
    )
    # No other exit codes hiding behind conditionals — the smoke
    # scenario is the canary for the runner, not for crossmem.
    other_exits = re.findall(r"^\s*exit\s+(?!0\b)\d+", content, re.MULTILINE)
    assert not other_exits, (
        f"smoke scenario contains non-zero exit(s) {other_exits}; the canary "
        f"must always pass so a fail signals a runner regression, not a logic bug"
    )


# ---------------------------------------------------------------------------
# README documents the Windows runner
# ---------------------------------------------------------------------------


def test_readme_documents_windows_runner() -> None:
    """README must teach a human how to invoke the Windows suite.

    Spec mandates the manual command and the expected outcome. Without
    a concrete invocation the README would be aspirational prose —
    keeping the test specific lets future scenario tasks (27.4+
    PowerShell mirrors) extend the README without breaking the
    contract.
    """
    content = _read(README)
    assert "pwsh tests/e2e/run_all.ps1" in content, (
        "README does not show `pwsh tests/e2e/run_all.ps1`"
    )
    assert "Dockerfile.windows" in content, "README does not mention Dockerfile.windows"


def test_readme_contains_windows_report_example() -> None:
    """The README's Windows example JSON must satisfy the spec schema.

    The Linux sample is already validated by
    ``test_docker_base_linux.py``. The Windows sample uses the same
    schema with ``runner: "windows"``; we parse every fenced ``json``
    block in the README, look for the one with ``runner == "windows"``,
    and validate it. That keeps the existing Linux sample intact.
    """
    content = _read(README)
    blocks = re.findall(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    assert blocks, "README has no fenced ```json``` example blocks"
    windows_payload = None
    for block in blocks:
        payload = json.loads(block)
        if payload.get("runner") == "windows":
            windows_payload = payload
            break
    assert windows_payload is not None, (
        'README has no fenced ```json``` block with `runner: "windows"` — '
        "the Windows sample documents the schema for downstream consumers"
    )
    assert isinstance(windows_payload["scenarios"], list)
    assert windows_payload["scenarios"], "windows sample has empty scenarios"
    for scn in windows_payload["scenarios"]:
        assert set(scn) >= {"name", "status", "duration_s", "log_path"}, scn
        assert scn["status"] in {"pass", "fail"}, scn["status"]
        assert isinstance(scn["duration_s"], int | float), scn["duration_s"]
    assert "started_at" in windows_payload
    assert "finished_at" in windows_payload
