"""Static validation for the claude-code install-validate scenario (task 27.4).

The Docker-based E2E suite cannot run inside the normal pytest invocation —
spinning containers and seeding fake home directories is the manual
``bash tests/e2e/run_all.sh`` command's job. What this module guards is
the **shape** of the per-CLI scenario scripts the runner shells out to:
if the bash version drops ``set -euo pipefail`` or the PowerShell mirror
forgets ``$ErrorActionPreference = 'Stop'``, the manual run would either
swallow a failure silently or write a half-valid report.

The assertions deliberately stay close to the spec wording in TODO 27.4:

* both bash and PowerShell variants live under
  ``tests/e2e/docker/scenarios/claude-code/``;
* each script seeds a fake home with a pre-existing ``~/.claude.json``
  entry, invokes ``crossmem install``, validates the rewritten config
  against the published Claude-Code MCP layout (``mcpServers`` key with
  ``command``/``args``/``env`` fields on the ``crossmem`` entry),
  asserts ``<config>.bak.*`` was produced, re-runs ``crossmem install``
  to prove idempotency (no duplicated entry, backup count bounded by
  the retention cap), and finally invokes ``crossmem uninstall`` so the
  ``existing`` entry is preserved and the ``crossmem`` entry is gone;
* each script appends a JSON-report fragment matching the 27.1 schema
  to ``$REPORT_PATH`` (``%REPORT_PATH%`` on PowerShell) and exits 0 on
  success / 1 on failure.

Re-running these tests is microseconds — the payoff is that a future
refactor of the scenario format trips immediately instead of silently
breaking the next manual run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = REPO_ROOT / "tests" / "e2e" / "docker" / "scenarios" / "claude-code"
BASH_SCRIPT = SCENARIO_DIR / "install-validate.sh"
PS_SCRIPT = SCENARIO_DIR / "install-validate.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
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


def test_scenario_dir_exists() -> None:
    """Both scripts live under the per-CLI scenario directory.

    Splitting the existence check from the content checks keeps the
    failure message specific — a missing file fails here once instead
    of cascading through every shape assertion below.
    """
    assert SCENARIO_DIR.is_dir(), f"missing {SCENARIO_DIR}"
    assert BASH_SCRIPT.is_file(), f"missing {BASH_SCRIPT}"
    assert PS_SCRIPT.is_file(), f"missing {PS_SCRIPT}"


# ---------------------------------------------------------------------------
# install-validate.sh — bash variant
# ---------------------------------------------------------------------------


def test_bash_has_bash_shebang() -> None:
    """The script must declare bash explicitly via ``#!/usr/bin/env bash``.

    A ``#!/bin/sh`` shebang would run under dash on Debian, breaking
    bash-isms (``[[ ... ]]``, arrays, ``shopt``) the script relies on.
    """
    head = _read(BASH_SCRIPT).splitlines()[0]
    assert head.startswith("#!"), f"bash scenario missing shebang: {head!r}"
    assert "bash" in head, f"bash scenario shebang is not bash: {head!r}"


def test_bash_uses_strict_mode() -> None:
    """``set -euo pipefail`` is mandatory.

    Without ``-e`` a failed ``crossmem install`` would still let the
    script append a ``status: pass`` fragment to the report. Without
    ``pipefail`` a piped ``jq`` step would mask its own non-zero exit.
    """
    content = _read(BASH_SCRIPT)
    assert "set -euo pipefail" in content, (
        "install-validate.sh missing `set -euo pipefail`"
    )


def test_bash_seeds_fake_home() -> None:
    """The scenario must operate against an isolated ``HOME``.

    Seeding ``$HOME`` (or ``$XDG_*``) keeps the host's real
    ``~/.claude.json`` untouched and makes the script safe to re-run
    locally. We assert the script reassigns ``HOME`` and creates a
    seed entry so the ``existing`` key check at the end is meaningful.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\bHOME=", content), (
        "install-validate.sh does not override HOME — would clobber the host"
    )
    # The seed must contain an entry we can verify survives the uninstall.
    assert "existing" in content, (
        "install-validate.sh does not seed an `existing` entry for the "
        "post-uninstall preservation check"
    )


def test_bash_invokes_install_and_uninstall() -> None:
    """The script must drive the full install -> verify -> uninstall loop.

    Spec mandates: seed -> ``crossmem install`` -> validate -> re-run
    install (idempotency) -> ``crossmem uninstall`` -> verify removal.
    The double install is what proves idempotency; we count occurrences
    so a regression that drops the second invocation trips here.
    """
    content = _read(BASH_SCRIPT)
    install_calls = len(re.findall(r"crossmem\s+install\b", content))
    assert install_calls >= 2, (
        f"install-validate.sh calls `crossmem install` only {install_calls}x; "
        "spec requires two calls to prove idempotency"
    )
    assert re.search(r"crossmem\s+uninstall\b", content), (
        "install-validate.sh does not invoke `crossmem uninstall`"
    )


def test_bash_validates_mcp_schema() -> None:
    """The script must check the Claude-Code MCP layout after install.

    Spec mandates validating the ``mcpServers`` key with ``command``,
    ``args`` and ``env`` fields on the ``crossmem`` entry. Grepping the
    script for the literal field names is the cheapest static guard.
    """
    content = _read(BASH_SCRIPT)
    for token in ('"mcpServers"', '"crossmem"', '"command"', '"args"', '"env"'):
        assert token in content, (
            f"install-validate.sh does not validate {token} in the rewritten config"
        )


def test_bash_checks_backup_exists() -> None:
    """``<config>.bak.*`` must be verified after the first install.

    The connector uses timestamped backups (``.bak.<ts>``) under
    ``BACKUP_RETENTION``. The script checks ``*.bak*`` presence via
    a glob — assert that pattern is wired up.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\.claude\.json\.bak", content) or re.search(
        r"\.bak\b", content
    ), "install-validate.sh does not check that <config>.bak* was produced"


def test_bash_appends_report_fragment() -> None:
    """The script must append a JSON fragment to ``$REPORT_PATH``.

    Spec: "JSON-Report-Eintrag im Format aus 27.1 wird per
    ``>> $REPORT_PATH`` ergaenzt." The fragment must carry the same
    keys as the 27.1 per-scenario entry (``name``, ``status``,
    ``duration_s``, ``log_path``) so ``run_all.sh`` can splice the
    file in unchanged.
    """
    content = _read(BASH_SCRIPT)
    assert "$REPORT_PATH" in content or "${REPORT_PATH}" in content, (
        "install-validate.sh does not reference $REPORT_PATH"
    )
    assert re.search(r">>\s*\"?\$\{?REPORT_PATH\}?\"?", content), (
        "install-validate.sh does not append (`>>`) to $REPORT_PATH"
    )
    for key in ('"name"', '"status"', '"duration_s"', '"log_path"'):
        assert key in content, f"install-validate.sh report fragment missing {key}"


def test_bash_uses_explicit_exit_codes() -> None:
    """``exit 0`` on OK / ``exit 1`` on Fail — spec literal.

    Mirroring the 27.1 schema means the orchestrator pipes through
    these codes unchanged. We assert both literals appear so a
    refactor that returns implicit truthiness fails here.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\bexit\s+0\b", content), (
        "install-validate.sh does not contain `exit 0`"
    )
    assert re.search(r"\bexit\s+1\b", content), (
        "install-validate.sh does not contain `exit 1` for the failure path"
    )


# ---------------------------------------------------------------------------
# install-validate.ps1 — PowerShell mirror
# ---------------------------------------------------------------------------


def test_ps_uses_strict_mode() -> None:
    """PowerShell mirror must set ``$ErrorActionPreference = 'Stop'``.

    Without ``Stop`` mode, non-terminating errors from native exes
    (``crossmem``, ``Test-Path``) would let the script march on and
    record a green fragment for a broken run.
    """
    content = _read(PS_SCRIPT)
    assert re.search(r"\$ErrorActionPreference\s*=\s*['\"]Stop['\"]", content), (
        "install-validate.ps1 must set $ErrorActionPreference = 'Stop'"
    )


def test_ps_invokes_install_and_uninstall() -> None:
    """PowerShell mirror runs install twice and then uninstall.

    Same shape contract as the bash variant — the only difference is
    invocation syntax (``& crossmem`` or ``crossmem.exe``). We grep
    for the verb names so either form passes.
    """
    content = _read(PS_SCRIPT)
    install_calls = len(re.findall(r"crossmem(?:\.exe)?\s+install\b", content))
    assert install_calls >= 2, (
        f"install-validate.ps1 calls `crossmem install` only {install_calls}x; "
        "spec requires two calls to prove idempotency"
    )
    assert re.search(r"crossmem(?:\.exe)?\s+uninstall\b", content), (
        "install-validate.ps1 does not invoke `crossmem uninstall`"
    )


def test_ps_validates_mcp_schema() -> None:
    """PowerShell mirror checks the Claude-Code MCP layout.

    Same fields as the bash variant — kept symmetrical so a future
    refactor of one doesn't silently drift the other.
    """
    content = _read(PS_SCRIPT)
    for token in ("mcpServers", "crossmem", "command", "args", "env"):
        assert token in content, f"install-validate.ps1 does not validate {token!r}"


def test_ps_appends_report_fragment() -> None:
    """PowerShell mirror appends a JSON fragment to ``$env:REPORT_PATH``.

    Symmetrical to the bash variant; the only difference is the
    PowerShell-specific ``$env:`` prefix. ``Add-Content`` /
    ``Out-File -Append`` both qualify as "append" so we accept either.
    """
    content = _read(PS_SCRIPT)
    assert "$env:REPORT_PATH" in content or "$Env:REPORT_PATH" in content, (
        "install-validate.ps1 does not reference $env:REPORT_PATH"
    )
    assert re.search(r"(Add-Content|Out-File\s+-Append|>>)", content), (
        "install-validate.ps1 does not append to the report path"
    )
    for key in ('"name"', '"status"', '"duration_s"', '"log_path"'):
        assert key in content, f"install-validate.ps1 report fragment missing {key}"


def test_ps_uses_explicit_exit_codes() -> None:
    """``exit 0`` on OK / ``exit 1`` on Fail — same as bash variant.

    PowerShell ``exit N`` propagates the code to the parent shell,
    which the Windows runner (Task 27.2) will record verbatim.
    """
    content = _read(PS_SCRIPT)
    assert re.search(r"\bexit\s+0\b", content), (
        "install-validate.ps1 does not contain `exit 0`"
    )
    assert re.search(r"\bexit\s+1\b", content), (
        "install-validate.ps1 does not contain `exit 1` for the failure path"
    )


def test_ps_not_contains_precedence_is_parenthesised() -> None:
    """``-not`` must not bind to a bare variable before ``-contains``.

    PowerShell parses ``-not $x -contains 'y'`` as
    ``(-not $x) -contains 'y'`` — boolean against a string — which
    silently bypasses the intended existence check on
    ``PSObject.Properties.Name``. The script must wrap the
    ``-contains`` expression in parentheses so negation applies to
    the membership test, e.g. ``-not ($x.PSObject.Properties.Name
    -contains 'y')``.
    """
    content = _read(PS_SCRIPT)
    bad = re.search(r"-not\s+\$\w[\w.]*\s+-contains", content)
    assert bad is None, (
        f"install-validate.ps1 uses `{bad.group(0)}`-style precedence; "
        "wrap the `-contains` expression in parentheses so the `-not` "
        "negates the membership test, not the variable"
    )


# ---------------------------------------------------------------------------
# Executable bit in the git index — bash variant only
# ---------------------------------------------------------------------------


def test_bash_script_tracked_executable() -> None:
    """The bash scenario must be tracked as ``100755`` in the git index.

    Windows hosts default to ``core.filemode=false`` — without an
    explicit ``git update-index --chmod=+x`` a Linux clone hits
    ``Permission denied`` when ``run_all.sh`` shells out to it.
    """
    rel_path = "tests/e2e/docker/scenarios/claude-code/install-validate.sh"
    mode = _git_ls_files_mode(rel_path)
    if not mode:
        # Not yet staged; the commit hook re-runs this assertion on the
        # staged copy so we tolerate the unstaged state during local
        # iteration instead of blocking the developer.
        import pytest

        pytest.skip(f"{rel_path} not yet staged; will be enforced on commit")
    assert mode == "100755", (
        f"{rel_path} tracked as {mode}; run "
        f"`git update-index --add --chmod=+x {rel_path}`"
    )
