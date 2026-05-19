"""Static validation for the Kilo Code install-validate scenario (task 27.9).

The Docker-based E2E suite cannot run inside the normal pytest invocation
— spinning containers and seeding fake home directories is the manual
``bash tests/e2e/run_all.sh`` / ``pwsh tests/e2e/run_all.ps1`` command's
job. What this module guards is the **shape** of the per-CLI scenario
scripts the runner shells out to: if the bash version drops
``set -euo pipefail`` or the PowerShell mirror forgets
``$ErrorActionPreference = 'Stop'``, the manual run would either swallow
a failure silently or write a half-valid report.

The assertions stay close to the spec wording in TODO 27.9 (analog
27.4):

* both bash and PowerShell variants live under
  ``tests/e2e/docker/scenarios/kilo-code/``;
* each script seeds a fake home with a pre-existing Kilo Code
  ``mcp_settings.json`` neighbour entry, pins the VSCode user root via
  ``CROSSMEM_VSCODE_USER_DIR`` so the connector resolves the config
  path deterministically on every platform, invokes ``crossmem
  install``, validates the rewritten config against the Kilo Code MCP
  layout (``mcpServers`` key with ``command``/``args``/``env`` fields
  on the ``crossmem`` entry), asserts ``<config>.bak.*`` was produced,
  re-runs ``crossmem install`` to prove idempotency (no duplicated
  entry, backup count bounded by the retention cap), and finally
  invokes ``crossmem uninstall`` so the neighbour is preserved and the
  ``crossmem`` entry is gone;
* each script appends a JSON-report fragment matching the 27.1 schema
  to ``$REPORT_PATH`` (``$env:REPORT_PATH`` on PowerShell) and exits 0
  on success / 1 on failure.

Re-running these tests is microseconds — the payoff is that a future
refactor of the scenario format trips immediately instead of silently
breaking the next manual run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = REPO_ROOT / "tests" / "e2e" / "docker" / "scenarios" / "kilo-code"
BASH_SCRIPT = SCENARIO_DIR / "install-validate.sh"
PS_SCRIPT = SCENARIO_DIR / "install-validate.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
    """Return the git-tracked file mode (e.g. ``100755``) or ``""``.

    Mirrors the helper in :mod:`tests.e2e_runner.test_scenario_claude_code`
    so the executable-bit assertion stays consistent across runners.
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
    ``pipefail`` a piped helper step would mask its own non-zero exit.
    """
    content = _read(BASH_SCRIPT)
    assert "set -euo pipefail" in content, (
        "install-validate.sh missing `set -euo pipefail`"
    )


def test_bash_seeds_fake_home_with_minimal_kilocode_config() -> None:
    """The scenario must seed a Fake-Home containing a valid Kilo Code config.

    Kilo Code's MCP config lives at
    ``<vscode-user>/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json``.
    The Fake-Home must already contain a neighbour entry so the
    idempotency + uninstall checks have something to compare against.
    Pinning the VSCode user root via ``CROSSMEM_VSCODE_USER_DIR``
    keeps the path deterministic on Linux/macOS/Windows.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\bHOME=", content), (
        "install-validate.sh does not override HOME — would clobber the host"
    )
    assert "CROSSMEM_VSCODE_USER_DIR" in content, (
        "install-validate.sh does not pin CROSSMEM_VSCODE_USER_DIR — the "
        "kilocode connector would otherwise resolve to the host's real "
        "VSCode user root"
    )
    assert "kilocode.kilo-code" in content, (
        "install-validate.sh does not seed the kilocode.kilo-code globalStorage dir"
    )
    assert "mcp_settings.json" in content, (
        "install-validate.sh does not seed mcp_settings.json"
    )
    assert "mcpServers" in content, (
        "install-validate.sh does not seed the `mcpServers` top-level key"
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
    """The script must check the Kilo Code MCP layout after install.

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
    ``BACKUP_RETENTION``. The script checks ``mcp_settings.json.bak.*``
    presence via a glob — assert that pattern is wired up and that the
    count is taken so the idempotency cap can be enforced.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"mcp_settings\.json\.bak", content) or re.search(
        r"\.bak\b", content
    ), "install-validate.sh does not check that <config>.bak* was produced"
    assert re.search(r"(count|wc\s+-l|len)", content, re.IGNORECASE), (
        "install-validate.sh does not count backups (idempotency invariant)"
    )


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


def test_ps_seeds_fake_home_with_minimal_kilocode_config() -> None:
    """PowerShell mirror seeds an equivalent Fake-Home.

    Same shape contract as the bash variant: override the home
    variables, pin the VSCode root via ``CROSSMEM_VSCODE_USER_DIR``,
    seed the kilocode globalStorage directory with an ``mcp_settings.json``
    that already carries a neighbour entry under ``mcpServers``.
    """
    content = _read(PS_SCRIPT)
    assert re.search(r"\$env:USERPROFILE\s*=", content), (
        "install-validate.ps1 does not override USERPROFILE"
    )
    assert "CROSSMEM_VSCODE_USER_DIR" in content, (
        "install-validate.ps1 does not pin CROSSMEM_VSCODE_USER_DIR"
    )
    assert "kilocode.kilo-code" in content, (
        "install-validate.ps1 does not seed the kilocode.kilo-code globalStorage dir"
    )
    assert "mcp_settings.json" in content, (
        "install-validate.ps1 does not seed mcp_settings.json"
    )
    assert "mcpServers" in content, (
        "install-validate.ps1 does not seed the `mcpServers` top-level key"
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
    """PowerShell mirror checks the Kilo Code MCP layout.

    Same fields as the bash variant — kept symmetrical so a future
    refactor of one doesn't silently drift the other.
    """
    content = _read(PS_SCRIPT)
    for token in ("mcpServers", "crossmem", "command", "args", "env"):
        assert token in content, f"install-validate.ps1 does not validate {token!r}"


def test_ps_checks_bak_idempotency() -> None:
    """PS1 script must check that the second install respects the retention cap.

    Mirror of :func:`test_bash_checks_backup_exists`. The
    ``.bak.<timestamp>`` glob must appear and a count check must be
    visible (Measure-Object / Count / Length).
    """
    content = _read(PS_SCRIPT)
    assert ".bak" in content, (
        "install-validate.ps1 does not check for the `.bak` backup file"
    )
    assert re.search(r"(Count|Measure|Length)", content), (
        "install-validate.ps1 does not count backups (idempotency invariant)"
    )


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
    which the Windows runner (Task 27.2) records verbatim.
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
# Parity invariants between the two scripts
# ---------------------------------------------------------------------------


def test_scripts_target_the_same_kilocode_path() -> None:
    """Both scripts must point at the kilocode globalStorage path.

    The path component ``kilocode.kilo-code`` and the filename
    ``mcp_settings.json`` are platform-independent; only the VSCode
    user root differs between Linux/macOS/Windows. Asserting the
    extension folder + filename catches a drift where one script
    targets, say, the Cline path instead.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS_SCRIPT)
    assert "kilocode.kilo-code" in bash and "mcp_settings.json" in bash
    assert "kilocode.kilo-code" in ps1 and "mcp_settings.json" in ps1


def test_scripts_use_the_same_scenario_name() -> None:
    """The ``name`` field in both report fragments must match.

    Otherwise the merged report would carry two different entries for
    what is conceptually the same scenario on Linux vs Windows. The
    spec mandates a per-scenario report row; that row's name is the
    scenario script's relative path, so we assert each script embeds
    its own canonical path.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS_SCRIPT)
    assert "scenarios/kilo-code/install-validate.sh" in bash, (
        "bash report fragment does not embed its canonical scenario name"
    )
    assert "scenarios/kilo-code/install-validate.ps1" in ps1, (
        "PS1 report fragment does not embed its canonical scenario name"
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
    rel_path = "tests/e2e/docker/scenarios/kilo-code/install-validate.sh"
    mode = _git_ls_files_mode(rel_path)
    if not mode:
        # Not yet staged; the commit hook re-runs this assertion on the
        # staged copy so we tolerate the unstaged state during local
        # iteration instead of blocking the developer.
        pytest.skip(f"{rel_path} not yet staged; will be enforced on commit")
    assert mode == "100755", (
        f"{rel_path} tracked as {mode}; run "
        f"`git update-index --add --chmod=+x {rel_path}`"
    )
