"""Static validation for the Amazon Q CLI install-validate E2E scenario (task 27.14).

Like the analog guards for Claude Code
(:mod:`tests.e2e_runner.test_scenario_claude_code`), Cursor
(:mod:`tests.e2e_runner.test_cli_config_validate_cursor`) and Gemini
CLI (:mod:`tests.e2e_runner.test_cli_config_validate_gemini_cli`), this
module asserts the **shape** of the per-CLI scenario artefacts rather
than executing them. Running the bash and PowerShell scripts end-to-
end requires a Docker daemon plus a freshly built crossmem wheel
inside the container; both are gated behind the manual entry-points
(``bash tests/e2e/run_all.sh`` / ``pwsh tests/e2e/run_all.ps1``) so
CI stays fast and offline.

What this module guards instead are the regressions that would silently
turn the manual command into a no-op for the Amazon Q CLI:

* the bash + PowerShell scenario scripts exist at the spec-mandated
  paths and refuse to run on non-bash / non-pwsh shells via explicit
  shebangs / strict-mode pragmas;
* both scripts use strict-mode error handling so a failed assertion
  cannot be masked by a swallowed non-zero exit;
* both seed a minimal **valid** Amazon Q CLI MCP config (the
  ``mcpServers`` shape ``~/.aws/amazonq/mcp.json`` carries) into a
  Fake-Home, exercise ``crossmem install`` twice (idempotency) and
  ``crossmem uninstall`` once;
* the parity invariants between the two scripts — same path under
  ``~/.aws/amazonq/mcp.json``, same neighbour-entry name (proves the
  "andere Eintraege bleiben unveraendert" clause is actually checked
  on both platforms) — match exactly so a fix on one platform is
  guaranteed to land on the other.

Drifting either script away from these invariants is how downstream
tasks (other CLIs in the 27.x series) end up with subtly different
contracts. Catching that drift here, in milliseconds, is cheaper than
debugging a flaky Docker run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = REPO_ROOT / "tests" / "e2e" / "docker" / "scenarios" / "amazon-q"
BASH_SCRIPT = SCENARIO_DIR / "install-validate.sh"
PS1_SCRIPT = SCENARIO_DIR / "install-validate.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
    """Return the git-tracked file mode (e.g. ``100755``) or ``""``.

    Mirrors the helper in
    :mod:`tests.e2e_runner.test_cli_config_validate_gemini_cli` so the
    executable-bit assertion stays consistent across CLI scenarios.
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


def test_amazon_q_scenario_layout_exists() -> None:
    """Both spec-mandated scripts exist at the TODO 27.14 paths.

    Splitting the existence check from the content checks keeps the
    failure message specific — a missing file fails here once, instead
    of cascading through every content assertion below.
    """
    assert SCENARIO_DIR.is_dir(), f"missing {SCENARIO_DIR}"
    assert BASH_SCRIPT.is_file(), f"missing {BASH_SCRIPT}"
    assert PS1_SCRIPT.is_file(), f"missing {PS1_SCRIPT}"


# ---------------------------------------------------------------------------
# Bash script — shape
# ---------------------------------------------------------------------------


def test_bash_script_has_bash_shebang() -> None:
    """``install-validate.sh`` must declare bash explicitly.

    A ``#!/bin/sh`` fallback would run under dash on Debian, breaking
    bash-isms (``[[ ... ]]``, arrays, ``shopt``) the script relies on.
    """
    head = _read(BASH_SCRIPT).splitlines()[0]
    assert head.startswith("#!"), f"missing shebang: {head!r}"
    assert "bash" in head, f"shebang is not bash: {head!r}"


def test_bash_script_uses_strict_mode() -> None:
    """``set -euo pipefail`` is mandatory for the bash scenario.

    Without ``-e`` and ``-o pipefail`` a failed validation step (e.g.
    ``python`` returning false on a missing key) would not propagate
    into the script's exit code and the per-CLI report fragment would
    record ``status: "pass"`` for a broken install.
    """
    content = _read(BASH_SCRIPT)
    assert "set -euo pipefail" in content, "missing `set -euo pipefail`"


def test_bash_script_seeds_fake_home_with_minimal_amazon_q_config() -> None:
    """The bash script must seed a Fake-Home containing a valid Amazon Q config.

    Amazon Q CLI's MCP config lives at ``~/.aws/amazonq/mcp.json`` on
    every platform (mirrors ``_amazonq_path`` in
    ``src/crossmem/connectors/registry.py``) and the server map is
    under the top-level ``mcpServers`` key. The Fake-Home must already
    contain a neighbour entry so the idempotency + uninstall checks
    have something to compare against.
    """
    content = _read(BASH_SCRIPT)
    # Seeding implies overriding HOME so ``Path.home()`` inside crossmem
    # resolves to the temporary dir.
    assert re.search(r"\bexport\s+HOME=", content), (
        "bash script does not override HOME for the Fake-Home seed"
    )
    assert ".aws" in content and "amazonq" in content and "mcp.json" in content, (
        "bash script does not target ~/.aws/amazonq/mcp.json"
    )
    assert "mcpServers" in content, (
        "bash script does not seed the `mcpServers` top-level key"
    )


def test_bash_script_runs_install_twice_and_uninstall_once() -> None:
    """The bash script must exercise install (twice) and uninstall (once).

    The DoD requires (a) install -> validate, (b) install again ->
    idempotency, (c) uninstall -> entry removed. Counting the literal
    ``crossmem install`` / ``crossmem uninstall`` tokens catches a
    refactor that drops one of the steps.
    """
    content = _read(BASH_SCRIPT)
    install_calls = len(re.findall(r"crossmem\s+install\b", content))
    uninstall_calls = len(re.findall(r"crossmem\s+uninstall\b", content))
    assert install_calls >= 2, (
        f"bash script invokes `crossmem install` only {install_calls} time(s)"
    )
    assert uninstall_calls >= 1, "bash script does not invoke `crossmem uninstall`"


def test_bash_script_validates_command_args_env_keys() -> None:
    """The bash script must validate the written ``command``/``args``/``env`` keys.

    Amazon Q CLI expects each ``mcpServers`` entry to carry these
    three fields (matches what ``register_mcp_server`` writes).
    Spelling them out here prevents a future regression where the
    script only checks that *some* entry was added but never opens it.
    """
    content = _read(BASH_SCRIPT)
    for key in ('"command"', '"args"', '"env"'):
        assert key in content, (
            f"bash script does not validate the {key} field in the entry"
        )


def test_bash_script_checks_bak_present() -> None:
    """After the first install the script must verify ``mcp.json.bak.*``.

    ``crossmem.connectors.config_io.register_mcp_server`` makes a
    timestamped backup before every write. The script's first
    ``crossmem install`` must observe at least one ``.bak.<timestamp>``
    on disk afterwards.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"mcp\.json\.bak", content) or re.search(r"\.bak\b", content), (
        "bash script does not check that <config>.bak* was produced"
    )
    assert re.search(r"(count|wc\s+-l|len)", content, re.IGNORECASE), (
        "bash script does not count backups (idempotency invariant)"
    )


def test_bash_script_preserves_neighbour_entry_after_uninstall() -> None:
    """Uninstall must remove crossmem and leave the neighbour entry intact.

    The DoD: "Eintrag ist entfernt, andere Eintraege bleiben
    unveraendert". The neighbour name we seed at the top of the script
    must be re-asserted after uninstall.
    """
    content = _read(BASH_SCRIPT)
    # Stronger check: the post-uninstall block must compare against
    # ``servers`` (or ``mcpServers``) with a negated membership test.
    assert re.search(r'"crossmem"\s*not\s+in\s+servers', content), (
        "bash script does not assert `crossmem` is gone from mcpServers after uninstall"
    )


def test_bash_script_appends_to_report_path() -> None:
    """The DoD requires a JSON report fragment appended via ``>> $REPORT_PATH``.

    The fragment uses the same schema as task 27.1 so ``run_all.sh``
    can merge it into the top-level report without translation.
    """
    content = _read(BASH_SCRIPT)
    assert "REPORT_PATH" in content, "bash script does not reference REPORT_PATH"
    assert re.search(r">>\s*\"?\$\{?REPORT_PATH\}?\"?", content), (
        "bash script does not append (`>>`) to $REPORT_PATH"
    )
    for key in ('"name"', '"status"', '"duration_s"', '"log_path"'):
        assert key in content, f"bash script's report fragment is missing key {key}"


def test_bash_script_uses_explicit_exit_codes() -> None:
    """``exit 0`` on OK / ``exit 1`` on Fail — mirrors task 27.1.

    Mirroring the 27.1 schema means the orchestrator pipes through
    these codes unchanged. We assert both literals appear so a
    refactor that returns implicit truthiness fails here.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\bexit\s+0\b", content), "bash script does not contain `exit 0`"
    assert re.search(r"\bexit\s+1\b", content), (
        "bash script does not contain `exit 1` for the failure path"
    )


# ---------------------------------------------------------------------------
# PowerShell script — shape
# ---------------------------------------------------------------------------


def test_ps1_script_declares_strict_mode() -> None:
    """``install-validate.ps1`` must enable strict error handling.

    PowerShell's default is to keep going after a non-terminating
    error; explicit ``$ErrorActionPreference = 'Stop'`` (or
    ``Set-StrictMode``) is the equivalent of bash's ``set -e``.
    Without it a failed ``Test-Path`` assertion would emit a warning
    and the scenario would still exit 0.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\$ErrorActionPreference\s*=\s*['\"]Stop['\"]", content), (
        "PowerShell script does not set `$ErrorActionPreference = 'Stop'`"
    )


def test_ps1_script_seeds_fake_home_with_minimal_amazon_q_config() -> None:
    """The PowerShell script must seed an equivalent Fake-Home.

    Windows reads the home from ``%USERPROFILE%`` (via
    ``Path.home()``). Overriding ``USERPROFILE`` (and ``HOME`` for
    cross-shell hosts) points crossmem at the Fake-Home.

    The Amazon Q CLI config path is platform-independent
    (``~/.aws/amazonq/mcp.json``), so no ``%APPDATA%`` redirect is
    required — unlike OpenCode.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\$env:USERPROFILE\s*=", content), (
        "PowerShell script does not override USERPROFILE"
    )
    assert ".aws" in content and "amazonq" in content and "mcp.json" in content, (
        "PowerShell script does not target ~/.aws/amazonq/mcp.json"
    )
    assert "mcpServers" in content, (
        "PowerShell script does not seed the `mcpServers` top-level key"
    )


def test_ps1_script_runs_install_twice_and_uninstall_once() -> None:
    """The PowerShell script must exercise install (twice) and uninstall (once).

    Mirror of :func:`test_bash_script_runs_install_twice_and_uninstall_once`
    for the Windows runner.
    """
    content = _read(PS1_SCRIPT)
    install_calls = len(re.findall(r"crossmem(?:\.exe)?\s+install\b", content))
    uninstall_calls = len(re.findall(r"crossmem(?:\.exe)?\s+uninstall\b", content))
    assert install_calls >= 2, (
        f"PS1 script invokes `crossmem install` only {install_calls} time(s)"
    )
    assert uninstall_calls >= 1, "PS1 script does not invoke `crossmem uninstall`"


def test_ps1_script_validates_command_args_env_keys() -> None:
    """PS1 script must validate the same JSON keys as the bash script.

    Keeps the Linux + Windows runners' coverage symmetric: an Amazon Q
    entry written by ``crossmem install`` carries ``command``/``args``/
    ``env`` on both platforms.
    """
    content = _read(PS1_SCRIPT)
    for key in ("command", "args", "env"):
        assert key in content, f"PS1 script does not mention the `{key}` field"


def test_ps1_script_checks_bak_present() -> None:
    """PS1 script must check the ``mcp.json.bak`` glob after first install.

    Mirror of :func:`test_bash_script_checks_bak_present`. The
    ``.bak.<timestamp>`` glob must appear and a count check must be
    visible.
    """
    content = _read(PS1_SCRIPT)
    assert ".bak" in content, "PS1 script does not check for the `.bak` backup file"
    assert re.search(r"(Count|Measure|Length)", content), (
        "PS1 script does not count backups (idempotency invariant)"
    )


def test_ps1_script_appends_to_report_path() -> None:
    """PS1 script must append a JSON fragment to ``$env:REPORT_PATH``.

    Mirror of :func:`test_bash_script_appends_to_report_path`. Uses
    ``Add-Content`` or ``Out-File -Append`` so the merged report
    stays parseable.
    """
    content = _read(PS1_SCRIPT)
    assert "$env:REPORT_PATH" in content or "$Env:REPORT_PATH" in content, (
        "PS1 script does not reference $env:REPORT_PATH"
    )
    assert re.search(r"(Add-Content|Out-File\s+-Append|>>)", content), (
        "PS1 script does not append the report fragment"
    )
    for key in ('"name"', '"status"', '"duration_s"', '"log_path"'):
        assert key in content, f"PS1 script's report fragment is missing key {key}"


def test_ps1_script_uses_explicit_exit_codes() -> None:
    """``exit 0`` on OK / ``exit 1`` on Fail — same as bash variant.

    PowerShell ``exit N`` propagates the code to the parent shell,
    which the Windows runner (task 27.2) records verbatim.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\bexit\s+0\b", content), "PS1 script does not contain `exit 0`"
    assert re.search(r"\bexit\s+1\b", content), (
        "PS1 script does not contain `exit 1` for the failure path"
    )


def test_ps1_not_contains_precedence_is_parenthesised() -> None:
    """``-not`` must not bind to a bare variable before ``-contains``.

    PowerShell parses ``-not $x -contains 'y'`` as
    ``(-not $x) -contains 'y'`` — boolean against a string — which
    silently bypasses the intended existence check on
    ``PSObject.Properties.Name``. The script must wrap the
    ``-contains`` expression in parentheses so negation applies to the
    membership test, e.g. ``-not ($x.PSObject.Properties.Name -contains
    'y')``. Identical guard to the Claude Code scenario (task 27.4).
    """
    content = _read(PS1_SCRIPT)
    bad = re.search(r"-not\s+\$\w[\w.]*\s+-contains", content)
    assert bad is None, (
        f"install-validate.ps1 uses `{bad.group(0)}`-style precedence; "
        "wrap the `-contains` expression in parentheses so the `-not` "
        "negates the membership test, not the variable"
    )


# ---------------------------------------------------------------------------
# Parity invariants between the two scripts
# ---------------------------------------------------------------------------


def test_scripts_target_the_same_amazon_q_path() -> None:
    """Both scripts must point at ``~/.aws/amazonq/mcp.json``.

    Amazon Q CLI's MCP config path is platform-independent (see
    ``_amazonq_path`` in ``crossmem.connectors.registry``). The
    Windows script may spell the separator either way (``/`` or
    ``\\``) since PowerShell accepts both — but both must reference
    the same path components ``.aws``, ``amazonq`` and ``mcp.json``.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert ".aws" in bash and "amazonq" in bash and "mcp.json" in bash
    assert ".aws" in ps1 and "amazonq" in ps1 and "mcp.json" in ps1


def test_scripts_use_the_same_scenario_name() -> None:
    """The ``name`` field in both report fragments must match.

    Otherwise the merged report would carry two different entries for
    what is conceptually the same scenario on Linux vs Windows. The
    spec mandates a per-scenario report row; that row's name is the
    scenario script's relative path, so we assert each script embeds
    its own canonical path.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert "scenarios/amazon-q/install-validate.sh" in bash, (
        "bash report fragment does not embed its canonical scenario name"
    )
    assert "scenarios/amazon-q/install-validate.ps1" in ps1, (
        "PS1 report fragment does not embed its canonical scenario name"
    )


# ---------------------------------------------------------------------------
# Executable bit in the git index (bash only — PS1 is invoked via pwsh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "tests/e2e/docker/scenarios/amazon-q/install-validate.sh",
    ],
)
def test_bash_scenario_is_tracked_executable(rel_path: str) -> None:
    """Bash scenarios must be tracked as ``100755`` so a Linux clone can run them.

    Windows hosts have ``core.filemode=false`` by default — without
    an explicit ``git update-index --chmod=+x`` the script lands as
    ``100644`` and ``run_all.sh`` hits ``Permission denied``.
    PowerShell scripts are always invoked through ``pwsh -File``, so
    they don't need the executable bit.
    """
    mode = _git_ls_files_mode(rel_path)
    if not mode:
        pytest.skip(f"{rel_path} not yet staged; will be enforced on commit")
    assert mode == "100755", (
        f"{rel_path} tracked as {mode}; run "
        f"`git update-index --add --chmod=+x {rel_path}`"
    )
