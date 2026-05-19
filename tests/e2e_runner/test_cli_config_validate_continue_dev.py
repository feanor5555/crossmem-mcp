"""Static validation for the Continue.dev CLI-config E2E scenario (task 27.10).

Mirror of :mod:`tests.e2e_runner.test_cli_config_validate_cursor` for
the Continue.dev connector. Running the bash and PowerShell scripts
end-to-end requires a Docker daemon plus a freshly built crossmem wheel
inside the container; both are gated behind the manual entry-points
(``bash tests/e2e/run_all.sh`` / ``pwsh tests/e2e/run_all.ps1``) so CI
stays fast and offline.

What this module guards instead are the regressions that would silently
turn the manual command into a no-op for Continue.dev:

* the bash + PowerShell scenario scripts exist at the spec-mandated
  paths and refuse to run on non-bash / non-pwsh shells via explicit
  shebangs / strict-mode pragmas;
* both scripts use strict-mode error handling so a failed assertion
  cannot be masked by a swallowed non-zero exit;
* both seed a minimal **valid** Continue.dev YAML config (the
  ``config.yaml`` 2.x layout the connector prefers when both variants
  exist) into a Fake-Home, exercise ``crossmem install`` twice
  (idempotency) and ``crossmem uninstall`` once;
* the parity invariants between the two scripts — same path under
  ``~/.continue/config.yaml``, same neighbour-entry name (proves the
  "andere Eintraege bleiben unveraendert" clause is actually checked
  on both platforms) — match exactly so a fix on one platform is
  guaranteed to land on the other.

Drifting either script away from these invariants is how downstream
tasks (27.11+ for the remaining CLIs) end up with subtly different
contracts. Catching that drift here, in milliseconds, is cheaper than
debugging a flaky Docker run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = REPO_ROOT / "tests" / "e2e" / "docker" / "scenarios" / "continue-dev"
BASH_SCRIPT = SCENARIO_DIR / "install-validate.sh"
PS1_SCRIPT = SCENARIO_DIR / "install-validate.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
    """Return the git-tracked file mode (e.g. ``100755``) or ``""``.

    Mirrors the helper in :mod:`tests.e2e_runner.test_cli_config_validate_cursor`
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


def test_continue_dev_scenario_layout_exists() -> None:
    """Both spec-mandated scripts exist at the TODO 27.10 paths.

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

    The script uses ``[[ ... ]]`` and ``$()`` arithmetic; a fall-back
    to ``/bin/sh`` (dash on Debian) would break those without a clear
    error message.
    """
    head = _read(BASH_SCRIPT).splitlines()[0]
    assert head.startswith("#!"), f"missing shebang: {head!r}"
    assert "bash" in head, f"shebang is not bash: {head!r}"


def test_bash_script_uses_strict_mode() -> None:
    """``set -euo pipefail`` is mandatory for the bash scenario.

    Without ``-e`` and ``-o pipefail`` a failed validation step (e.g.
    a Python sub-shell returning non-zero on a missing key) would not
    propagate into the script's exit code and the per-CLI report
    fragment would record ``status: "pass"`` for a broken install.
    """
    content = _read(BASH_SCRIPT)
    assert "set -euo pipefail" in content, "missing `set -euo pipefail`"


def test_bash_script_seeds_fake_home_with_minimal_continuedev_config() -> None:
    """The bash script must seed a Fake-Home containing a valid Continue.dev config.

    Continue.dev's 2.x MCP config lives at ``~/.continue/config.yaml``
    and the server map is under the top-level ``mcpServers`` key (see
    ``ContinueDevConnector._register_yaml`` in
    ``src/crossmem/connectors/continuedev.py``). The Fake-Home must
    already contain a neighbour entry so the idempotency + uninstall
    checks have something to compare against.
    """
    content = _read(BASH_SCRIPT)
    # Seeding implies overriding HOME so ``Path.home()`` inside crossmem
    # resolves to the temporary dir.
    assert re.search(r"\bexport\s+HOME=", content), (
        "bash script does not override HOME for the Fake-Home seed"
    )
    assert ".continue/config.yaml" in content, (
        "bash script does not target ~/.continue/config.yaml"
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

    Continue.dev expects each ``mcpServers`` entry to carry these three
    fields (matches what ``ContinueDevConnector._build_entry`` writes).
    Spelling them out here prevents a future regression where the
    script only checks that *some* entry was added but never opens it.
    """
    content = _read(BASH_SCRIPT)
    for key in ('"command"', '"args"', '"env"'):
        assert key in content, (
            f"bash script does not validate the {key} field in the entry"
        )


def test_bash_script_checks_bak_idempotency() -> None:
    """Idempotency: exactly one ``*.bak.*`` after two install runs.

    ``crossmem.connectors.config_io.backup_config`` writes a timestamped
    backup before every register. The script's second ``crossmem
    install`` must observe that the existing entry is already there and
    skip the rewrite — otherwise a second ``.bak.<timestamp>`` would
    appear next to ``config.yaml``.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\.bak", content), (
        "bash script does not check for the `.bak` backup file"
    )
    # The script must compare the backup count before/after the second
    # install — search for an explicit count check.
    assert re.search(r"(count|wc\s+-l|len)", content, re.IGNORECASE), (
        "bash script does not count backups (idempotency invariant)"
    )


def test_bash_script_preserves_neighbour_entry_after_uninstall() -> None:
    """Uninstall must remove crossmem and leave the neighbour entry intact.

    The DoD: "Eintrag ist entfernt, andere Eintraege bleiben
    unveraendert". The neighbour name we seed at the top of the script
    must be re-asserted after uninstall — and the assertion must phrase
    crossmem's absence in a Python-``assert``-friendly form so the
    Python sub-shell raises on regression.
    """
    content = _read(BASH_SCRIPT)
    # Both halves of the invariant must be present: crossmem gone AND
    # neighbour still present. ``"crossmem" not in servers`` is the
    # exact assertion the embedded Python uses.
    assert '"crossmem" not in servers' in content, (
        "bash script does not assert the crossmem entry is removed after uninstall"
    )
    assert '"echo-server" in servers' in content, (
        "bash script does not assert the neighbour `echo-server` survives uninstall"
    )


def test_bash_script_appends_to_report_path() -> None:
    """The DoD requires a JSON report fragment appended via ``>> $REPORT_PATH``.

    The fragment uses the same schema as task 27.1 so ``run_all.sh``
    can merge it into the top-level report without translation.
    """
    content = _read(BASH_SCRIPT)
    assert "REPORT_PATH" in content, "bash script does not reference REPORT_PATH"
    assert ">>" in content, (
        "bash script does not append to the report (uses `>` instead?)"
    )
    # Schema keys from task 27.1.
    for key in ('"name"', '"status"', '"duration_s"'):
        assert key in content, f"bash script's report fragment is missing key {key}"


# ---------------------------------------------------------------------------
# PowerShell script — shape
# ---------------------------------------------------------------------------


def test_ps1_script_declares_strict_mode() -> None:
    """``install-validate.ps1`` must enable strict error handling.

    PowerShell's default is to keep going after a non-terminating
    error; explicit ``$ErrorActionPreference = 'Stop'`` (or ``Set-
    StrictMode``) is the equivalent of bash's ``set -e``. Without it
    a failed ``Test-Path`` assertion would emit a warning and the
    scenario would still exit 0.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\$ErrorActionPreference\s*=\s*['\"]Stop['\"]", content), (
        "PowerShell script does not set `$ErrorActionPreference = 'Stop'`"
    )


def test_ps1_script_seeds_fake_home_with_minimal_continuedev_config() -> None:
    """The PowerShell script must seed an equivalent Fake-Home.

    Windows reads the home from ``%USERPROFILE%`` (via ``Path.home()``).
    Overriding ``USERPROFILE`` (and ``HOME`` for cross-shell hosts)
    points crossmem at the Fake-Home.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\$env:USERPROFILE\s*=", content), (
        "PowerShell script does not override USERPROFILE"
    )
    assert ".continue" in content and "config.yaml" in content, (
        "PowerShell script does not target ~/.continue/config.yaml"
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
    install_calls = len(re.findall(r"crossmem\s+install\b", content))
    uninstall_calls = len(re.findall(r"crossmem\s+uninstall\b", content))
    assert install_calls >= 2, (
        f"PS1 script invokes `crossmem install` only {install_calls} time(s)"
    )
    assert uninstall_calls >= 1, "PS1 script does not invoke `crossmem uninstall`"


def test_ps1_script_validates_command_args_env_keys() -> None:
    """PS1 script must validate the same YAML keys as the bash script.

    Keeps the Linux + Windows runners' coverage symmetric: a Continue.dev
    entry written by ``crossmem install`` carries ``command``/``args``/
    ``env`` on both platforms.
    """
    content = _read(PS1_SCRIPT)
    for key in ("command", "args", "env"):
        assert key in content, f"PS1 script does not mention the `{key}` field"


def test_ps1_script_checks_bak_idempotency() -> None:
    """PS1 script must check that the second install does not create a second backup.

    Mirror of :func:`test_bash_script_checks_bak_idempotency`. The
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
    assert "REPORT_PATH" in content, "PS1 script does not reference REPORT_PATH"
    assert re.search(r"(Add-Content|Out-File\s+-Append|>>)", content), (
        "PS1 script does not append the report fragment"
    )
    for key in ('"name"', '"status"', '"duration_s"'):
        assert key in content, f"PS1 script's report fragment is missing key {key}"


def test_ps1_script_not_contains_precedence_is_parenthesised() -> None:
    """``-not`` must not bind to a bare variable before ``-contains``.

    PowerShell parses ``-not $x -contains 'y'`` as
    ``(-not $x) -contains 'y'`` — boolean against a string — which
    silently bypasses the intended existence check. Same guard as in
    ``test_scenario_claude_code.test_ps_not_contains_precedence_is_parenthesised``
    so a future migration of the script to ``-contains``-style
    membership tests stays correct.
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


def test_scripts_target_the_same_continuedev_path() -> None:
    """Both scripts must point at ``~/.continue/config.yaml``.

    Continue.dev's MCP config path is platform-independent (see
    ``ContinueDevConnector.yaml_config_path`` — it always returns
    ``Path.home() / ".continue" / "config.yaml"``). The Windows script
    may spell the separator either way (``/`` or ``\\``) since
    PowerShell accepts both — but both must reference the same final
    component ``config.yaml`` under a ``.continue`` directory.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert ".continue" in bash and "config.yaml" in bash
    assert ".continue" in ps1 and "config.yaml" in ps1


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
    assert "scenarios/continue-dev/install-validate.sh" in bash, (
        "bash report fragment does not embed its canonical scenario name"
    )
    assert "scenarios/continue-dev/install-validate.ps1" in ps1, (
        "PS1 report fragment does not embed its canonical scenario name"
    )


def test_scripts_use_the_same_neighbour_entry_name() -> None:
    """Both scripts must seed the same neighbour entry (``echo-server``).

    The neighbour name is the pivot for the "andere Eintraege bleiben
    unveraendert" assertion. Drifting the name on one platform means
    the uninstall preservation check stops verifying anything common.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert "echo-server" in bash, "bash script does not seed `echo-server`"
    assert "echo-server" in ps1, "PS1 script does not seed `echo-server`"


# ---------------------------------------------------------------------------
# Executable bit in the git index (bash only — PS1 is invoked via pwsh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "tests/e2e/docker/scenarios/continue-dev/install-validate.sh",
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
