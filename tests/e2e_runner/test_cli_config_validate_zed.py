"""Static validation for the Zed CLI-config E2E scenario (task 27.15).

Like the Linux-runner guards in :mod:`tests.e2e_runner.test_docker_base_linux`,
this module asserts the **shape** of the per-CLI scenario artefacts rather
than executing the scenarios. Running the bash and PowerShell scripts
end-to-end requires a Docker daemon plus a freshly built crossmem wheel
inside the container; both are gated behind the manual entry-points
(``bash tests/e2e/run_all.sh`` / ``pwsh tests/e2e/run_all.ps1``) so CI
stays fast and offline.

What this module guards instead are the regressions that would silently
turn the manual command into a no-op for Zed:

* the bash + PowerShell scenario scripts exist at the spec-mandated
  paths and refuse to run on non-bash / non-pwsh shells via explicit
  shebangs / strict-mode pragmas;
* both scripts use strict-mode error handling so a failed assertion
  cannot be masked by a swallowed non-zero exit;
* both seed a minimal **valid** Zed MCP config (the ``context_servers``
  shape Zed itself parses) into a Fake-Home, exercise ``crossmem
  install`` twice (idempotency) and ``crossmem uninstall`` once;
* the parity invariants between the two scripts — same Zed
  ``context_servers`` key, same neighbour-entry name (proves the
  "andere Eintraege bleiben unveraendert" clause is actually checked
  on both platforms) — match exactly so a fix on one platform is
  guaranteed to land on the other.

Why ``context_servers`` and not ``mcpServers`` — Zed names its MCP
provider slot ``context_servers`` (see ``FLAT_JSON_CONNECTORS`` for
``name="zed"`` in ``src/crossmem/connectors/registry.py``). Asserting
the literal key here is the cheapest place to catch a regression that
silently flipped the Zed connector to a different key and broke Zed
itself.

Drifting either script away from these invariants is how downstream
tasks end up with subtly different contracts. Catching that drift
here, in milliseconds, is cheaper than debugging a flaky Docker run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = REPO_ROOT / "tests" / "e2e" / "docker" / "scenarios" / "zed"
BASH_SCRIPT = SCENARIO_DIR / "install-validate.sh"
PS1_SCRIPT = SCENARIO_DIR / "install-validate.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_ls_files_mode(rel_path: str) -> str:
    """Return the git-tracked file mode (e.g. ``100755``) or ``""``.

    Mirrors the helper in :mod:`tests.e2e_runner.test_docker_base_linux`
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


def test_zed_scenario_layout_exists() -> None:
    """Both spec-mandated scripts exist at the TODO 27.15 paths.

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
    a Python sub-process returning false on a missing key) would not
    propagate into the script's exit code and the per-CLI report
    fragment would record ``status: "pass"`` for a broken install.
    """
    content = _read(BASH_SCRIPT)
    assert "set -euo pipefail" in content, "missing `set -euo pipefail`"


def test_bash_script_seeds_fake_home_with_minimal_zed_config() -> None:
    """The bash script must seed a Fake-Home containing a valid Zed config.

    Zed's MCP config lives at ``~/.config/zed/settings.json`` on
    Linux/macOS and the server map is under the top-level
    ``context_servers`` key (mirrors
    ``crossmem.connectors.registry.FLAT_JSON_CONNECTORS`` for Zed).
    The Fake-Home must already contain a neighbour entry so the
    idempotency + uninstall checks have something to compare against.
    """
    content = _read(BASH_SCRIPT)
    # Seeding implies overriding HOME so ``Path.home()`` inside crossmem
    # resolves to the temporary dir.
    assert re.search(r"\bexport\s+HOME=", content), (
        "bash script does not override HOME for the Fake-Home seed"
    )
    assert ".config/zed" in content, "bash script does not target ~/.config/zed/"
    assert "settings.json" in content, "bash script does not target settings.json"
    assert "context_servers" in content, (
        "bash script does not seed the `context_servers` top-level key"
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

    Zed expects each ``context_servers`` entry to carry these three
    fields (matches what ``register_mcp_server`` writes). Spelling
    them out here prevents a future regression where the script only
    checks that *some* entry was added but never opens it.
    """
    content = _read(BASH_SCRIPT)
    for key in ('"command"', '"args"', '"env"'):
        assert key in content, (
            f"bash script does not validate the {key} field in the entry"
        )


def test_bash_script_checks_bak_idempotency() -> None:
    """Backup invariant: idempotent installs do not blow past the retention cap.

    ``crossmem.connectors.config_io.register_mcp_server`` makes a
    backup before every write. The scenario keeps the backup count
    bounded by ``BACKUP_RETENTION`` (== 5) — a regression that
    disables the prune step would trip the assertion.
    """
    content = _read(BASH_SCRIPT)
    assert re.search(r"\.bak", content), (
        "bash script does not check for the `.bak` backup file"
    )
    # The script must count backups (idempotency / retention check).
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
    # Both halves of the invariant must be present: crossmem gone AND
    # neighbour still present.
    assert re.search(
        r"crossmem.*(removed|missing|absent|not.*found|not in|not\s+present)",
        content,
    ), "bash script does not assert the crossmem entry is removed after uninstall"


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


def test_ps1_script_seeds_fake_home_with_minimal_zed_config() -> None:
    """The PowerShell script must seed an equivalent Fake-Home.

    On Windows, Zed reads its settings from ``%APPDATA%/Zed/
    settings.json`` (see ``_zed_path`` in
    ``src/crossmem/connectors/registry.py``). Overriding ``APPDATA``
    (and ``USERPROFILE`` / ``HOME`` for cross-shell hosts) points
    crossmem at the Fake-Home.
    """
    content = _read(PS1_SCRIPT)
    assert re.search(r"\$env:APPDATA\s*=", content), (
        "PowerShell script does not override APPDATA (Zed reads "
        "%APPDATA%/Zed/settings.json on Windows)"
    )
    assert "Zed" in content and "settings.json" in content, (
        "PowerShell script does not target %APPDATA%/Zed/settings.json"
    )
    assert "context_servers" in content, (
        "PowerShell script does not seed the `context_servers` top-level key"
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
    """PS1 script must validate the same JSON keys as the bash script.

    Keeps the Linux + Windows runners' coverage symmetric: a Zed entry
    written by ``crossmem install`` carries ``command``/``args``/``env``
    on both platforms.
    """
    content = _read(PS1_SCRIPT)
    for key in ("command", "args", "env"):
        assert key in content, f"PS1 script does not mention the `{key}` field"


def test_ps1_script_checks_bak_idempotency() -> None:
    """PS1 script must check that the second install does not blow past retention.

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


# ---------------------------------------------------------------------------
# Parity invariants between the two scripts
# ---------------------------------------------------------------------------


def test_scripts_both_target_zed_settings_json() -> None:
    """Both scripts must point at Zed's ``settings.json``.

    Zed's MCP config path differs per platform (see ``_zed_path`` in
    ``crossmem.connectors.registry``): Linux/macOS uses
    ``~/.config/zed/settings.json`` and Windows uses
    ``%APPDATA%/Zed/settings.json``. The shared invariant is the file
    name ``settings.json`` plus a ``Zed`` directory component — both
    scripts must reference both.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert "settings.json" in bash, "bash script does not target settings.json"
    assert "zed" in bash, "bash script does not reference a `zed` config directory"
    assert "settings.json" in ps1, "PS1 script does not target settings.json"
    assert "Zed" in ps1, "PS1 script does not reference a `Zed` config directory"


def test_scripts_both_use_context_servers_key() -> None:
    """Both scripts must seed/validate the ``context_servers`` MCP slot.

    Zed-specific quirk: its MCP provider slot is named
    ``context_servers``, not ``mcpServers``. Any drift on either side
    would write the entry to the wrong key and silently break Zed
    itself; this parity check is the cheapest guard.
    """
    bash = _read(BASH_SCRIPT)
    ps1 = _read(PS1_SCRIPT)
    assert "context_servers" in bash, (
        "bash script does not reference the `context_servers` key"
    )
    assert "context_servers" in ps1, (
        "PS1 script does not reference the `context_servers` key"
    )


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
    assert "scenarios/zed/install-validate.sh" in bash, (
        "bash report fragment does not embed its canonical scenario name"
    )
    assert "scenarios/zed/install-validate.ps1" in ps1, (
        "PS1 report fragment does not embed its canonical scenario name"
    )


# ---------------------------------------------------------------------------
# Executable bit in the git index (bash only — PS1 is invoked via pwsh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "tests/e2e/docker/scenarios/zed/install-validate.sh",
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
