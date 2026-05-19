# Continue.dev CLI-config validation scenario for task 27.10 (Windows mirror).
#
# Counterpart to ``install-validate.sh`` for the Windows E2E container
# produced by ``tests/e2e/docker/Dockerfile.windows`` (task 27.2). The
# script exercises the same four invariants — install + validate, install
# idempotency, uninstall removes only the crossmem entry — using a
# Fake-Home rooted under ``$env:TEMP`` so the developer's real
# ``~/.continue/`` is never touched.
#
# Why a separate PowerShell file rather than running the bash script via
# Git-Bash: the Windows container ships ``servercore`` which has no Git
# Bash; the only shells available are ``cmd.exe`` and PowerShell. Keeping
# the two scripts in lock-step (same path, same neighbour entry, same
# report fragment shape) is guarded by
# :mod:`tests.e2e_runner.test_cli_config_validate_continue_dev`.
#
# Why YAML — Continue 2.x writes ``~/.continue/config.yaml`` and the
# connector ``ContinueDevConnector.register`` prefers YAML when both
# variants exist. PowerShell has no built-in YAML parser, so we shell
# out to ``python`` (PyYAML ships with the crossmem wheel) for the
# round-trip validation — the same approach as the bash variant.

$ErrorActionPreference = 'Stop'

$ScenarioName = 'scenarios/continue-dev/install-validate.ps1'
$ReportPath = if ($env:REPORT_PATH) { $env:REPORT_PATH } else { 'NUL' }
$LogPath = if ($env:LOG_PATH) { $env:LOG_PATH } else { 'NUL' }

# ``Get-Date -UFormat %s`` would give us a Unix epoch but is not
# guaranteed to ship with all PowerShell editions; using
# ``DateTime.UtcNow`` + a stable epoch is portable across 5.1 and 7+.
$Epoch = [DateTime]'1970-01-01T00:00:00Z'
$ScenarioStart = ([DateTime]::UtcNow - $Epoch).TotalSeconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Failure {
    param([string]$Message)
    # ``Write-Error`` would set ``$?`` to false but does not abort with
    # ``$ErrorActionPreference = 'Stop'`` if the caller is in a try.
    # ``throw`` guarantees both — and the global ``trap`` below catches
    # it so the fail fragment is still appended.
    throw "FAIL: $Message"
}

function Get-BackupCount {
    param([string]$Directory)
    # ``Get-ChildItem`` with the literal-path mode and a filter glob
    # mirrors ``ls -1 config.yaml.bak.*`` in the bash script. ``Measure-
    # Object`` exposes ``.Count`` regardless of how many items match
    # (zero, one, many).
    @(Get-ChildItem -LiteralPath $Directory -Filter 'config.yaml.bak.*' -ErrorAction SilentlyContinue) `
        | Measure-Object `
        | Select-Object -ExpandProperty Count
}

function Invoke-YamlAssertion {
    param([string]$Path, [string]$Script, [string]$FailureMessage)
    # Shell out to python so the assertion uses the same PyYAML the
    # connector itself relies on. ``$Script`` is a here-string with
    # the inline assertion body; we pipe it on stdin and pass the
    # config path as ``argv[1]`` so the layout matches the bash
    # ``python - "${CONFIG_PATH}" <<'PY' ... PY`` idiom one-for-one.
    $Script | & python - $Path
    if ($LASTEXITCODE -ne 0) {
        Write-Failure $FailureMessage
    }
}

function Add-ReportFragment {
    param([string]$Status)
    $end = ([DateTime]::UtcNow - $Epoch).TotalSeconds
    $duration = '{0:F3}' -f ($end - $ScenarioStart)
    $line = '{{"name": "{0}", "status": "{1}", "duration_s": {2}, "log_path": "{3}"}}' `
        -f $ScenarioName, $Status, $duration, $LogPath
    if ($ReportPath -ne 'NUL') {
        Add-Content -LiteralPath $ReportPath -Value $line -Encoding utf8
    }
}

# Wrap the main body in try/catch/finally so a thrown failure path
# emits the ``fail`` fragment exactly once and the success path emits
# ``pass`` exactly once.
$failed = $false
try {

    # -----------------------------------------------------------------------
    # Seed a Fake-Home with a minimal valid Continue.dev YAML config
    # -----------------------------------------------------------------------

    $FakeHome = New-Item -ItemType Directory `
        -Path (Join-Path $env:TEMP ("crossmem-e2e-continue-" + [Guid]::NewGuid().Guid))
    $env:USERPROFILE = $FakeHome.FullName
    $env:HOME = $FakeHome.FullName

    $ConfigDir = Join-Path $FakeHome.FullName '.continue'
    $ConfigPath = Join-Path $ConfigDir 'config.yaml'
    New-Item -ItemType Directory -Path $ConfigDir | Out-Null

    # Seed the YAML 2.x variant: an unrelated top-level key (``name``,
    # Continue.dev's active-assistant slot) plus a neighbour entry under
    # ``mcpServers``. We write the literal YAML text via a here-string
    # rather than ``ConvertTo-Yaml`` because PowerShell lacks a built-in
    # YAML serialiser; ``Set-Content -Encoding utf8`` mirrors the LF/no-
    # BOM output PyYAML produces.
    $seed = @'
name: e2e-assistant
mcpServers:
  echo-server:
    command: echo
    args:
      - hello
    env: {}
'@
    Set-Content -LiteralPath $ConfigPath -Value $seed -Encoding utf8

    # -----------------------------------------------------------------------
    # Step 1: install once, validate the written entry + backup
    # -----------------------------------------------------------------------

    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Failure "first crossmem install exited $LASTEXITCODE" }

    $firstAssertion = @'
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

assert isinstance(data, dict), "config root must be a mapping"
servers = data.get("mcpServers")
assert isinstance(servers, dict), "missing top-level mcpServers mapping"
assert "crossmem" in servers, "missing `crossmem` entry under mcpServers"
entry = servers["crossmem"]
for key in ("command", "args", "env"):
    assert key in entry, f"missing `{key}` in crossmem entry"
assert isinstance(entry["command"], str), "`command` must be a string"
assert isinstance(entry["args"], list), "`args` must be a list"
assert isinstance(entry["env"], dict), "`env` must be a mapping"

assert "echo-server" in servers, "neighbour `echo-server` disappeared"
assert data.get("name") == "e2e-assistant", "top-level `name` key was clobbered"
'@
    Invoke-YamlAssertion -Path $ConfigPath -Script $firstAssertion `
        -FailureMessage 'first install did not write the expected entry'

    $backupsAfterFirst = Get-BackupCount -Directory $ConfigDir
    if ($backupsAfterFirst -lt 1) {
        Write-Failure "expected >=1 .bak.<timestamp> after first install, got $backupsAfterFirst"
    }

    # -----------------------------------------------------------------------
    # Step 2: install again — idempotent
    # -----------------------------------------------------------------------

    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Failure "second crossmem install exited $LASTEXITCODE" }

    $secondAssertion = @'
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" in servers, "crossmem entry vanished on second install"
assert isinstance(servers["crossmem"], dict), "crossmem entry duplicated as a list"
'@
    Invoke-YamlAssertion -Path $ConfigPath -Script $secondAssertion `
        -FailureMessage 'second install changed the entry shape'

    $backupsAfterSecond = Get-BackupCount -Directory $ConfigDir
    if ($backupsAfterSecond -ne $backupsAfterFirst) {
        Write-Failure ("second install created an extra backup " +
            "(before=$backupsAfterFirst, after=$backupsAfterSecond)")
    }

    # -----------------------------------------------------------------------
    # Step 3: uninstall — entry removed, neighbour preserved
    # -----------------------------------------------------------------------

    # ``--yes`` is accepted by ``crossmem uninstall`` when purge is
    # involved; without ``--purge`` the flag is a harmless no-op, so
    # we pass it unconditionally for forward-compat with newer builds.
    & crossmem uninstall --yes | Out-Null
    if ($LASTEXITCODE -ne 0) {
        # Older builds may not accept --yes; retry without it.
        & crossmem uninstall | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "crossmem uninstall exited $LASTEXITCODE"
        }
    }

    $thirdAssertion = @'
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" not in servers, "crossmem entry still present after uninstall"
assert "echo-server" in servers, "neighbour `echo-server` was removed by uninstall"
assert data.get("name") == "e2e-assistant", "top-level `name` key was removed by uninstall"
'@
    Invoke-YamlAssertion -Path $ConfigPath -Script $thirdAssertion `
        -FailureMessage 'uninstall did not clean up correctly'

}
catch {
    $failed = $true
    Add-ReportFragment -Status 'fail'
    Write-Error $_
    exit 1
}
finally {
    if (-not $failed) {
        Add-ReportFragment -Status 'pass'
    }
}

exit 0
