# goose install-validate scenario, PowerShell mirror (task 27.12).
#
# Mirrors ``install-validate.sh`` in lockstep so the Windows runner
# (task 27.2) gets the same install -> validate -> idempotency ->
# uninstall coverage. The scenario seeds an isolated fake home with a
# minimal valid Goose config (one unrelated top-level entry plus an
# existing ``extensions`` slot), drives the full ``crossmem install``
# -> ``crossmem install`` -> ``crossmem uninstall`` lifecycle, and
# emits a single JSON report fragment to ``$env:REPORT_PATH`` in the
# schema fixed by task 27.1.
#
# Why mirror in PowerShell — bash inside a Windows container is not
# guaranteed (``ltsc2022`` does not ship Git Bash); the runner script
# from task 27.2 needs a native ``.ps1`` entry-point per scenario.
#
# Why ``%APPDATA%/goose/config.yaml`` on Windows — Goose's config root
# branches on ``sys.platform`` (see ``_config_root_for_platform`` in
# ``src/crossmem/connectors/goose.py``). On Windows the connector reads
# ``%APPDATA%`` instead of ``$HOME/.config``; remapping HOME alone
# would leave the installer writing to the real ``%APPDATA%/goose``
# slot of the host. We therefore redirect ``APPDATA`` (and ``HOME`` /
# ``USERPROFILE`` for cross-shell hosts and the bash mirror) to the
# Fake-Home.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

$ScenarioName = 'scenarios/goose/install-validate.ps1'
$LogPath = if ($env:LOG_PATH) { $env:LOG_PATH } else { 'reports/goose/install-validate.log' }

# ``$env:REPORT_PATH`` is supplied by ``run_all.ps1``. When the scenario
# is launched stand-alone (developer debugging) we fall back to a
# temp file so the script never aborts on an undefined variable.
$ReportPath = if ($env:REPORT_PATH) {
    $env:REPORT_PATH
} else {
    Join-Path ([System.IO.Path]::GetTempPath()) 'crossmem-e2e-report.jsonl'
}

# Wall-clock timer. ``Get-Date`` produces .NET DateTime which we
# subtract at the end for a TotalSeconds float.
$StartTime = Get-Date

# Create a tmpdir that doubles as ``$HOME`` for the duration of the
# scenario. Cleanup runs unconditionally via the ``finally`` block at
# the bottom of the script so subsequent runs start blank.
$FakeHome = Join-Path ([System.IO.Path]::GetTempPath()) ("crossmem-e2e-goose-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $FakeHome -Force | Out-Null

# Remap both ``HOME`` (used by ``pathlib.Path.home()`` on Windows when
# set) and ``USERPROFILE`` (the official Windows variable) so the
# crossmem installer's home-resolution lands in the fake dir. Also
# remap ``APPDATA`` since Goose's Windows branch reads it directly to
# locate ``goose/config.yaml`` — without this the installer would still
# write to the real ``%APPDATA%/goose`` on the host.
$env:HOME = $FakeHome
$env:USERPROFILE = $FakeHome
$env:APPDATA = $FakeHome

# Goose on Windows resolves the config root to ``%APPDATA%/goose``,
# so the YAML file lands directly under ``$FakeHome/goose``. Pre-create
# the dir explicitly so the seed write below succeeds before the first
# ``crossmem install`` is run.
$ConfigDir = Join-Path $FakeHome 'goose'
New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
$ConfigPath = Join-Path $ConfigDir 'config.yaml'

# Seed the config — same shape as the bash variant so the two tests
# validate identical post-conditions. Goose's extension schema uses
# ``cmd`` (not ``command``) and ``enabled`` (not ``env``), so the
# ``other`` neighbour entry stays YAML-native here too.
$SeedConfig = @'
existing: preserve-me
extensions:
  other:
    type: stdio
    cmd: C:\Windows\System32\cmd.exe
    args: []
    enabled: true
'@
Set-Content -LiteralPath $ConfigPath -Value $SeedConfig -Encoding utf8

# ---------------------------------------------------------------------------
# Helpers — status tracking + report emit
# ---------------------------------------------------------------------------

$script:Status = 'pass'
$script:FailReason = ''

function Set-Failure {
    param([string]$Reason)
    if ($script:Status -eq 'pass') {
        $script:Status = 'fail'
        $script:FailReason = $Reason
        Write-Error -ErrorAction Continue -Message "FAIL: $Reason"
    }
}

function Emit-Report {
    $duration = [Math]::Round(((Get-Date) - $StartTime).TotalSeconds, 3)
    # Mirror the bash printf exactly — the orchestrator joins fragments
    # by string concatenation, so any deviation (different field order,
    # alternate quoting) breaks the report.
    $fragment = '{{"name": "{0}", "status": "{1}", "duration_s": {2}, "log_path": "{3}"}}' -f `
        $ScenarioName, $script:Status, $duration, $LogPath
    Add-Content -LiteralPath $ReportPath -Value $fragment -Encoding utf8
}

# Helper — invoke a small Python snippet that parses the on-disk YAML
# and returns 0 on success / non-zero on assertion failure. PowerShell
# has no native YAML parser, so we shell out to ``python`` (the same
# interpreter pipx used to install crossmem, hence ``pyyaml`` is on
# its path as a transitive dep).
function Invoke-YamlAssert {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Script,
        [Parameter(Mandatory)] [string]$FailureMessage
    )
    # Write the snippet to a tmp file so PowerShell quoting does not
    # mangle Python indentation or quotes. ``Set-Content -Encoding utf8``
    # gives us a BOM-free file the Python interpreter can read directly.
    $snippetPath = Join-Path ([System.IO.Path]::GetTempPath()) ("crossmem-yaml-" + [Guid]::NewGuid().ToString('N') + ".py")
    try {
        Set-Content -LiteralPath $snippetPath -Value $Script -Encoding utf8
        & python $snippetPath $Path | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Set-Failure $FailureMessage
        }
    }
    finally {
        if (Test-Path -LiteralPath $snippetPath) {
            Remove-Item -LiteralPath $snippetPath -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    # -----------------------------------------------------------------------
    # Step 1 — first ``crossmem install``
    # -----------------------------------------------------------------------
    Write-Host '==> first crossmem install'
    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Set-Failure 'first crossmem install exited non-zero'
    }

    # Validate the rewritten config against the Goose stdio-extension
    # layout. The snippet must exercise the same schema fields as the
    # bash mirror (``type``/``cmd``/``args``/``enabled``) — drift here
    # is what the static validator test catches.
    $postInstallScript = @'
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

assert isinstance(data, dict), "config root must be a mapping"
assert "extensions" in data, "missing top-level extensions"
extensions = data["extensions"]
assert isinstance(extensions, dict), "extensions not a mapping"
assert "crossmem" in extensions, "crossmem entry not registered"
entry = extensions["crossmem"]
for field in ("type", "cmd", "args", "enabled"):
    assert field in entry, f"crossmem entry missing {field!r}"
assert entry["type"] == "stdio", f"unexpected type {entry['type']!r}"
assert isinstance(entry["cmd"], str) and entry["cmd"], "empty cmd"
assert isinstance(entry["args"], list), "args not a list"
assert entry["enabled"] is True, "extension is not enabled"

assert "other" in extensions, "pre-existing 'other' entry was clobbered"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
'@
    Invoke-YamlAssert -Path $ConfigPath -Script $postInstallScript `
        -FailureMessage 'post-install config validation failed'

    # Backup file is named ``<config>.bak.<ts>`` per ``connectors/config_io.py``.
    $backups = Get-ChildItem -LiteralPath $ConfigDir -Filter 'config.yaml.bak.*' `
        -File -ErrorAction SilentlyContinue
    if (($backups | Measure-Object).Count -lt 1) {
        Set-Failure 'expected at least one config.yaml.bak.* after first install'
    }

    # -----------------------------------------------------------------------
    # Step 2 — second ``crossmem install`` proves idempotency
    # -----------------------------------------------------------------------
    Write-Host '==> second crossmem install (idempotency)'
    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Set-Failure 'second crossmem install exited non-zero'
    }

    $idempotencyScript = @'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

extensions = data["extensions"]
crossmem_keys = [k for k in extensions if k == "crossmem"]
assert len(crossmem_keys) == 1, f"duplicate crossmem entries: {crossmem_keys}"
assert "other" in extensions, "'other' entry lost on re-install"
'@
    Invoke-YamlAssert -Path $ConfigPath -Script $idempotencyScript `
        -FailureMessage 'post-second-install config validation failed'

    $backupsAfter = Get-ChildItem -LiteralPath $ConfigDir -Filter 'config.yaml.bak.*' `
        -File -ErrorAction SilentlyContinue
    $backupCount = ($backupsAfter | Measure-Object).Count
    # Retention cap from ``connectors/config_io.BACKUP_RETENTION`` is 5;
    # the count must not exceed it after any number of re-installs.
    if ($backupCount -gt 5) {
        Set-Failure "backup count exploded: $backupCount > BACKUP_RETENTION"
    }

    # -----------------------------------------------------------------------
    # Step 3 — ``crossmem uninstall`` removes the entry, leaves the rest
    # -----------------------------------------------------------------------
    Write-Host '==> crossmem uninstall'
    & crossmem uninstall | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Set-Failure 'crossmem uninstall exited non-zero'
    }

    $postUninstallScript = @'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

extensions = data.get("extensions", {})
assert "crossmem" not in extensions, "crossmem entry still present after uninstall"
assert "other" in extensions, "'other' entry removed by uninstall"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
'@
    Invoke-YamlAssert -Path $ConfigPath -Script $postUninstallScript `
        -FailureMessage 'post-uninstall config validation failed'
}
finally {
    Emit-Report
    if (Test-Path -LiteralPath $FakeHome) {
        Remove-Item -LiteralPath $FakeHome -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($script:Status -eq 'pass') {
    Write-Host '==> goose install-validate: OK'
    exit 0
}

Write-Error -ErrorAction Continue -Message "==> goose install-validate: FAIL ($($script:FailReason))"
exit 1
