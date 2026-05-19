# claude-code install-validate scenario, PowerShell mirror (task 27.4).
#
# Mirrors ``install-validate.sh`` in lockstep so the Windows runner
# (task 27.2, TBD) gets the same install -> validate -> idempotency ->
# uninstall coverage. The scenario seeds an isolated fake home with a
# minimal valid Claude-Code config (one unrelated top-level entry plus
# an existing ``mcpServers`` slot), drives the full ``crossmem install``
# -> ``crossmem install`` -> ``crossmem uninstall`` lifecycle, and emits
# a single JSON report fragment to ``$env:REPORT_PATH`` in the schema
# fixed by task 27.1.
#
# Why mirror in PowerShell — bash inside a Windows container is not
# guaranteed (``ltsc2022`` does not ship Git Bash); the runner script
# from task 27.2 needs a native ``.ps1`` entry-point per scenario.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

$ScenarioName = 'scenarios/claude-code/install-validate.ps1'
$LogPath = if ($env:LOG_PATH) { $env:LOG_PATH } else { 'reports/claude-code/install-validate.log' }

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
$FakeHome = Join-Path ([System.IO.Path]::GetTempPath()) ("crossmem-e2e-claude-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $FakeHome -Force | Out-Null

# Remap both ``HOME`` (used by ``pathlib.Path.home()`` on Windows when
# set) and ``USERPROFILE`` (the official Windows variable) so the
# crossmem installer resolves ``~/.claude.json`` to the fake dir.
$env:HOME = $FakeHome
$env:USERPROFILE = $FakeHome

$ConfigPath = Join-Path $FakeHome '.claude.json'

# Seed the config — same shape as the bash variant so the two tests
# validate identical post-conditions.
$SeedConfig = @'
{
  "existing": "preserve-me",
  "mcpServers": {
    "other": {
      "command": "C:\\Windows\\System32\\cmd.exe",
      "args": [],
      "env": {}
    }
  }
}
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

try {
    # -----------------------------------------------------------------------
    # Step 1 — first ``crossmem install``
    # -----------------------------------------------------------------------
    Write-Host '==> first crossmem install'
    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Set-Failure 'first crossmem install exited non-zero'
    }

    # Validate the rewritten config against the Claude-Code MCP layout.
    $data = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if (-not ($data.PSObject.Properties.Name -contains 'mcpServers')) {
        Set-Failure 'missing top-level mcpServers'
    }
    elseif (-not ($data.mcpServers.PSObject.Properties.Name -contains 'crossmem')) {
        Set-Failure 'crossmem entry not registered'
    }
    else {
        $entry = $data.mcpServers.crossmem
        foreach ($field in @('command', 'args', 'env')) {
            if (-not ($entry.PSObject.Properties.Name -contains $field)) {
                Set-Failure "crossmem entry missing '$field'"
            }
        }
        if (-not ($data.mcpServers.PSObject.Properties.Name -contains 'other')) {
            Set-Failure "pre-existing 'other' entry was clobbered"
        }
        if ($data.existing -ne 'preserve-me') {
            Set-Failure "top-level 'existing' lost"
        }
    }

    # Backup file is named ``<config>.bak.<ts>`` per ``connectors/config_io.py``.
    $backups = Get-ChildItem -LiteralPath $FakeHome -Filter '.claude.json.bak.*' `
        -File -ErrorAction SilentlyContinue
    if (($backups | Measure-Object).Count -lt 1) {
        Set-Failure 'expected at least one .claude.json.bak.* after first install'
    }

    # -----------------------------------------------------------------------
    # Step 2 — second ``crossmem install`` proves idempotency
    # -----------------------------------------------------------------------
    Write-Host '==> second crossmem install (idempotency)'
    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Set-Failure 'second crossmem install exited non-zero'
    }

    $data = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    $crossmemKeys = @($data.mcpServers.PSObject.Properties.Name | Where-Object { $_ -eq 'crossmem' })
    if ($crossmemKeys.Count -ne 1) {
        Set-Failure "duplicate crossmem entries: $($crossmemKeys.Count)"
    }
    if (-not ($data.mcpServers.PSObject.Properties.Name -contains 'other')) {
        Set-Failure "'other' entry lost on re-install"
    }

    $backupsAfter = Get-ChildItem -LiteralPath $FakeHome -Filter '.claude.json.bak.*' `
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

    $data = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if ($data.mcpServers.PSObject.Properties.Name -contains 'crossmem') {
        Set-Failure 'crossmem entry still present after uninstall'
    }
    if (-not ($data.mcpServers.PSObject.Properties.Name -contains 'other')) {
        Set-Failure "'other' entry removed by uninstall"
    }
    if ($data.existing -ne 'preserve-me') {
        Set-Failure "top-level 'existing' lost"
    }
}
finally {
    Emit-Report
    if (Test-Path -LiteralPath $FakeHome) {
        Remove-Item -LiteralPath $FakeHome -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($script:Status -eq 'pass') {
    Write-Host '==> claude-code install-validate: OK'
    exit 0
}

Write-Error -ErrorAction Continue -Message "==> claude-code install-validate: FAIL ($($script:FailReason))"
exit 1
