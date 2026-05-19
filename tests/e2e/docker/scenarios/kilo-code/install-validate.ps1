# Kilo Code CLI-config validation scenario, PowerShell mirror (task 27.9).
#
# Mirrors ``install-validate.sh`` in lockstep so the Windows runner
# (task 27.2) gets the same install -> validate -> idempotency ->
# uninstall coverage for Kilo Code. The scenario seeds an isolated
# fake home with a minimal valid Kilo Code MCP config (one unrelated
# entry under ``mcpServers``), drives the full ``crossmem install`` ->
# ``crossmem install`` -> ``crossmem uninstall`` lifecycle, and emits
# a single JSON report fragment to ``$env:REPORT_PATH`` in the schema
# fixed by task 27.1.
#
# Kilo Code is a Cline fork; its MCP config lives inside the VSCode
# globalStorage tree at
# ``<vscode-user>\User\globalStorage\kilocode.kilo-code\settings\mcp_settings.json``
# (mirrors ``_kilocode_path`` in ``src\crossmem\connectors\registry.py``).
# Rather than mocking the Windows-specific ``%APPDATA%\Code`` root we
# use the ``CROSSMEM_VSCODE_USER_DIR`` environment override exposed by
# ``crossmem.connectors._vscode`` to point the connector at our
# Fake-Home subdirectory. That keeps the bash and PowerShell variants
# symmetric and means the script also works on Linux/macOS Pwsh hosts
# during a manual debug run.
#
# Why a separate PowerShell file rather than running the bash script
# via Git-Bash: the Windows runner image ships ``servercore`` which
# has no Git Bash; the only shells available are ``cmd.exe`` and
# PowerShell. Keeping the two scripts in lock-step (same path, same
# neighbour entry, same report fragment shape) is guarded by
# :mod:`tests.e2e_runner.test_cli_config_validate_kilo_code`.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

$ScenarioName = 'scenarios/kilo-code/install-validate.ps1'
$LogPath = if ($env:LOG_PATH) { $env:LOG_PATH } else { 'reports/kilo-code/install-validate.log' }

# ``$env:REPORT_PATH`` is supplied by ``run_all.ps1``. When the scenario
# is launched stand-alone (developer debugging) we fall back to a
# temp file so the script never aborts on an undefined variable.
$ReportPath = if ($env:REPORT_PATH) {
    $env:REPORT_PATH
} else {
    Join-Path ([System.IO.Path]::GetTempPath()) 'crossmem-e2e-report.jsonl'
}

$StartTime = Get-Date

# Create a tmpdir that doubles as ``$HOME`` for the duration of the
# scenario. Cleanup runs unconditionally via the ``finally`` block at
# the bottom of the script so subsequent runs start blank.
$FakeHome = Join-Path ([System.IO.Path]::GetTempPath()) ("crossmem-e2e-kilo-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $FakeHome -Force | Out-Null

# Remap both ``HOME`` and ``USERPROFILE`` so any Path.home() fallback
# inside crossmem still resolves to the Fake-Home. ``APPDATA`` is left
# untouched — the VSCode root override below supersedes it for this
# scenario.
$env:HOME = $FakeHome
$env:USERPROFILE = $FakeHome

# Pin the VSCode user root via the documented env override so the
# Kilo Code connector resolves its config path to a deterministic
# subdirectory of the Fake-Home on every platform.
$VsCodeUserDir = Join-Path $FakeHome 'Code'
$env:CROSSMEM_VSCODE_USER_DIR = $VsCodeUserDir

$ConfigDir = Join-Path $VsCodeUserDir 'User/globalStorage/kilocode.kilo-code/settings'
$ConfigPath = Join-Path $ConfigDir 'mcp_settings.json'
New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null

# Seed the config — same shape as the bash variant so the two tests
# validate identical post-conditions.
$SeedConfig = @'
{
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

function Get-BackupCount {
    @(Get-ChildItem -LiteralPath $ConfigDir -Filter 'mcp_settings.json.bak.*' `
        -File -ErrorAction SilentlyContinue) `
        | Measure-Object `
        | Select-Object -ExpandProperty Count
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

    # Validate the rewritten config against the Kilo Code MCP layout.
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
    }

    # Backup file is named ``<config>.bak.<ts>`` per ``connectors/config_io.py``.
    $backupsAfterFirst = Get-BackupCount
    if ($backupsAfterFirst -lt 1) {
        Set-Failure 'expected at least one mcp_settings.json.bak.* after first install'
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

    $backupCount = Get-BackupCount
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
}
finally {
    Emit-Report
    if (Test-Path -LiteralPath $FakeHome) {
        Remove-Item -LiteralPath $FakeHome -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($script:Status -eq 'pass') {
    Write-Host '==> kilo-code install-validate: OK'
    exit 0
}

Write-Error -ErrorAction Continue -Message "==> kilo-code install-validate: FAIL ($($script:FailReason))"
exit 1
