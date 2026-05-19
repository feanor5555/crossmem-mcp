# Cursor CLI-config validation scenario for task 27.5 (Windows mirror).
#
# Counterpart to ``install-validate.sh`` for the Windows E2E container
# produced by ``tests/e2e/docker/Dockerfile.windows`` (task 27.2). The
# script exercises the same four invariants — install + validate, install
# idempotency, uninstall removes only the crossmem entry — using a
# Fake-Home rooted at a ``New-TemporaryFile``-derived directory so the
# developer's real ``~/.cursor/`` is never touched.
#
# Why a separate PowerShell file rather than running the bash script via
# Git-Bash: the Windows container ships ``servercore`` which has no Git
# Bash; the only shells available are ``cmd.exe`` and PowerShell. Keeping
# the two scripts in lock-step (same path, same neighbour entry, same
# report fragment shape) is guarded by
# :mod:`tests.e2e_runner.test_cli_config_validate_cursor`.

$ErrorActionPreference = 'Stop'

$ScenarioName = 'scenarios/cursor/install-validate.ps1'
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
    # mirrors ``ls -1 mcp.json.bak.*`` in the bash script. ``Measure-
    # Object`` exposes ``.Count`` regardless of how many items match
    # (zero, one, many).
    @(Get-ChildItem -LiteralPath $Directory -Filter 'mcp.json.bak.*' -ErrorAction SilentlyContinue) `
        | Measure-Object `
        | Select-Object -ExpandProperty Count
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
    # Seed a Fake-Home with a minimal valid Cursor config
    # -----------------------------------------------------------------------

    $FakeHome = New-Item -ItemType Directory `
        -Path (Join-Path $env:TEMP ("crossmem-e2e-cursor-" + [Guid]::NewGuid().Guid))
    $env:USERPROFILE = $FakeHome.FullName
    $env:HOME = $FakeHome.FullName

    $ConfigDir = Join-Path $FakeHome.FullName '.cursor'
    $ConfigPath = Join-Path $ConfigDir 'mcp.json'
    New-Item -ItemType Directory -Path $ConfigDir | Out-Null

    # Neighbour entry proves "andere Eintraege bleiben unveraendert"
    # after uninstall. ``ConvertTo-Json`` is round-tripped through
    # ``Set-Content -Encoding utf8`` so the file is in the same shape
    # the Python json module produces (no BOM, LF line endings).
    $seed = [ordered]@{
        mcpServers = [ordered]@{
            'echo-server' = [ordered]@{
                command = 'echo'
                args = @('hello')
                env = @{}
            }
        }
    }
    $seed | ConvertTo-Json -Depth 10 `
        | Set-Content -LiteralPath $ConfigPath -Encoding utf8

    # -----------------------------------------------------------------------
    # Step 1: install once, validate the written entry + backup
    # -----------------------------------------------------------------------

    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Failure "first crossmem install exited $LASTEXITCODE" }

    $data = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if (-not $data.mcpServers) { Write-Failure 'mcpServers missing after first install' }
    $entry = $data.mcpServers.crossmem
    if (-not $entry) { Write-Failure 'crossmem entry missing after first install' }
    foreach ($key in @('command', 'args', 'env')) {
        if ($null -eq $entry.$key) {
            Write-Failure "crossmem entry missing `$key after first install"
        }
    }
    if (-not $data.mcpServers.'echo-server') {
        Write-Failure 'neighbour echo-server disappeared after first install'
    }

    $backupsAfterFirst = Get-BackupCount -Directory $ConfigDir
    if ($backupsAfterFirst -lt 1) {
        Write-Failure "expected >=1 .bak.<timestamp> after first install, got $backupsAfterFirst"
    }

    # -----------------------------------------------------------------------
    # Step 2: install again — idempotent
    # -----------------------------------------------------------------------

    & crossmem install | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Failure "second crossmem install exited $LASTEXITCODE" }

    $data2 = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if (-not $data2.mcpServers.crossmem) {
        Write-Failure 'crossmem entry vanished on second install'
    }
    # ConvertFrom-Json returns a PSCustomObject for a JSON object and an
    # array for a JSON list — make sure the entry is still an object,
    # not a list of duplicates.
    if ($data2.mcpServers.crossmem -is [System.Array]) {
        Write-Failure 'crossmem entry duplicated as an array on second install'
    }

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

    $data3 = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if ($data3.mcpServers.crossmem) {
        Write-Failure 'crossmem entry still present after uninstall'
    }
    if (-not $data3.mcpServers.'echo-server') {
        Write-Failure 'neighbour echo-server was removed by uninstall'
    }

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
