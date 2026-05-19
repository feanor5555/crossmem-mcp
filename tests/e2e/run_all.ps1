# Windows entry-point for the CrossMem E2E suite (task 27.2).
#
# Mirror of ``run_all.sh`` for the Windows column of the matrix.
# Responsibilities — deliberately small so the script stays auditable
# in one screen and reusable for tasks 27.4+ (per-CLI scenarios append
# themselves to the same report):
#
#   1. Build the windows runner image from ``docker/Dockerfile.windows``
#      with the repo root as build context (so the working-copy gets
#      baked into the image).
#   2. Run each scenario script inside that image, capturing stdout +
#      stderr into ``reports/<timestamp>/<scenario>.log``.
#   3. Emit ``reports/<timestamp>.json`` with the schema mandated by
#      TODO.md 27.1 / 27.2:
#         {"runner": "windows", "scenarios": [...],
#          "started_at": "...", "finished_at": "..."}
#
# The script is the human-facing manual test for 27.2 (DoD: ``pwsh
# tests/e2e/run_all.ps1`` exits 0 and a valid JSON report appears
# under ``reports/``). It is **not** invoked from pytest — the python
# tests in ``tests/e2e_runner/`` only validate the script's *shape*.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve paths relative to this script, not to the caller's CWD —
# letting ``pwsh tests/e2e/run_all.ps1`` from the repo root behave
# identically to ``cd tests/e2e ; pwsh run_all.ps1``.
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot    = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$Dockerfile  = Join-Path $ScriptDir 'docker\Dockerfile.windows'
$ReportsDir  = Join-Path $ScriptDir 'reports'
$ImageTag    = if ($env:CROSSMEM_E2E_WINDOWS_IMAGE) {
    $env:CROSSMEM_E2E_WINDOWS_IMAGE
} else {
    'crossmem-e2e:windows'
}

# ``yyyyMMddTHHmmssZ`` produces basic-format ISO-8601 with no colons —
# mandatory on NTFS, which forbids ``:`` in filenames. The
# ``started_at`` field below uses extended format for human
# readability inside the JSON.
$Timestamp   = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$StartedAt   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$ReportJson  = Join-Path $ReportsDir ($Timestamp + '.json')
$LogDir      = Join-Path $ReportsDir $Timestamp

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Output "==> building $ImageTag from $Dockerfile"
& docker build --file $Dockerfile --tag $ImageTag $RepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed with exit code $LASTEXITCODE"
}

# Each scenario is one entry in this array; the smoke scenario is the
# only one for task 27.2, later tasks (27.4+) prepend their own
# scenarios above the smoke one or replace the loop with a discovery
# step. Keeping the array explicit makes the report deterministic.
# Path uses forward slashes so the JSON ``name`` field is identical
# to the Linux runner's (downstream consumers should not care which
# OS produced the report).
$Scenarios = @(
    'scenarios/_smoke/hello.ps1'
)

# Per-scenario JSON fragments accumulate here; we serialise the whole
# document via ``ConvertTo-Json`` at the end so the final payload
# stays valid even when the loop runs zero or many times.
$ScenarioFragments = New-Object System.Collections.Generic.List[object]

foreach ($scenario in $Scenarios) {
    $name        = $scenario
    $baseName    = [IO.Path]::GetFileNameWithoutExtension($scenario)
    $logFile     = Join-Path $LogDir ($baseName + '.log')
    $relLog      = "reports/$Timestamp/$baseName.log"
    Write-Output "==> running scenario $name"

    $startEpoch = [double](Get-Date -UFormat %s)
    # Volume-mount the scenarios dir read-only at C:\scenarios inside
    # the container; ``--workdir C:\work`` matches the Dockerfile's
    # final WORKDIR. The scenario path inside the container uses
    # backslashes because powershell.exe parses the argument.
    $hostScenariosDir = Join-Path $ScriptDir 'docker\scenarios'
    $containerScenario = 'C:\scenarios\' + ($scenario -replace '/', '\')
    & docker run --rm `
        --volume "${hostScenariosDir}:C:\scenarios:ro" `
        --workdir 'C:\work' `
        $ImageTag `
        powershell -NoProfile -ExecutionPolicy Bypass -File $containerScenario `
        *> $logFile
    $scenarioExit = $LASTEXITCODE
    $endEpoch = [double](Get-Date -UFormat %s)

    # ``%.3f`` gives ms-precision which matches the Linux sibling's
    # spec ("<float>") without leaking sub-millisecond noise into the
    # report.
    $durationS = [math]::Round($endEpoch - $startEpoch, 3)

    $status = if ($scenarioExit -eq 0) { 'pass' } else { 'fail' }

    # Build the fragment as an ordered hashtable so the resulting
    # JSON has stable key order. ``ConvertTo-Json`` quotes these keys
    # at serialisation time, so the emitted document carries the
    # literal field names "name", "status", "duration_s", "log_path"
    # mandated by the schema documented in README.md.
    $ScenarioFragments.Add([ordered]@{
        name        = $name
        status      = $status
        duration_s  = $durationS
        log_path    = $relLog
    }) | Out-Null
}

$FinishedAt = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

# Assemble the final report. ``ConvertTo-Json -Depth 5`` is sufficient
# for the two-level structure (top-level + scenario fragments). The
# top-level keys serialise to the literal field names "runner",
# "scenarios", "started_at", "finished_at" required by the schema
# (Linux sibling uses the same names; downstream consumers should not
# care which OS produced the report). We write to a ``.tmp`` file
# first and ``Move-Item`` to make the write atomic — readers (CI,
# humans tailing the dir) never observe a half-written file.
$Report = [ordered]@{
    runner       = 'windows'
    scenarios    = $ScenarioFragments
    started_at   = $StartedAt
    finished_at  = $FinishedAt
}
$tmpReport = Join-Path $ReportsDir (".$Timestamp.tmp.json")
$Report | ConvertTo-Json -Depth 5 | Set-Content -Path $tmpReport -Encoding utf8
Move-Item -Force -Path $tmpReport -Destination $ReportJson

Write-Output "==> wrote $ReportJson"

# Exit non-zero if any scenario failed so callers (CI, the developer's
# shell) see the failure even when the JSON report itself was written
# successfully.
if ($ScenarioFragments | Where-Object { $_.status -eq 'fail' }) {
    Write-Error "==> at least one scenario failed; see $LogDir"
    exit 1
}
exit 0
