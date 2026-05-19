# Smoke scenario for task 27.2 (Windows mirror of hello.sh).
#
# The Windows runner needs at least one scenario invocation to prove
# the "build image -> run container -> write report" pipeline works
# end-to-end. This script is intentionally trivial: it prints a fixed
# line so ``run_all.ps1`` can redirect it into the per-scenario log,
# then exits 0 so the report records ``status: "pass"``.
#
# Do not add real assertions here — the smoke scenario is the canary
# for the *runner*, not for crossmem itself. Real per-CLI scenarios
# live under ``scenarios/<cli>/`` and are introduced by tasks 27.4+.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Output 'hello from the crossmem e2e windows runner'
exit 0
