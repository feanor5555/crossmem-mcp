#!/usr/bin/env bash
# Smoke scenario for task 27.1.
#
# The Linux runner needs at least one scenario invocation to prove the
# "build image -> run container -> write report" pipeline works end-to-
# end. This script is intentionally trivial: it prints a fixed line so
# ``run_all.sh`` can ``tee`` it into the per-scenario log, then exits
# 0 so the report records ``status: "pass"``.
#
# Do not add real assertions here — the smoke scenario is the canary
# for the *runner*, not for crossmem itself. Real per-CLI scenarios
# live under ``scenarios/<cli>/`` and are introduced by tasks 27.4+.

set -euo pipefail

echo "hello from the crossmem e2e linux runner"
exit 0
