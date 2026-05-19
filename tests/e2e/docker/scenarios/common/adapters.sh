#!/usr/bin/env bash
# Adapter-matrix scenario (task 27.16).
#
# Drives every built-in source adapter (web, github, context7) through
# a positive Mock-HTTP case plus the full SSRF payload list and a DNS-
# pin rebind trace. The real work lives in
# ``adapter_matrix_helper.py`` next to this script — the shell wrapper
# only orchestrates env setup, log capture, and the JSON report
# fragment in the 27.1 schema.
#
# Why a Python companion — every adapter is a Python class whose
# contract (Document fields, SSRFError raising, pinning rewrite) is
# only observable from inside Python. ``httpx.MockTransport`` provides
# the mock-HTTP layer in-process; the spec asks for ``responses`` but
# that library only intercepts ``requests``/``urllib3`` while every
# adapter dispatches through ``httpx``. The semantic intent ("mock
# outbound HTTP in a companion Python script") is preserved without
# adding a Dockerfile dependency.
#
# Why ``/opt/pipx/venvs/crossmem/bin/python`` instead of the system
# python3 — the system interpreter does not have ``crossmem`` or
# ``httpx`` on its path. The pipx venv carries the exact versions the
# production CLI uses, so the scenario exercises the real adapter
# code rather than a re-implementation.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — log + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/common/adapters.sh"
LOG_PATH="${LOG_PATH:-reports/common/adapters.log}"
REPORT_PATH="${REPORT_PATH:-/tmp/crossmem-e2e-report.jsonl}"

START_EPOCH="$(date +%s.%N)"

STATUS="pass"
FAIL_REASON=""

fail() {
    if [[ "${STATUS}" == "pass" ]]; then
        STATUS="fail"
        FAIL_REASON="$1"
        echo "FAIL: $1" >&2
    fi
}

emit_report() {
    local end_epoch duration
    end_epoch="$(date +%s.%N)"
    duration="$(awk -v s="${START_EPOCH}" -v e="${end_epoch}" \
        'BEGIN { printf "%.3f", e - s }')"
    printf '{"name": "%s", "status": "%s", "duration_s": %s, "log_path": "%s"}\n' \
        "${SCENARIO_NAME}" "${STATUS}" "${duration}" "${LOG_PATH}" \
        >>"${REPORT_PATH}"
}

trap 'emit_report' EXIT

# ---------------------------------------------------------------------------
# Locate the pipx venv Python
# ---------------------------------------------------------------------------

# The Linux runner image installs crossmem via ``pipx install`` which
# yields a venv at ``${PIPX_HOME}/venvs/crossmem``. Falling back to a
# search keeps the scenario portable to ad-hoc local debugging where
# crossmem might live in a user pipx home (``~/.local/pipx``).
CROSSMEM_PYTHON=""
for candidate in \
    "${PIPX_HOME:-/opt/pipx}/venvs/crossmem/bin/python" \
    "${PIPX_HOME:-/opt/pipx}/venvs/crossmem/bin/python3" \
    "${HOME:-/root}/.local/pipx/venvs/crossmem/bin/python" \
    "${HOME:-/root}/.local/pipx/venvs/crossmem/bin/python3"
do
    if [[ -x "${candidate}" ]]; then
        CROSSMEM_PYTHON="${candidate}"
        break
    fi
done

if [[ -z "${CROSSMEM_PYTHON}" ]]; then
    fail "could not locate the pipx venv python for crossmem"
    exit 1
fi

echo "==> using crossmem venv python: ${CROSSMEM_PYTHON}"

# ---------------------------------------------------------------------------
# Drive the Python helper
# ---------------------------------------------------------------------------

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCENARIO_DIR}/adapter_matrix_helper.py"

if [[ ! -f "${HELPER}" ]]; then
    fail "adapter_matrix_helper.py missing next to scenario script"
    exit 1
fi

echo "==> running adapter-matrix checks via ${HELPER}"
if ! "${CROSSMEM_PYTHON}" "${HELPER}"; then
    fail "adapter_matrix_helper.py reported one or more failures"
fi

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> adapter-matrix: OK"
    exit 0
fi

echo "==> adapter-matrix: FAIL (${FAIL_REASON})" >&2
exit 1
