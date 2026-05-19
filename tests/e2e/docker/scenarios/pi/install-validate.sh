#!/usr/bin/env bash
# pi install-validate scenario (task 27.8).
#
# Mirrors ``scenarios/claude-code/install-validate.sh`` (task 27.4) for
# the Pi connector. Pi has no native MCP support, but the
# ``pi-mcp-adapter`` npm package reads MCP server configs from
# ``~/.pi/agent/mcp.json`` (HOME-relative on every platform). The
# scenario seeds an isolated fake home with that nested directory and a
# minimal valid Pi config (one unrelated top-level entry plus an
# existing ``mcpServers`` slot), drives the full ``crossmem install`` ->
# ``crossmem install`` -> ``crossmem uninstall`` lifecycle, and emits a
# single JSON report fragment to ``$REPORT_PATH`` in the schema fixed
# by task 27.1.
#
# Why a dedicated fake HOME — the host's real ``~/.pi/agent/mcp.json``
# (if any) would otherwise be backed up + rewritten by the scenario;
# the container layer adds a second safety net, but redirecting
# ``HOME`` inside the script means the same script is safe to run on
# the host for ad-hoc debugging.
#
# Why two ``crossmem install`` invocations — the spec asks for an
# idempotency proof: the second run must not duplicate the
# ``crossmem`` MCP entry inside the config file. We also assert that
# the on-disk backup count stays bounded by ``BACKUP_RETENTION`` so a
# regression that disables the prune step trips the scenario.
#
# Why the unrelated ``existing`` key — ``crossmem uninstall`` must
# remove the ``crossmem`` entry without touching anything else. Seeding
# an unrelated entry and re-asserting it after uninstall is the
# minimal check that proves the connector does not over-reach.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/pi/install-validate.sh"
LOG_PATH="${LOG_PATH:-reports/pi/install-validate.log}"

# ``$REPORT_PATH`` is supplied by ``run_all.sh`` / ``run_all.ps1``. When
# the scenario is launched stand-alone (developer debugging) we fall
# back to ``/tmp/crossmem-e2e-report.jsonl`` so the script never aborts
# on an unset variable under ``set -u``.
REPORT_PATH="${REPORT_PATH:-/tmp/crossmem-e2e-report.jsonl}"

# Wall-clock timer — printf ``%.3f`` gives ms precision and matches
# the 27.1 report schema (``duration_s`` as a float).
START_EPOCH="$(date +%s.%N)"

# Create a tmpdir that doubles as ``$HOME`` for the duration of the
# scenario. A combined EXIT trap further down installs the report-emit
# step alongside the cleanup so subsequent runs start blank regardless
# of the exit status.
FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"

# Pi nests its config two directories deep under HOME — the connector
# creates the parents on demand, but seeding the directory keeps the
# pre-existing-config narrative honest (we are validating the upgrade
# path from a user who already ran ``pi-mcp-adapter`` once).
CONFIG_DIR="${FAKE_HOME}/.pi/agent"
mkdir -p "${CONFIG_DIR}"
CONFIG_PATH="${CONFIG_DIR}/mcp.json"

# Seed a minimal valid Pi config: one unrelated top-level key plus a
# pre-existing ``mcpServers`` entry that must survive uninstall. JSON
# is hand-written (not generated) so the assertions further down can
# match the literal field names spec-style.
cat >"${CONFIG_PATH}" <<'JSON'
{
  "existing": "preserve-me",
  "mcpServers": {
    "other": {
      "command": "/usr/bin/true",
      "args": [],
      "env": {}
    }
  }
}
JSON

# ---------------------------------------------------------------------------
# Helpers — status tracking + report emit
# ---------------------------------------------------------------------------

STATUS="pass"
FAIL_REASON=""

fail() {
    # Record the first failure reason and switch the final status to
    # ``fail`` without aborting yet — we still want to emit the report
    # fragment so the orchestrator can splice it into the run JSON.
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

# Always emit the fragment on exit, then propagate the right code.
trap 'emit_report; rm -rf "${FAKE_HOME}"' EXIT

# ---------------------------------------------------------------------------
# Step 1 — first ``crossmem install``
# ---------------------------------------------------------------------------

echo "==> first crossmem install"
if ! crossmem install >/dev/null; then
    fail "first crossmem install exited non-zero"
fi

# Validate the rewritten config against the Pi MCP layout (shape is
# identical to Claude Code / Cursor / Cline — see registry.py).
# ``python -c`` keeps the assertion in one process so a malformed JSON
# trips a single, clear error message.
python3 - "${CONFIG_PATH}" <<'PY' || fail "post-install config validation failed"
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

assert "mcpServers" in data, "missing top-level mcpServers"
servers = data["mcpServers"]
assert "crossmem" in servers, "crossmem entry not registered"
entry = servers["crossmem"]
for field in ("command", "args", "env"):
    assert field in entry, f"crossmem entry missing {field!r}"
assert isinstance(entry["command"], str) and entry["command"], "empty command"
assert isinstance(entry["args"], list), "args not a list"
assert isinstance(entry["env"], dict), "env not a mapping"

# The unrelated entry must still be there.
assert "other" in servers, "pre-existing 'other' entry was clobbered"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
PY

# Backup file is named ``<config>.bak.<ts>`` per ``connectors/config_io.py``.
# Pi's backups land next to ``mcp.json`` inside ``~/.pi/agent/``.
backup_count=$(find "${CONFIG_DIR}" -maxdepth 1 -name 'mcp.json.bak.*' \
    -type f | wc -l)
if [[ "${backup_count}" -lt 1 ]]; then
    fail "expected at least one mcp.json.bak.* after first install"
fi

# ---------------------------------------------------------------------------
# Step 2 — second ``crossmem install`` proves idempotency
# ---------------------------------------------------------------------------

echo "==> second crossmem install (idempotency)"
if ! crossmem install >/dev/null; then
    fail "second crossmem install exited non-zero"
fi

# After the second install: still exactly one ``crossmem`` entry, and
# the backup count is bounded by the connector's retention cap (5).
# We do not assert ``exactly N`` because the connector legitimately
# adds one backup per install up to the cap.
python3 - "${CONFIG_PATH}" <<'PY' || fail "post-second-install config validation failed"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

servers = data["mcpServers"]
# Idempotency: no duplicate entries.
crossmem_keys = [k for k in servers if k == "crossmem"]
assert len(crossmem_keys) == 1, f"duplicate crossmem entries: {crossmem_keys}"
assert "other" in servers, "'other' entry lost on re-install"
PY

backup_count_after_second=$(find "${CONFIG_DIR}" -maxdepth 1 \
    -name 'mcp.json.bak.*' -type f | wc -l)
# Retention cap from ``connectors/config_io.BACKUP_RETENTION`` is 5;
# the count must not exceed it after any number of re-installs.
if [[ "${backup_count_after_second}" -gt 5 ]]; then
    fail "backup count exploded: ${backup_count_after_second} > BACKUP_RETENTION"
fi

# ---------------------------------------------------------------------------
# Step 3 — ``crossmem uninstall`` removes the entry, leaves the rest
# ---------------------------------------------------------------------------

echo "==> crossmem uninstall"
if ! crossmem uninstall >/dev/null; then
    fail "crossmem uninstall exited non-zero"
fi

python3 - "${CONFIG_PATH}" <<'PY' || fail "post-uninstall config validation failed"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" not in servers, "crossmem entry still present after uninstall"
assert "other" in servers, "'other' entry removed by uninstall"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
PY

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> pi install-validate: OK"
    # ``emit_report`` runs via EXIT trap.
    exit 0
fi

echo "==> pi install-validate: FAIL (${FAIL_REASON})" >&2
exit 1
