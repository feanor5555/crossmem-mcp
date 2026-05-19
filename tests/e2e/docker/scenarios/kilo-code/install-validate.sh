#!/usr/bin/env bash
# Kilo Code CLI-config validation scenario for task 27.9.
#
# Runs inside the Linux runner image (``Dockerfile.linux``) where
# ``crossmem`` is already on ``PATH`` via pipx. The scenario seeds an
# isolated fake home with a minimal valid Kilo Code MCP config and
# drives the full ``crossmem install`` -> ``crossmem install`` ->
# ``crossmem uninstall`` lifecycle, emitting a single JSON report
# fragment to ``$REPORT_PATH`` in the schema fixed by task 27.1.
#
# Kilo Code is a Cline fork and stores its MCP config inside the VSCode
# globalStorage tree at
# ``<vscode-user>/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json``
# (mirrors ``_kilocode_path`` in ``src/crossmem/connectors/registry.py``).
# The top-level shape is the same flat ``mcpServers`` mapping every
# other flat-JSON connector uses; each entry carries ``command`` /
# ``args`` / ``env``.
#
# Rather than mocking the platform-specific VSCode root
# (``~/.config/Code`` on Linux, ``%APPDATA%/Code`` on Windows,
# ``~/Library/Application Support/Code`` on macOS) we use the
# ``CROSSMEM_VSCODE_USER_DIR`` environment override exposed by
# ``crossmem.connectors._vscode`` to point the connector at our
# Fake-Home subdirectory on every host. That keeps the bash and
# PowerShell variants symmetric and means a developer running the
# script on macOS during a debug session does not hit the wrong VSCode
# path.
#
# Outputs:
#   * Exit 0 on success, 1 on any validation failure.
#   * Appends one JSON report fragment to ``$REPORT_PATH`` matching the
#     task-27.1 schema (``name`` / ``status`` / ``duration_s`` /
#     ``log_path``). ``REPORT_PATH`` is provided by ``run_all.sh``; the
#     script falls back to ``/tmp/crossmem-e2e-report.jsonl`` so a
#     manual ``bash install-validate.sh`` for debugging still completes.
#
# The script is intentionally self-contained: it never mutates the host
# filesystem outside its ``mktemp -d`` Fake-Home. That makes a
# debugging run idempotent w.r.t. the developer's real Kilo Code
# install and keeps the suite safe to re-run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/kilo-code/install-validate.sh"
LOG_PATH="${LOG_PATH:-reports/kilo-code/install-validate.log}"
REPORT_PATH="${REPORT_PATH:-/tmp/crossmem-e2e-report.jsonl}"

# Wall-clock timer — ``date +%s.%N`` gives ms precision and matches the
# 27.1 report schema (``duration_s`` as a float).
START_EPOCH="$(date +%s.%N)"

FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"

# Pin the VSCode user root to a deterministic subdirectory of the
# Fake-Home so the kilocode connector's ``_vscode_user_root`` lookup
# resolves to a known location on every platform.
VSCODE_USER_DIR="${FAKE_HOME}/Code"
export CROSSMEM_VSCODE_USER_DIR="${VSCODE_USER_DIR}"

CONFIG_DIR="${VSCODE_USER_DIR}/User/globalStorage/kilocode.kilo-code/settings"
CONFIG_PATH="${CONFIG_DIR}/mcp_settings.json"
mkdir -p "${CONFIG_DIR}"

# Seed a minimal valid Kilo Code config: an unrelated ``mcpServers``
# entry that must survive both re-install and uninstall. JSON is
# hand-written so the assertions further down can match the literal
# field names spec-style.
cat >"${CONFIG_PATH}" <<'JSON'
{
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

count_backups() {
    # Count ``mcp_settings.json.bak.*`` siblings. ``find`` keeps the
    # helper portable across coreutils versions; the ``2>/dev/null``
    # swallows the "no match" stderr from the glob.
    find "${CONFIG_DIR}" -maxdepth 1 -name 'mcp_settings.json.bak.*' \
        -type f 2>/dev/null | wc -l | tr -d ' '
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

# Always emit the fragment on exit, then clean up the Fake-Home.
trap 'emit_report; rm -rf "${FAKE_HOME}"' EXIT

# ---------------------------------------------------------------------------
# Step 1 — first ``crossmem install``
# ---------------------------------------------------------------------------

echo "==> first crossmem install"
if ! crossmem install >/dev/null; then
    fail "first crossmem install exited non-zero"
fi

# Validate the rewritten config against the Kilo Code MCP layout. The
# script uses ``python3 -`` for round-trip parsing + field-level
# assertions because ``jq`` is not guaranteed inside the base image.
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

# The unrelated neighbour must still be there.
assert "other" in servers, "pre-existing 'other' entry was clobbered"
PY

BACKUPS_AFTER_FIRST="$(count_backups)"
if [[ "${BACKUPS_AFTER_FIRST}" -lt 1 ]]; then
    fail "expected at least one mcp_settings.json.bak.* after first install"
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
crossmem_keys = [k for k in servers if k == "crossmem"]
assert len(crossmem_keys) == 1, f"duplicate crossmem entries: {crossmem_keys}"
assert "other" in servers, "'other' entry lost on re-install"
PY

BACKUPS_AFTER_SECOND="$(count_backups)"
# Retention cap from ``connectors/config_io.BACKUP_RETENTION`` is 5;
# the count must not exceed it after any number of re-installs.
if [[ "${BACKUPS_AFTER_SECOND}" -gt 5 ]]; then
    fail "backup count exploded: ${BACKUPS_AFTER_SECOND} > BACKUP_RETENTION"
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
PY

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> kilo-code install-validate: OK"
    exit 0
fi

echo "==> kilo-code install-validate: FAIL (${FAIL_REASON})" >&2
exit 1
