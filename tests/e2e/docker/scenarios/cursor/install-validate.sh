#!/usr/bin/env bash
# Cursor CLI-config validation scenario for task 27.5.
#
# Runs inside the Linux E2E container produced by ``tests/e2e/docker/
# Dockerfile.linux`` and exercises the full ``crossmem install`` -> validate
# -> ``crossmem install`` (idempotency) -> ``crossmem uninstall`` cycle
# against a Fake-Home that already contains a Cursor MCP config with a
# neighbour entry. Layout follows the analog scenario for Claude Code
# (task 27.4) so the merge logic in ``run_all.sh`` does not need a
# per-CLI special case.
#
# Cursor stores its MCP servers in ``~/.cursor/mcp.json`` on every
# platform (mirrors ``_cursor_path`` in
# ``src/crossmem/connectors/registry.py``) under a top-level
# ``mcpServers`` map. Each server entry carries ``command`` / ``args``
# / ``env`` — the shape MCP clients spawn via ``Popen([command, *args])``
# and the shape ``register_mcp_server`` writes.
#
# Outputs:
#   * Exit 0 on success, 1 on any validation failure.
#   * Appends one JSON report fragment to ``$REPORT_PATH`` matching the
#     task-27.1 schema (``name``/``status``/``duration_s``/``log_path``).
#     ``REPORT_PATH`` is provided by ``run_all.sh``; the script falls
#     back to ``/dev/null`` so a manual ``bash install-validate.sh``
#     for debugging still completes.
#
# The script is intentionally self-contained: it never mutates the host
# filesystem outside its ``mktemp -d`` Fake-Home. That makes a debugging
# run idempotent w.r.t. the developer's real ``~/.cursor/`` and keeps
# the suite safe to re-run.

set -euo pipefail

SCENARIO_NAME="scenarios/cursor/install-validate.sh"
REPORT_PATH="${REPORT_PATH:-/dev/null}"
LOG_PATH="${LOG_PATH:-/dev/null}"

# Use ``date +%s.%N`` for ms-precision wall-clock — same idiom as
# ``run_all.sh`` so the merged report stays comparable.
SCENARIO_START_EPOCH="$(date +%s.%N)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fail() {
    # Print a diagnostic to stderr and bail out via the trap.
    echo "FAIL: $*" >&2
    exit 1
}

count_backups() {
    # Count ``mcp.json.bak.*`` siblings of the Cursor config. Using a
    # glob via ``ls -1`` keeps the helper portable across coreutils
    # versions; the ``2>/dev/null`` swallows the "no match" stderr.
    local dir="$1"
    ls -1 "${dir}"/mcp.json.bak.* 2>/dev/null | wc -l | tr -d ' '
}

emit_report_fragment() {
    # Append one JSON object to ``$REPORT_PATH``. ``run_all.sh`` merges
    # the per-scenario fragments at the end of the run, so each line
    # must stand on its own as a JSON value.
    local status="$1"
    local end_epoch
    end_epoch="$(date +%s.%N)"
    local duration_s
    duration_s="$(awk -v s="${SCENARIO_START_EPOCH}" -v e="${end_epoch}" \
        'BEGIN { printf "%.3f", e - s }')"
    printf '{"name": "%s", "status": "%s", "duration_s": %s, "log_path": "%s"}\n' \
        "${SCENARIO_NAME}" "${status}" "${duration_s}" "${LOG_PATH}" \
        >>"${REPORT_PATH}"
}

# Ensure that any failure path still writes a ``fail`` fragment so the
# merged report shows the scenario ran. ``trap`` fires on every exit;
# we branch on the captured exit code so success paths skip the second
# write.
on_exit() {
    local code=$?
    if [[ "${code}" -ne 0 ]]; then
        emit_report_fragment "fail"
    fi
    exit "${code}"
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Seed a Fake-Home with a minimal valid Cursor config
# ---------------------------------------------------------------------------

FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"

CONFIG_DIR="${FAKE_HOME}/.cursor"
CONFIG_PATH="${CONFIG_DIR}/mcp.json"
mkdir -p "${CONFIG_DIR}"

# The neighbour entry proves "andere Eintraege bleiben unveraendert"
# after uninstall. We use ``echo-server`` because it is shape-only —
# Cursor never invokes it during this scenario.
cat >"${CONFIG_PATH}" <<'JSON'
{
  "mcpServers": {
    "echo-server": {
      "command": "echo",
      "args": ["hello"],
      "env": {}
    }
  }
}
JSON

# ---------------------------------------------------------------------------
# Step 1: install once, validate the written entry + backup
# ---------------------------------------------------------------------------

crossmem install >/dev/null

# The written entry must include the three Cursor-required fields. We
# use ``python -m json.tool`` for round-trip validation (fails fast on
# malformed JSON) and ``python -c`` for the field-level assertions —
# ``jq`` is not guaranteed inside the base image.
python -m json.tool <"${CONFIG_PATH}" >/dev/null \
    || fail "config is not valid JSON after first install"

python - "${CONFIG_PATH}" <<'PY' || fail "first install did not write the expected entry"
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" in servers, "missing `crossmem` entry under mcpServers"
entry = servers["crossmem"]
for key in ("command", "args", "env"):
    assert key in entry, f"missing `{key}` in crossmem entry"
assert isinstance(entry["command"], str), "`command` must be a string"
assert isinstance(entry["args"], list), "`args` must be a list"
assert isinstance(entry["env"], dict), "`env` must be a mapping"

# The neighbour must still be present.
assert "echo-server" in servers, "neighbour `echo-server` disappeared"
PY

BACKUPS_AFTER_FIRST="$(count_backups "${CONFIG_DIR}")"
[[ "${BACKUPS_AFTER_FIRST}" -ge 1 ]] \
    || fail "expected at least one .bak.<timestamp> after first install, got ${BACKUPS_AFTER_FIRST}"

# ---------------------------------------------------------------------------
# Step 2: install again — idempotent (no second crossmem entry, no extra .bak)
# ---------------------------------------------------------------------------

crossmem install >/dev/null

python - "${CONFIG_PATH}" <<'PY' || fail "second install changed the entry shape"
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

servers = data.get("mcpServers", {})
# Exactly one crossmem entry (a dict, not a list of dicts).
assert "crossmem" in servers, "crossmem entry vanished on second install"
assert isinstance(servers["crossmem"], dict), "crossmem entry duplicated as a list"
PY

BACKUPS_AFTER_SECOND="$(count_backups "${CONFIG_DIR}")"
[[ "${BACKUPS_AFTER_SECOND}" -eq "${BACKUPS_AFTER_FIRST}" ]] \
    || fail "second install created an extra backup (before=${BACKUPS_AFTER_FIRST}, after=${BACKUPS_AFTER_SECOND})"

# ---------------------------------------------------------------------------
# Step 3: uninstall — entry removed, neighbour preserved
# ---------------------------------------------------------------------------

crossmem uninstall --yes >/dev/null 2>&1 || crossmem uninstall >/dev/null

python - "${CONFIG_PATH}" <<'PY' || fail "uninstall did not clean up correctly"
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" not in servers, "crossmem entry still present after uninstall"
assert "echo-server" in servers, "neighbour `echo-server` was removed by uninstall"
PY

# ---------------------------------------------------------------------------
# Success: emit the pass fragment and exit cleanly.
# ---------------------------------------------------------------------------

emit_report_fragment "pass"
trap - EXIT
exit 0
