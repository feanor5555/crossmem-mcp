#!/usr/bin/env bash
# Continue.dev CLI-config validation scenario for task 27.10.
#
# Runs inside the Linux E2E container produced by ``tests/e2e/docker/
# Dockerfile.linux`` and exercises the full ``crossmem install`` ->
# validate -> ``crossmem install`` (idempotency) -> ``crossmem uninstall``
# cycle against a Fake-Home that already contains a Continue.dev MCP
# config with a neighbour entry. Layout follows the analog scenarios
# for Claude Code (27.4) and Cursor (27.5) so the merge logic in
# ``run_all.sh`` does not need a per-CLI special case.
#
# Continue.dev keeps its user config in ``~/.continue/`` on every
# platform with two on-disk variants in parallel:
#
#   * JSON (legacy / 1.x) — ``~/.continue/config.json`` with MCP servers
#     under the nested key ``experimental.modelContextProtocolServers``.
#   * YAML (2.x) — ``~/.continue/config.yaml`` with MCP servers under a
#     flat top-level ``mcpServers`` mapping.
#
# When both files exist the YAML variant wins (see
# ``ContinueDevConnector.register`` in
# ``src/crossmem/connectors/continuedev.py``). We seed the YAML variant
# here because it is the active path on a fresh Continue 2.x install —
# the JSON branch is covered by the connector's unit tests.
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
# run idempotent w.r.t. the developer's real ``~/.continue/`` and keeps
# the suite safe to re-run.

set -euo pipefail

SCENARIO_NAME="scenarios/continue-dev/install-validate.sh"
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
    # Count ``config.yaml.bak.*`` siblings of the Continue.dev config.
    # The connector uses :func:`backup_config` from
    # ``connectors/config_io.py`` which writes ``<config>.bak.<ts>``;
    # ``ls -1`` is portable across coreutils versions and ``2>/dev/null``
    # swallows the "no match" stderr.
    local dir="$1"
    ls -1 "${dir}"/config.yaml.bak.* 2>/dev/null | wc -l | tr -d ' '
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
# Seed a Fake-Home with a minimal valid Continue.dev YAML config
# ---------------------------------------------------------------------------

FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"

CONFIG_DIR="${FAKE_HOME}/.continue"
CONFIG_PATH="${CONFIG_DIR}/config.yaml"
mkdir -p "${CONFIG_DIR}"

# Seed the YAML 2.x variant: an unrelated top-level key plus a neighbour
# entry under ``mcpServers``. The neighbour proves "andere Eintraege
# bleiben unveraendert" after uninstall; the top-level ``name`` key
# (Continue.dev's spec field for the active assistant) proves the
# connector does not over-reach beyond the ``mcpServers`` subtree.
cat >"${CONFIG_PATH}" <<'YAML'
name: e2e-assistant
mcpServers:
  echo-server:
    command: echo
    args:
      - hello
    env: {}
YAML

# ---------------------------------------------------------------------------
# Step 1: install once, validate the written entry + backup
# ---------------------------------------------------------------------------

crossmem install >/dev/null

# Field-level assertions via ``python -c`` — ``yq`` is not guaranteed
# inside the base image, and PyYAML ships with the wheel anyway.
python - "${CONFIG_PATH}" <<'PY' || fail "first install did not write the expected entry"
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

assert isinstance(data, dict), "config root must be a mapping"
servers = data.get("mcpServers")
assert isinstance(servers, dict), "missing top-level mcpServers mapping"
assert "crossmem" in servers, "missing `crossmem` entry under mcpServers"
entry = servers["crossmem"]
for key in ("command", "args", "env"):
    assert key in entry, f"missing `{key}` in crossmem entry"
assert isinstance(entry["command"], str), "`command` must be a string"
assert isinstance(entry["args"], list), "`args` must be a list"
assert isinstance(entry["env"], dict), "`env` must be a mapping"

# The neighbour must still be present, and the unrelated top-level key too.
assert "echo-server" in servers, "neighbour `echo-server` disappeared"
assert data.get("name") == "e2e-assistant", "top-level `name` key was clobbered"
PY

BACKUPS_AFTER_FIRST="$(count_backups "${CONFIG_DIR}")"
[[ "${BACKUPS_AFTER_FIRST}" -ge 1 ]] \
    || fail "expected at least one .bak.<timestamp> after first install, got ${BACKUPS_AFTER_FIRST}"

# ---------------------------------------------------------------------------
# Step 2: install again — idempotent (no second crossmem entry, no extra .bak)
# ---------------------------------------------------------------------------

crossmem install >/dev/null

python - "${CONFIG_PATH}" <<'PY' || fail "second install changed the entry shape"
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

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
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

servers = data.get("mcpServers", {})
assert "crossmem" not in servers, "crossmem entry still present after uninstall"
assert "echo-server" in servers, "neighbour `echo-server` was removed by uninstall"
assert data.get("name") == "e2e-assistant", "top-level `name` key was removed by uninstall"
PY

# ---------------------------------------------------------------------------
# Success: emit the pass fragment and exit cleanly.
# ---------------------------------------------------------------------------

emit_report_fragment "pass"
trap - EXIT
exit 0
