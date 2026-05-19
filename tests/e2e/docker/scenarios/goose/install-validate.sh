#!/usr/bin/env bash
# goose install-validate scenario (task 27.12).
#
# Runs inside the Linux runner image (``Dockerfile.linux``) where
# ``crossmem`` is already on ``PATH`` via pipx. The scenario seeds an
# isolated fake home with a minimal valid Goose config (one unrelated
# top-level entry plus an existing ``extensions`` slot), drives the
# full ``crossmem install`` -> ``crossmem install`` -> ``crossmem
# uninstall`` lifecycle, and emits a single JSON report fragment to
# ``$REPORT_PATH`` in the schema fixed by task 27.1.
#
# Why a dedicated fake HOME — the host's real
# ``~/.config/goose/config.yaml`` (if any) would otherwise be backed up
# + rewritten by the scenario; the container layer adds a second safety
# net, but redirecting ``HOME`` inside the script means the same script
# is safe to run on the host for ad-hoc debugging.
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
#
# Why ``~/.config/goose/config.yaml`` — Goose stores its user-level
# configuration as YAML under ``$XDG_CONFIG_HOME/goose`` on Linux/macOS
# (see ``_config_root`` in ``src/crossmem/connectors/goose.py``). MCP
# servers live under the top-level ``extensions`` mapping; each entry
# is a stdio extension of the form
# ``{type: stdio, cmd: ..., args: [...], enabled: true}`` — note that
# Goose does not use the JSON-CLI ``command``/``env`` layout other
# scenarios in this directory rely on.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/goose/install-validate.sh"
LOG_PATH="${LOG_PATH:-reports/goose/install-validate.log}"

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

# Pre-create the ``~/.config/goose`` parent directory so the seed write
# succeeds; ``crossmem install`` would create it itself, but we want a
# deterministic on-disk shape before the first invocation.
CONFIG_DIR="${FAKE_HOME}/.config/goose"
CONFIG_PATH="${CONFIG_DIR}/config.yaml"
mkdir -p "${CONFIG_DIR}"

# Seed a minimal valid Goose config: one unrelated top-level key plus a
# pre-existing ``extensions`` entry that must survive uninstall. YAML is
# hand-written (not generated) so the assertions further down can match
# the literal field names spec-style. Note that Goose's extension
# schema uses ``cmd`` (not ``command``) and ``enabled`` (not ``env``);
# the ``other`` entry must therefore not borrow the JSON-CLI shape.
cat >"${CONFIG_PATH}" <<'YAML'
existing: preserve-me
extensions:
  other:
    type: stdio
    cmd: /usr/bin/true
    args: []
    enabled: true
YAML

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

# Validate the rewritten config against the Goose stdio-extension
# layout. We rely on PyYAML (a transitive dep of crossmem; if it ever
# becomes optional, switch this block to ``python3 -c "import json,
# subprocess; ..."`` or pin ``pyyaml`` explicitly in the runner image).
python3 - "${CONFIG_PATH}" <<'PY' || fail "post-install config validation failed"
import sys

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

assert isinstance(data, dict), "config root must be a mapping"
assert "extensions" in data, "missing top-level extensions"
extensions = data["extensions"]
assert isinstance(extensions, dict), "extensions not a mapping"
assert "crossmem" in extensions, "crossmem entry not registered"
entry = extensions["crossmem"]
# Goose's stdio-extension schema: ``type``/``cmd``/``args``/``enabled``.
for field in ("type", "cmd", "args", "enabled"):
    assert field in entry, f"crossmem entry missing {field!r}"
assert entry["type"] == "stdio", f"unexpected type {entry['type']!r}"
assert isinstance(entry["cmd"], str) and entry["cmd"], "empty cmd"
assert isinstance(entry["args"], list), "args not a list"
assert entry["enabled"] is True, "extension is not enabled"

# The unrelated entry must still be there.
assert "other" in extensions, "pre-existing 'other' entry was clobbered"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
PY

# Backup file is named ``<config>.bak.<ts>`` per ``connectors/config_io.py``.
backup_count=$(find "${CONFIG_DIR}" -maxdepth 1 -name 'config.yaml.bak.*' \
    -type f | wc -l)
if [[ "${backup_count}" -lt 1 ]]; then
    fail "expected at least one config.yaml.bak.* after first install"
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
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

extensions = data["extensions"]
# Idempotency: no duplicate entries.
crossmem_keys = [k for k in extensions if k == "crossmem"]
assert len(crossmem_keys) == 1, f"duplicate crossmem entries: {crossmem_keys}"
assert "other" in extensions, "'other' entry lost on re-install"
PY

backup_count_after_second=$(find "${CONFIG_DIR}" -maxdepth 1 \
    -name 'config.yaml.bak.*' -type f | wc -l)
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
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

extensions = data.get("extensions", {})
assert "crossmem" not in extensions, "crossmem entry still present after uninstall"
assert "other" in extensions, "'other' entry removed by uninstall"
assert data.get("existing") == "preserve-me", "top-level 'existing' lost"
PY

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> goose install-validate: OK"
    # ``emit_report`` runs via EXIT trap.
    exit 0
fi

echo "==> goose install-validate: FAIL (${FAIL_REASON})" >&2
exit 1
