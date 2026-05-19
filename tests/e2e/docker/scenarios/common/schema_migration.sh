#!/usr/bin/env bash
# schema-migration scenario (task 27.18).
#
# Verifies that a SQLite knowledge.db without a ``schema_version``
# table — the pre-versioned layout shipped by ``crossmem`` before the
# in-place migration story landed — is upgraded transparently on the
# next backend open, preserves its documents, and is idempotent on a
# second open.
#
# Why a binary fixture (``tests/e2e/fixtures/schema_v0.db``):
# regenerating the pre-versioned layout from source every run would
# couple this scenario to the very migration code under test. The
# fixture is built once by ``tests/e2e/fixtures/build_schema_v0.py``
# and checked in, so what the scenario reads is byte-identical to
# what was reviewed.
#
# Why ``crossmem doctor`` AND a direct backend open: the spec names
# ``crossmem doctor`` as the migration trigger, but in the current
# code path the migration runs inside ``SQLiteBackend._init_db`` —
# which the doctor command intentionally does *not* invoke (its
# ``backend_dim_matches_model`` check opens the DB read-only via a
# URI to avoid creating or upgrading files as a preflight side
# effect). To match the spec's intent (doctor passes against a
# legacy DB) and keep the migration assertion meaningful, the
# scenario runs doctor for the env-health gate and then opens the
# backend via Python so the on-disk PRAGMAs and ``schema_version``
# row are written. A future task that wires migration into doctor
# itself can drop the explicit backend-open step without touching
# the assertions below.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/common/schema_migration.sh"
LOG_PATH="${LOG_PATH:-reports/common/schema_migration.log}"
REPORT_PATH="${REPORT_PATH:-/tmp/crossmem-e2e-report.jsonl}"

# Fixture lookup: ``run_all.sh`` mounts ``tests/e2e/fixtures`` at
# ``/fixtures`` inside the container; standalone host runs fall back
# to a repo-relative path so the scenario stays debuggable outside
# Docker.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at ``tests/e2e/docker/scenarios/common/`` — three
# ``..`` walks land back at ``tests/e2e/`` where ``fixtures/`` sits.
DEFAULT_HOST_FIXTURES="${SCRIPT_DIR}/../../../fixtures"
FIXTURES_DIR="${FIXTURES_DIR:-/fixtures}"
if [[ ! -d "${FIXTURES_DIR}" ]]; then
    FIXTURES_DIR="${DEFAULT_HOST_FIXTURES}"
fi
FIXTURE_DB="${FIXTURES_DIR}/schema_v0.db"

START_EPOCH="$(date +%s.%N)"

FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"
CROSSMEM_DIR="${FAKE_HOME}/.crossmem"
DB_PATH="${CROSSMEM_DIR}/knowledge.db"
mkdir -p "${CROSSMEM_DIR}"

# ---------------------------------------------------------------------------
# Helpers — status tracking + report emit
# ---------------------------------------------------------------------------

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

trap 'emit_report; rm -rf "${FAKE_HOME}"' EXIT

# ---------------------------------------------------------------------------
# Step 0 — copy fixture into the fake home and assert the pre-migration shape
# ---------------------------------------------------------------------------

if [[ ! -f "${FIXTURE_DB}" ]]; then
    fail "missing fixture at ${FIXTURE_DB} (rebuild via tests/e2e/fixtures/build_schema_v0.py)"
    exit 1
fi

cp "${FIXTURE_DB}" "${DB_PATH}"

# ``CROSSMEM_DB_PATH`` makes the backend factory bypass the default
# ``~/.crossmem/knowledge.db`` resolution; ``HOME`` already points at
# the fake home but the env override keeps the scenario robust against
# future changes to ``_default_sqlite_path``.
export CROSSMEM_DB_PATH="${DB_PATH}"

echo "==> verifying pre-migration shape (no schema_version table)"
python3 - "${DB_PATH}" <<'PY' || fail "pre-migration shape check failed"
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path)
try:
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "documents" in tables, "fixture missing 'documents' table"
    assert "schema_version" not in tables, (
        "fixture must not carry a schema_version table"
    )
    ids = sorted(row[0] for row in conn.execute("SELECT id FROM documents"))
    assert ids == ["legacy-doc-1", "legacy-doc-2"], (
        f"unexpected pre-migration document ids: {ids}"
    )
finally:
    conn.close()
PY

# ---------------------------------------------------------------------------
# Step 1 — ``crossmem doctor`` against the legacy DB (env-gate sanity)
# ---------------------------------------------------------------------------

echo "==> first crossmem doctor (legacy DB on disk)"
# Doctor must not fail on a pre-versioned DB — its ``backend_dim``
# check opens the file read-only and only ``warn``s for legacy dims.
if ! crossmem doctor >/dev/null; then
    fail "crossmem doctor exited non-zero against the legacy DB"
fi

# ---------------------------------------------------------------------------
# Step 2 — trigger the in-place migration by opening the backend
# ---------------------------------------------------------------------------

echo "==> opening SQLiteBackend to trigger migration"
python3 - "${DB_PATH}" <<'PY' || fail "first backend open (migration) raised"
import sys

from crossmem.backends.sqlite_backend import SQLiteBackend

backend = SQLiteBackend(sys.argv[1])
backend.close()
PY

# ---------------------------------------------------------------------------
# Step 3 — assert (a) schema_version present, (b) legacy docs readable
# ---------------------------------------------------------------------------

echo "==> verifying post-migration shape"
python3 - "${DB_PATH}" <<'PY' || fail "post-migration assertions failed"
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path)
try:
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "schema_version" in tables, "schema_version table not created"

    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert len(rows) == 1, f"expected exactly one schema_version row, got {rows}"
    version = rows[0][0]
    # The migration must bootstrap a positive integer version. The
    # current code lands on 3 (after the v0 -> v1 -> v3 walk); we
    # assert ``>= 1`` so a future v3 -> v4 bump does not break the
    # scenario without weakening the "version was written" contract.
    assert isinstance(version, int) and version >= 1, (
        f"unexpected schema_version value: {version!r}"
    )

    ids = sorted(row[0] for row in conn.execute("SELECT id FROM documents"))
    assert ids == ["legacy-doc-1", "legacy-doc-2"], (
        f"legacy documents were not preserved: {ids}"
    )

    # Spec (b): old docs are queryable. The trigram tokenizer is the
    # whole point of the v3 rebuild, so an FTS match for a literal
    # token from a legacy doc proves the rebuild copied historic
    # rows into the new index.
    match_count = conn.execute(
        "SELECT COUNT(*) FROM fts_documents WHERE fts_documents MATCH ?",
        ("test",),
    ).fetchone()[0]
    assert match_count >= 1, (
        "post-migration FTS5 finds 0 matches for 'test' "
        "(legacy docs lost during rebuild)"
    )
finally:
    conn.close()
PY

# ---------------------------------------------------------------------------
# Step 4 — idempotency: second open must not bump version or duplicate rows
# ---------------------------------------------------------------------------

echo "==> second backend open (idempotency)"
python3 - "${DB_PATH}" <<'PY' || fail "second backend open (idempotency) raised"
import sqlite3
import sys

from crossmem.backends.sqlite_backend import SQLiteBackend

db_path = sys.argv[1]

pre_conn = sqlite3.connect(db_path)
pre_version = pre_conn.execute(
    "SELECT version FROM schema_version"
).fetchone()[0]
pre_conn.close()

backend = SQLiteBackend(db_path)
backend.close()

post_conn = sqlite3.connect(db_path)
try:
    rows = post_conn.execute(
        "SELECT version FROM schema_version"
    ).fetchall()
    assert len(rows) == 1, (
        f"second open duplicated schema_version rows: {rows}"
    )
    assert rows[0][0] == pre_version, (
        f"second open changed schema_version: {pre_version} -> {rows[0][0]}"
    )
    ids = sorted(row[0] for row in post_conn.execute("SELECT id FROM documents"))
    assert ids == ["legacy-doc-1", "legacy-doc-2"], (
        f"second open mutated document set: {ids}"
    )
finally:
    post_conn.close()
PY

# ---------------------------------------------------------------------------
# Step 5 — doctor still happy after migration (closing the loop)
# ---------------------------------------------------------------------------

echo "==> second crossmem doctor (post-migration)"
if ! crossmem doctor >/dev/null; then
    fail "crossmem doctor exited non-zero after migration"
fi

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> common/schema_migration: OK"
    exit 0
fi

echo "==> common/schema_migration: FAIL (${FAIL_REASON})" >&2
exit 1
