#!/usr/bin/env bash
# Linux entry-point for the CrossMem E2E suite (task 27.1).
#
# Responsibilities — kept deliberately small so the script stays
# auditable in one screen and reusable for tasks 27.4+ (per-CLI
# scenarios append themselves to the same report):
#
#   1. Build the linux runner image from ``docker/Dockerfile.linux``
#      with the repo root as build context (so the working-copy gets
#      baked into the image).
#   2. Run each scenario script inside that image, capturing stdout
#      + stderr into ``reports/<timestamp>/<scenario>.log``.
#   3. Emit ``reports/<timestamp>.json`` with the schema mandated by
#      TODO.md 27.1:
#         {"runner": "linux", "scenarios": [...],
#          "started_at": "...", "finished_at": "..."}
#
# The script is the human-facing manual test for 27.1 (DoD: ``bash
# tests/e2e/run_all.sh`` exits 0 and a valid JSON report appears under
# ``reports/``). It is **not** invoked from pytest — the python tests
# in ``tests/e2e_runner/`` only validate the script's *shape*.

set -euo pipefail

# Resolve paths relative to this script, not to the caller's CWD —
# letting ``bash tests/e2e/run_all.sh`` from the repo root behave
# identically to ``cd tests/e2e && bash run_all.sh``.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/docker/Dockerfile.linux"
REPORTS_DIR="${SCRIPT_DIR}/reports"
IMAGE_TAG="${CROSSMEM_E2E_LINUX_IMAGE:-crossmem-e2e:linux}"

# ``date -u +%Y%m%dT%H%M%SZ`` produces basic-format ISO-8601 with no
# colons — safe on Windows filesystems when the same report is copied
# off a Linux runner. ``started_at`` uses extended format for human
# readability inside the JSON.
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPORT_JSON="${REPORTS_DIR}/${TIMESTAMP}.json"
LOG_DIR="${REPORTS_DIR}/${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

echo "==> building ${IMAGE_TAG} from ${DOCKERFILE}"
docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_TAG}" \
    "${REPO_ROOT}"

# Each scenario is one entry in this array; the smoke scenario is the
# only one for task 27.1, later tasks (27.4 ...) prepend their own
# scenarios above the smoke one or replace the loop with a discovery
# step. Keeping the array explicit makes the report deterministic.
SCENARIOS=(
    "scenarios/_smoke/hello.sh"
    "scenarios/common/schema_migration.sh"
)

# Per-scenario JSON fragments accumulate here; we join them with
# commas at the end so the final document stays valid even when the
# loop runs zero or many times.
SCENARIO_JSON=()

for scenario in "${SCENARIOS[@]}"; do
    name="${scenario}"
    log_file="${LOG_DIR}/$(basename "${scenario}" .sh).log"
    rel_log="reports/${TIMESTAMP}/$(basename "${scenario}" .sh).log"
    echo "==> running scenario ${name}"

    scenario_start_epoch="$(date +%s.%N)"
    set +e
    # ``tests/e2e/fixtures`` is mounted read-only at ``/fixtures`` so
    # scenarios (e.g. ``common/schema_migration.sh``) can copy known
    # legacy DB layouts into the container's fake-home without having
    # to regenerate them at run time. The mount is benign for scenarios
    # that don't read from it (Docker silently ignores unused volumes).
    docker run --rm \
        --volume "${SCRIPT_DIR}/docker/scenarios:/scenarios:ro" \
        --volume "${SCRIPT_DIR}/fixtures:/fixtures:ro" \
        --workdir /work \
        "${IMAGE_TAG}" \
        bash "/scenarios/$(basename "$(dirname "${scenario}")")/$(basename "${scenario}")" \
        >"${log_file}" 2>&1
    scenario_exit=$?
    set -e
    scenario_end_epoch="$(date +%s.%N)"

    # ``bc`` keeps the math portable; ``awk`` would work too but is a
    # heavier hammer for one subtraction. The format ``%.3f`` gives
    # ms-precision which matches the spec ("<float>") without leaking
    # nanosecond noise into the report.
    duration_s="$(awk -v s="${scenario_start_epoch}" -v e="${scenario_end_epoch}" \
        'BEGIN { printf "%.3f", e - s }')"

    if [[ "${scenario_exit}" -eq 0 ]]; then
        status="pass"
    else
        status="fail"
    fi

    SCENARIO_JSON+=("$(printf '{"name": "%s", "status": "%s", "duration_s": %s, "log_path": "%s"}' \
        "${name}" "${status}" "${duration_s}" "${rel_log}")")
done

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Join the per-scenario fragments with ``, `` — bash arrays do not
# have a built-in join so we use the ``IFS``+``${array[*]}`` trick.
_old_ifs="${IFS}"
IFS=", "
joined="${SCENARIO_JSON[*]}"
IFS="${_old_ifs}"

# Assemble the final report. printf is preferable to a heredoc here
# because every field is interpolated and the layout stays compact.
# ``mktemp`` + ``mv`` makes the write atomic — readers (CI, humans
# tailing the dir) never observe a half-written file.
tmp_report="$(mktemp "${REPORTS_DIR}/.${TIMESTAMP}.XXXXXX.json")"
printf '{"runner": "linux", "scenarios": [%s], "started_at": "%s", "finished_at": "%s"}\n' \
    "${joined}" "${STARTED_AT}" "${FINISHED_AT}" \
    >"${tmp_report}"
mv "${tmp_report}" "${REPORT_JSON}"

echo "==> wrote ${REPORT_JSON}"

# Exit non-zero if any scenario failed so callers (CI, the developer's
# shell) see the failure even when the JSON report itself was written
# successfully.
if printf '%s\n' "${SCENARIO_JSON[@]}" | grep -q '"status": "fail"'; then
    echo "==> at least one scenario failed; see ${LOG_DIR}/" >&2
    exit 1
fi
