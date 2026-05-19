#!/usr/bin/env bash
# import-export-roundtrip scenario (task 27.17).
#
# This is a "common" scenario — it does not exercise any single CLI
# connector, it exercises the export/import contract of crossmem
# itself end-to-end. The script:
#
#   1. seeds 100 documents whose ``content`` covers de/en/fr/ja/ar
#      (plus emoji and RTL bidi markers) so the JSONL UTF-8 path is
#      truly multilingual;
#   2. exports the DB to a ZIP via ``crossmem export``;
#   3. wipes the on-disk SQLite (including ``-wal`` / ``-shm`` side
#      files so the next process opens a fresh DB instead of replaying
#      an unflushed journal);
#   4. imports the ZIP back via ``crossmem import``;
#   5. asserts (id, content, tags, embedding) identity per document
#      between the pre-export snapshot and the post-import DB;
#   6. drives three negative cases against tampered exports — Zip-Slip
#      payload, wrong EOF sha-256, and a doc with the wrong embedding
#      dimension — each must be rejected by ``crossmem import`` with a
#      non-zero exit so a future regression in the importer's
#      validation trips the scenario.
#
# Why a dedicated fake HOME — same reasoning as the per-CLI scenarios
# under ``scenarios/<cli>/``: redirecting ``$HOME`` keeps the host's
# real ``~/.crossmem/knowledge.db`` untouched and lets ad-hoc local
# debugging be safe to re-run. ``CROSSMEM_DB_PATH`` is set explicitly
# so a stray ``crossmem configure`` artefact in the fake HOME cannot
# divert the DB location.
#
# Why deterministic synthetic embeddings — the importer's contract is
# bit-for-bit identity of the serialized payload (the EOF marker
# hashes the JSONL lines), not a property of the embedding model. A
# fixed pseudo-embedding derived from SHA-256(content) keeps the
# scenario hermetic (no 300MB fastembed download per run) while still
# exercising the real ``KnowledgeStore.export``/``import_data``
# code paths through the ``crossmem`` CLI.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup — isolated fake home + report bookkeeping
# ---------------------------------------------------------------------------

SCENARIO_NAME="scenarios/common/import_export.sh"
LOG_PATH="${LOG_PATH:-reports/common/import_export.log}"

# ``$REPORT_PATH`` is supplied by ``run_all.sh`` / ``run_all.ps1``; the
# fallback keeps the script runnable stand-alone for developer
# debugging without tripping ``set -u``.
REPORT_PATH="${REPORT_PATH:-/tmp/crossmem-e2e-report.jsonl}"

START_EPOCH="$(date +%s.%N)"

FAKE_HOME="$(mktemp -d)"
export HOME="${FAKE_HOME}"
mkdir -p "${FAKE_HOME}/.crossmem"

# Pin the DB inside the fake home; ``crossmem`` honours this env even
# when no ``config.toml`` exists (defaults to sqlite).
export CROSSMEM_DB_PATH="${FAKE_HOME}/.crossmem/knowledge.db"

# Workspace for export artefacts and tampered fixtures.
WORK_DIR="${FAKE_HOME}/work"
mkdir -p "${WORK_DIR}"

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
# Step 1 — seed 100 multilingual docs directly into the SQLite backend
# ---------------------------------------------------------------------------
#
# We bypass ``KnowledgeStore.store`` (which would load fastembed and
# pull the ~300MB model on first use) and write Documents straight to
# the backend with deterministic synthetic embeddings. The exporter
# does not care how the embeddings were produced — it serialises the
# vectors verbatim and the importer validates length + content hash.
# A 384-dim vector derived from SHA-256(content) is unique per doc and
# survives the JSON round-trip without floating-point drift because
# every component is an exactly representable 1/255 multiple.

echo "==> seeding 100 multilingual documents"
python3 - "${CROSSMEM_DB_PATH}" "${WORK_DIR}/snapshot.json" <<'PY' \
    || fail "seed step failed"
import hashlib
import json
import struct
import sys
from pathlib import Path

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.embedding import EMBEDDING_DIM
from crossmem.core.models import Document, generate_content_hash, generate_id

db_path = Path(sys.argv[1])
snapshot_path = Path(sys.argv[2])

# Five language buckets — Latin (de/en/fr), CJK (ja), and RTL (ar) —
# plus an emoji marker and an explicit U+200F RTL mark on the Arabic
# sentence so the JSONL encoder is exercised across UTF-8 ranges.
SAMPLES = [
    ("de", "Schoenes Wetter heute. Die Spatzen zwitschern auf dem Dach."),
    ("en", "The quick brown fox jumps over the lazy dog. Test {i}."),
    ("fr", "Voici un petit texte en francais avec des accents: e a c."),
    ("ja", "こんにちは、世界。テスト {i}."),  # konnichiwa, sekai
    ("ar", "‏مرحبًا بالعالم — {i}."),  # marhaban bil-aalam + RTL mark
]
EMOJI = "\U0001f600 \U0001f30d \U0001f44d"


def synth_embedding(content: str) -> list[float]:
    """384-dim deterministic vector derived from SHA-256 of content.

    Each output byte is rescaled to ``b / 255.0`` and then passed through
    a one-way fp32 quantization step (pack-as-``f`` then unpack-as-``f``)
    so the value carried in the Python snapshot exactly equals what the
    SQLite backend will read back after its own fp32 storage round-trip.
    Without the quantization the snapshot would be a fp64 number while
    the post-import value would lose the lower 29 mantissa bits — and a
    bit-exact comparison would fail for ~half of the components.
    """
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    assert EMBEDDING_DIM % len(digest) == 0
    repeats = EMBEDDING_DIM // len(digest)
    raw = [b / 255.0 for b in (digest * repeats)]
    # Same fp32 round-trip the SQLite backend performs via ``struct.pack``.
    packed = struct.pack(f"{len(raw)}f", *raw)
    return list(struct.unpack(f"{len(raw)}f", packed))


backend = SQLiteBackend(db_path)
snapshot: list[dict] = []
for i in range(100):
    lang, template = SAMPLES[i % len(SAMPLES)]
    content = template.format(i=i) + " " + EMOJI
    source_url = f"https://example.com/{lang}/doc-{i:03d}"
    title = f"Doc {i:03d} ({lang})"
    tags = sorted({lang, f"bucket-{i % 7}", "multilingual"})
    embedding = synth_embedding(content)
    content_hash = generate_content_hash(content)
    doc_id = generate_id("default", source_url, content_hash)
    doc = Document.from_dict(
        {
            "id": doc_id,
            "content": content,
            "embedding": embedding,
            "metadata": {
                "source_url": source_url,
                "title": title,
                "source_type": "web",
                "stored_at": "2026-01-01T00:00:00Z",
                "embedding_model": "synthetic-sha256",
                "embedding_dim": EMBEDDING_DIM,
                "namespace": "default",
                "tags": tags,
                "content_hash": content_hash,
            },
        }
    )
    backend.upsert_many([doc])
    snapshot.append(
        {
            "id": doc.id,
            "content": doc.content,
            "tags": list(doc.metadata.tags),
            "embedding": list(doc.embedding),
            "source_url": doc.metadata.source_url,
        }
    )
backend.close()

snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
print(f"seeded {len(snapshot)} docs into {db_path}")
PY

# ---------------------------------------------------------------------------
# Step 2 — export to ZIP via the CLI
# ---------------------------------------------------------------------------

EXPORT_ZIP="${WORK_DIR}/dump.zip"
echo "==> exporting to ${EXPORT_ZIP}"
if ! crossmem export --path "${EXPORT_ZIP}" --format zip >/dev/null; then
    fail "crossmem export exited non-zero"
fi
if [[ ! -s "${EXPORT_ZIP}" ]]; then
    fail "export ZIP is empty or missing"
fi

# ---------------------------------------------------------------------------
# Step 3 — wipe the DB, then import the ZIP back into a fresh DB
# ---------------------------------------------------------------------------

echo "==> wiping ${CROSSMEM_DB_PATH} (and WAL/SHM)"
rm -f "${CROSSMEM_DB_PATH}" "${CROSSMEM_DB_PATH}-wal" "${CROSSMEM_DB_PATH}-shm"

echo "==> importing ${EXPORT_ZIP}"
if ! crossmem import --path "${EXPORT_ZIP}" >/dev/null; then
    fail "crossmem import exited non-zero on a valid export"
fi

# ---------------------------------------------------------------------------
# Step 4 — compare snapshot against the post-import DB
# ---------------------------------------------------------------------------

echo "==> verifying post-import identity"
python3 - "${CROSSMEM_DB_PATH}" "${WORK_DIR}/snapshot.json" <<'PY' \
    || fail "post-import identity check failed"
import json
import sys
from pathlib import Path

from crossmem.backends.sqlite_backend import SQLiteBackend

db_path = Path(sys.argv[1])
snapshot = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

backend = SQLiteBackend(db_path)
by_id = {d.id: d for d in backend.iter_all()}
backend.close()

assert len(by_id) == len(snapshot), (
    f"doc count mismatch: snapshot={len(snapshot)} db={len(by_id)}"
)
for entry in snapshot:
    got = by_id.get(entry["id"])
    assert got is not None, f"missing id after import: {entry['id']}"
    assert got.content == entry["content"], f"content drift on {entry['id']}"
    assert list(got.metadata.tags) == entry["tags"], (
        f"tag drift on {entry['id']}: {got.metadata.tags!r} vs {entry['tags']!r}"
    )
    # Embeddings are exact 1/255 multiples encoded as JSON numbers, so
    # bit-equality across the JSON round-trip is the right check.
    assert list(got.embedding) == entry["embedding"], (
        f"embedding drift on {entry['id']}"
    )
    assert got.metadata.source_url == entry["source_url"]
print(f"verified {len(snapshot)} docs match by id, content, tags, embedding")
PY

# ---------------------------------------------------------------------------
# Step 5 — negative cases (Zip-Slip / wrong EOF sha / dim mismatch)
# ---------------------------------------------------------------------------
#
# Each negative case writes a tampered file and then invokes
# ``crossmem import`` against it. The CLI must exit non-zero. We use
# the ``!`` shell idiom so ``set -e`` does not abort the script on the
# expected failure — the assertion is "the importer rejected it".

assert_import_rejects() {
    local label="$1"
    local file_path="$2"
    set +e
    crossmem import --path "${file_path}" >/dev/null 2>&1
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
        fail "${label}: crossmem import accepted a tampered file"
    else
        echo "    ${label}: rejected (rc=${rc}) — ok"
    fi
}

# --- (a) Zip-Slip ------------------------------------------------------
#
# A ZIP whose member name traverses outside the archive root must be
# rejected up-front (``_ensure_safe_zip_member``). Note we keep the
# legitimate ``documents.jsonl`` entry *and* add a traversal member —
# the importer must reject the archive as a whole, not silently skip
# the bad member.

echo "==> negative case (a): Zip-Slip"
ZIPSLIP_ZIP="${WORK_DIR}/zipslip.zip"
python3 - "${ZIPSLIP_ZIP}" "${EXPORT_ZIP}" <<'PY' \
    || fail "failed to construct Zip-Slip fixture"
import shutil
import sys
import zipfile
from pathlib import Path

target = Path(sys.argv[1])
source = Path(sys.argv[2])
shutil.copyfile(source, target)
# Append a traversal entry to the otherwise-valid archive so the
# importer must inspect every member name, not just the first one.
with zipfile.ZipFile(target, "a") as zf:
    zf.writestr("../../etc/passwd", "root:x:0:0:root:/root:/bin/bash\n")
PY
assert_import_rejects "zip-slip" "${ZIPSLIP_ZIP}"

# --- (b) Wrong EOF sha-256 --------------------------------------------
#
# Decompress the legitimate ZIP, replace the EOF marker's ``sha256``
# field with a known-bad digest, repack, and feed it to the importer.
# The hash-check is the only thing standing between a silently
# rewritten export and a corrupted DB — a regression here is silent.

echo "==> negative case (b): wrong EOF sha"
BADSHA_ZIP="${WORK_DIR}/badsha.zip"
python3 - "${BADSHA_ZIP}" "${EXPORT_ZIP}" <<'PY' \
    || fail "failed to construct bad-sha fixture"
import json
import sys
import zipfile
from pathlib import Path

target = Path(sys.argv[1])
source = Path(sys.argv[2])

with zipfile.ZipFile(source) as zf:
    payload = zf.read("documents.jsonl").decode("utf-8")

lines = [ln for ln in payload.splitlines() if ln.strip()]
eof = json.loads(lines[-1])
assert eof.get("type") == "eof", f"unexpected EOF line: {lines[-1]!r}"
eof["sha256"] = "0" * 64  # deterministic invalid digest
lines[-1] = json.dumps(eof, sort_keys=True)
tampered = "\n".join(lines) + "\n"

with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("documents.jsonl", tampered)
PY
assert_import_rejects "bad-sha" "${BADSHA_ZIP}"

# --- (c) Embedding dimension mismatch ---------------------------------
#
# Construct a tiny valid export with a single document whose embedding
# is 256-dim instead of 384. The EOF sha must match the tampered
# content lines so we are testing the dim check itself, not the hash
# check; otherwise the importer would reject on the marker before it
# ever looked at the embedding.

echo "==> negative case (c): embedding dim mismatch"
BADDIM_ZIP="${WORK_DIR}/baddim.zip"
python3 - "${BADDIM_ZIP}" <<'PY' \
    || fail "failed to construct dim-mismatch fixture"
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from crossmem.core.models import generate_content_hash, generate_id

target = Path(sys.argv[1])

content = "Document with the wrong embedding dimension."
source_url = "https://example.com/baddim"
content_hash = generate_content_hash(content)
doc_id = generate_id("default", source_url, content_hash)

doc = {
    "id": doc_id,
    "content": content,
    "embedding": [0.0] * 256,  # spec demands 384 — this must be rejected
    "metadata": {
        "source_url": source_url,
        "title": "Bad dim",
        "source_type": "web",
        "stored_at": "2026-01-01T00:00:00Z",
        "embedding_model": "synthetic-baddim",
        "embedding_dim": 256,
        "namespace": "default",
        "tags": ["baddim"],
        "content_hash": content_hash,
    },
}
line = json.dumps(doc, sort_keys=True, ensure_ascii=False)
encoded = (line + "\n").encode("utf-8")
hasher = hashlib.sha256()
hasher.update(encoded)
eof = {"type": "eof", "count": 1, "sha256": hasher.hexdigest()}
payload = (line + "\n" + json.dumps(eof, sort_keys=True) + "\n").encode("utf-8")

with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("documents.jsonl", payload)
PY
assert_import_rejects "bad-dim" "${BADDIM_ZIP}"

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if [[ "${STATUS}" == "pass" ]]; then
    echo "==> import-export-roundtrip: OK"
    exit 0
fi

echo "==> import-export-roundtrip: FAIL (${FAIL_REASON})" >&2
exit 1
