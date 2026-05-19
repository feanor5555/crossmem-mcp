"""Tests for ``crossmem doctor --json`` stable schema output (task 17.1).

The JSON output is documented by ``schemas/doctor.json`` and consumed by
LLM-driven install flows (per-CLI install guides) plus future CI tooling.
The shape MUST be stable across crossmem versions:

* Existing top-level keys (``version``, ``checks``, ``summary``) are never
  renamed or removed.
* Existing check ``name`` values are never renamed or removed.
* Status values stay ``ok`` / ``warn`` / ``fail``.
* New checks MAY be appended; new top-level keys MAY be added — additive
  evolution only.

These tests pin that contract via JSON-Schema validation + explicit key
asserts. The schema file itself is committed at ``schemas/doctor.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem import cli
from crossmem.doctor import CheckResult

jsonschema = pytest.importorskip("jsonschema")


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "doctor.json"


# ---------------------------------------------------------------------------
# Schema file itself
# ---------------------------------------------------------------------------


def test_schema_file_exists() -> None:
    """The schema file ships with the repo at a stable path."""
    assert SCHEMA_PATH.is_file(), f"missing schema file at {SCHEMA_PATH}"


def test_schema_file_is_valid_json_schema() -> None:
    """The schema document itself must be a syntactically valid JSON Schema."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # ``check_schema`` raises on a malformed schema definition.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_declares_2020_12_draft() -> None:
    """We pin Draft 2020-12 to keep tooling expectations consistent."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"


# ---------------------------------------------------------------------------
# CLI output validates against the schema
# ---------------------------------------------------------------------------


def _sample_results() -> list[CheckResult]:
    return [
        CheckResult(name="python_version", status="ok", detail="Python 3.12.5"),
        CheckResult(name="module_fastembed", status="ok", detail="fastembed import ok"),
        CheckResult(
            name="optional_chromadb",
            status="warn",
            detail="chromadb not installed (optional).",
        ),
        CheckResult(
            name="db_dir_writable",
            status="fail",
            detail="cannot create /home/x/.crossmem: permission denied",
        ),
    ]


def _run_doctor_json(monkeypatch: pytest.MonkeyPatch, capsys) -> dict:
    monkeypatch.setattr(cli, "run_checks", _sample_results)
    exit_code = cli.main(["doctor", "--json"])
    captured = capsys.readouterr()
    # One fail in the sample -> exit code 1.
    assert exit_code == 1
    return json.loads(captured.out)


def test_doctor_json_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Doctor's JSON output validates cleanly against ``schemas/doctor.json``."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = _run_doctor_json(monkeypatch, capsys)
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_doctor_json_real_run_validates_against_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real ``run_checks()`` invocation also matches the schema.

    Guards against drift between the schema and what the in-tree checks
    actually emit (e.g. a new check that produces a non-conforming name).
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    exit_code = cli.main(["doctor", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    jsonschema.Draft202012Validator(schema).validate(payload)
    # Exit code is determined by the real environment — just sanity-check it's int.
    assert exit_code in {0, 1}


# ---------------------------------------------------------------------------
# Key stability (explicit asserts on top-level + per-check fields)
# ---------------------------------------------------------------------------


def test_doctor_json_top_level_keys_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level shape pins: object with ``version``, ``checks``, ``summary``."""
    payload = _run_doctor_json(monkeypatch, capsys)
    assert isinstance(payload, dict)
    # The trio of stable top-level keys.
    assert {"version", "checks", "summary"}.issubset(payload.keys())
    # Schema version is a string constant for this release line.
    assert payload["version"] == "1"


def test_doctor_json_checks_per_item_keys_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every check entry has exactly the stable keys ``name``/``status``/``detail``."""
    payload = _run_doctor_json(monkeypatch, capsys)
    assert isinstance(payload["checks"], list)
    assert payload["checks"], "expected at least one check"
    for item in payload["checks"]:
        assert set(item.keys()) == {"name", "status", "detail"}
        assert item["status"] in {"ok", "warn", "fail"}
        assert isinstance(item["name"], str) and item["name"]
        assert isinstance(item["detail"], str)


def test_doctor_json_summary_counts_match_checks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``summary.{ok,warn,fail,total}`` is consistent with ``checks``."""
    payload = _run_doctor_json(monkeypatch, capsys)
    summary = payload["summary"]
    assert set(summary.keys()) == {"ok", "warn", "fail", "total"}
    checks = payload["checks"]
    expected = {"ok": 0, "warn": 0, "fail": 0}
    for item in checks:
        expected[item["status"]] += 1
    assert summary["ok"] == expected["ok"]
    assert summary["warn"] == expected["warn"]
    assert summary["fail"] == expected["fail"]
    assert summary["total"] == len(checks)
    assert summary["total"] == summary["ok"] + summary["warn"] + summary["fail"]


def test_doctor_json_preserves_check_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``checks`` is emitted in the same order as ``run_checks()``."""
    payload = _run_doctor_json(monkeypatch, capsys)
    expected_names = [r.name for r in _sample_results()]
    actual_names = [c["name"] for c in payload["checks"]]
    assert actual_names == expected_names


def test_doctor_json_emits_no_color_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON output never contains ANSI escapes, even on a TTY."""
    monkeypatch.setattr(cli, "run_checks", _sample_results)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli.main(["doctor", "--json"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_doctor_json_known_check_names_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real run emits the documented set of checks.

    This is the stability anchor: if you rename a check, this test fails
    and you have to either revert the rename or bump ``schema.version``
    (which is a breaking change requiring a deprecation cycle).
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cli.main(["doctor", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    names = {c["name"] for c in payload["checks"]}
    documented = {
        "python_version",
        "module_fastembed",
        "module_sqlite_vec",
        "module_fastmcp",
        "module_httpx",
        "module_bs4",
        "module_yaml",
        "optional_chromadb",
        "optional_qdrant_client",
        "db_dir_writable",
        "embedding_cache_reachable",
        "embedding_model_supported",
    }
    missing = documented - names
    assert not missing, f"documented checks missing from output: {missing}"
