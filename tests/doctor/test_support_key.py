"""Tests for the ``support`` top-level key in ``crossmem doctor --json`` (task 19.1).

``skills/crossmem-install/SKILL.md`` instructs the LLM to surface the project
issue tracker via ``crossmem doctor --json``'s ``support`` key. The key carries
``issues_url`` (mandatory — that's the SKILL.md expectation), plus the
informational companions ``docs_url`` and ``version`` so an LLM has everything
needed to file a useful bug report without a second tool call.

The key is additive — ``schemas/doctor.json`` declares
``additionalProperties: true`` at the top level, so adding ``support`` keeps
the schema version at ``"1"`` and old consumers ignore it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem import cli
from crossmem.doctor import CheckResult, build_support_info

jsonschema = pytest.importorskip("jsonschema")


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "doctor.json"


def _sample_results() -> list[CheckResult]:
    return [
        CheckResult(name="python_version", status="ok", detail="Python 3.12.5"),
    ]


def _run_doctor_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> dict:
    monkeypatch.setattr(cli, "run_checks", _sample_results)
    cli.main(["doctor", "--json"])
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# build_support_info() shape
# ---------------------------------------------------------------------------


def test_build_support_info_returns_required_fields() -> None:
    """``support`` carries the SKILL.md-expected ``issues_url`` plus companions."""
    support = build_support_info()
    assert isinstance(support, dict)
    assert {"issues_url", "docs_url", "version"}.issubset(support.keys())
    for key in ("issues_url", "docs_url", "version"):
        assert isinstance(support[key], str) and support[key]


def test_build_support_info_issues_url_points_at_project_tracker() -> None:
    """The issue-tracker URL is the canonical GitHub Issues page for crossmem.

    SKILL.md tells the LLM the URL is "the project issue tracker"; surfacing
    anything else (e.g. the repo root) would mislead the LLM into the wrong
    landing page for filing bugs.
    """
    support = build_support_info()
    assert support["issues_url"].startswith("https://")
    assert "github.com/feanor5555/knowledge" in support["issues_url"]
    assert support["issues_url"].rstrip("/").endswith("/issues")


def test_build_support_info_docs_url_is_https() -> None:
    """``docs_url`` is a stable HTTPS URL the LLM can hand to the user."""
    support = build_support_info()
    assert support["docs_url"].startswith("https://")
    assert "github.com/feanor5555/knowledge" in support["docs_url"]


def test_build_support_info_version_matches_package_metadata() -> None:
    """``version`` is the installed crossmem package version (importlib.metadata)."""
    from importlib import metadata

    try:
        expected = metadata.version("crossmem")
    except metadata.PackageNotFoundError:
        expected = "0.0.0"
    assert build_support_info()["version"] == expected


def test_build_support_info_version_falls_back_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``crossmem`` isn't installed (source-tree run), ``version`` is ``"0.0.0"``.

    Mirrors the fallback used by ``crossmem.docs.install_template._crossmem_version``
    so the doctor payload stays well-formed even on a fresh checkout that hasn't
    been ``pip install``-ed yet.
    """
    from importlib import metadata

    from crossmem import doctor

    def _raise(_name: str) -> str:
        raise metadata.PackageNotFoundError("crossmem")

    monkeypatch.setattr(doctor.metadata, "version", _raise)
    assert doctor.build_support_info()["version"] == "0.0.0"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_doctor_json_emits_support_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI's JSON payload includes a top-level ``support`` object."""
    payload = _run_doctor_json(monkeypatch, capsys)
    assert "support" in payload
    assert isinstance(payload["support"], dict)
    assert {"issues_url", "docs_url", "version"}.issubset(payload["support"].keys())


def test_doctor_json_support_matches_build_support_info(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI surfaces exactly what ``build_support_info()`` produces — no drift."""
    payload = _run_doctor_json(monkeypatch, capsys)
    assert payload["support"] == build_support_info()


def test_doctor_json_with_support_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Adding the ``support`` key keeps the payload schema-valid (additive)."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = _run_doctor_json(monkeypatch, capsys)
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_doctor_json_support_issues_url_matches_skill_md_expectation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI output's ``support.issues_url`` is the issue tracker SKILL.md refers to.

    SKILL.md tells the LLM:

        The project issue tracker — link surfaced by `crossmem doctor --json`
        under the `support` key.

    This test pins the contract end-to-end.
    """
    payload = _run_doctor_json(monkeypatch, capsys)
    issues_url = payload["support"]["issues_url"]
    assert issues_url.startswith("https://")
    assert issues_url.rstrip("/").endswith("/issues")
