"""Tests for the mock E2E endpoint runner (task 27.3c).

Unlike the Qwen / Opus runners, the mock runner does **not** depend on
any external endpoint: it spins the in-process :class:`MockLLMServer`
up on demand (or accepts an already-running one) and therefore always
runs the scenario. No skip path exists — the offline-CI guarantee is
that ``mock`` is always available.

The tests cover:

* exit-code constants and fragment shape parity with the other runners
  (so the harness in :mod:`tests.e2e.conftest` can parametrise across
  runners without provider-specific branches),
* the run wrapper invoking the scenario with the mock LLM URL exported
  through the documented env var,
* scenario failure / exception conversion to ``status="fail"``,
* CLI entry point that emits a JSON fragment on stdout (used by the
  bash + PowerShell ``run_all`` scripts that do not host a Python
  process),
* README + ``.env.example`` invariants so reviewers cannot drop the
  doc without breaking a test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.runners import mock as mock_runner

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_ROOT = REPO_ROOT / "tests" / "e2e"
ENV_EXAMPLE = E2E_ROOT / ".env.example"
README = E2E_ROOT / "README.md"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_exit_codes_match_other_runners() -> None:
    # The matrix harness compares exit codes to a single set of
    # constants; drifting between runners would break the report.
    assert mock_runner.EXIT_OK == 0
    assert mock_runner.EXIT_FAIL == 1
    assert mock_runner.EXIT_SKIPPED == 2


def test_env_var_name_constant() -> None:
    # Scenarios discover the mock URL through this env var so the same
    # code path works for the live runners too.
    assert mock_runner.ENV_URL_VAR == "CROSSMEM_E2E_MOCK_URL"


# ---------------------------------------------------------------------------
# Scenario fragment shape (parity with qwen/opus)
# ---------------------------------------------------------------------------


def test_pass_fragment_omits_reason() -> None:
    fragment = mock_runner.build_scenario_fragment(
        name="scenarios/happy_path/store_query.py",
        status="pass",
        duration_s=0.123,
        log_path="reports/x/y.log",
    )
    assert fragment["status"] == "pass"
    assert "reason" not in fragment
    assert fragment["duration_s"] == 0.123


def test_fail_fragment_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        mock_runner.build_scenario_fragment(
            name="x", status="fail", duration_s=0.0, log_path="y"
        )


def test_skipped_fragment_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        mock_runner.build_scenario_fragment(
            name="x", status="skipped", duration_s=0.0, log_path="y"
        )


def test_pass_rejects_reason() -> None:
    with pytest.raises(ValueError, match="must not carry a reason"):
        mock_runner.build_scenario_fragment(
            name="x",
            status="pass",
            duration_s=0.0,
            log_path="y",
            reason="should not be here",
        )


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        mock_runner.build_scenario_fragment(
            name="x",
            status="bogus",
            duration_s=0.0,
            log_path="y",
        )


# ---------------------------------------------------------------------------
# Run wrapper — happy path
# ---------------------------------------------------------------------------


def test_run_invokes_scenario_and_exports_url() -> None:
    # The scenario callable receives the mock URL via the documented
    # env var; the run wrapper must set it before invoking the
    # scenario and clear / restore the previous value afterwards.
    seen: dict[str, str] = {}

    def _scenario() -> int:
        import os as _os

        seen["url"] = _os.environ.get(mock_runner.ENV_URL_VAR, "")
        return 0

    exit_code, fragment = mock_runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
    )
    assert exit_code == mock_runner.EXIT_OK
    assert fragment["status"] == "pass"
    assert seen["url"].startswith("http://127.0.0.1:")


def test_run_restores_env_var_after_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(mock_runner.ENV_URL_VAR, "pre-existing")

    def _scenario() -> int:
        return 0

    mock_runner.run(
        scenario_name="x",
        scenario=_scenario,
        log_path="y",
    )
    import os as _os

    assert _os.environ.get(mock_runner.ENV_URL_VAR) == "pre-existing"


def test_run_unsets_env_var_when_unset_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(mock_runner.ENV_URL_VAR, raising=False)

    def _scenario() -> int:
        return 0

    mock_runner.run(
        scenario_name="x",
        scenario=_scenario,
        log_path="y",
    )
    import os as _os

    assert mock_runner.ENV_URL_VAR not in _os.environ


def test_run_accepts_existing_server() -> None:
    # When the caller supplies an already-running MockLLMServer the
    # wrapper must reuse it (and not start a second one on a fresh
    # port) so test authors can share one server across multiple
    # scenarios.
    from tests.e2e.mock_llm import MockLLMServer

    with MockLLMServer() as server:
        seen: dict[str, str] = {}

        def _scenario() -> int:
            import os as _os

            seen["url"] = _os.environ.get(mock_runner.ENV_URL_VAR, "")
            return 0

        exit_code, fragment = mock_runner.run(
            scenario_name="x",
            scenario=_scenario,
            log_path="y",
            server=server,
        )
        assert exit_code == mock_runner.EXIT_OK
        assert fragment["status"] == "pass"
        assert seen["url"] == server.url


# ---------------------------------------------------------------------------
# Run wrapper — failure paths
# ---------------------------------------------------------------------------


def test_run_propagates_scenario_failure() -> None:
    def _scenario() -> int:
        return 7

    exit_code, fragment = mock_runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
    )
    assert exit_code == mock_runner.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "exit code 7" in fragment["reason"]


def test_run_converts_exception_to_fail() -> None:
    def _scenario() -> int:
        raise RuntimeError("scenario blew up")

    exit_code, fragment = mock_runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
    )
    assert exit_code == mock_runner.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "RuntimeError" in fragment["reason"]
    assert "scenario blew up" in fragment["reason"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_main_check_only_emits_pass_fragment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The mock runner has no external endpoint to probe — ``--check-
    # only`` always emits a pass fragment because the mock server is
    # available in-process.
    rc = mock_runner.main(
        [
            "--check-only",
            "--scenario-name",
            "mock/check",
            "--log-path",
            "reports/x/y.log",
        ]
    )
    assert rc == mock_runner.EXIT_OK
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "mock/check"
    assert payload["status"] == "pass"
    assert payload["log_path"] == "reports/x/y.log"
    assert "reason" not in payload


def test_cli_check_only_exits_zero(tmp_path: Path) -> None:
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(REPO_ROOT / "src")
        + __import__("os").pathsep
        + str(REPO_ROOT),
    }
    proc = subprocess.run(  # noqa: S603 - controlled test invocation
        [
            sys.executable,
            "-m",
            "tests.e2e.runners.mock",
            "--check-only",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == mock_runner.EXIT_OK, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "pass"
    assert "reason" not in payload


# ---------------------------------------------------------------------------
# Documentation invariants
# ---------------------------------------------------------------------------


def test_readme_documents_mock_runner() -> None:
    text = README.read_text(encoding="utf-8")
    # Matrix overview + mock-runner section + skip behaviour summary
    assert "mock" in text.lower()
    # Fault-injection mention is part of the DoD.
    assert "fault" in text.lower() or "fault-injection" in text.lower()
    # Offline-CI guarantee must be spelled out.
    assert "offline" in text.lower()


def test_env_example_documents_mock_url() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert mock_runner.ENV_URL_VAR in text
