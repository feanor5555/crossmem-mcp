"""Tests for the Qwen E2E endpoint runner (task 27.3a).

The runner targets a *real* Qwen backend exposed over an OpenAI-
compatible REST endpoint (default ``http://192.168.178.45:8080``, model
``qwen-3.5``). CI hosts almost never have that endpoint reachable, so
the central behaviour under test is the **skip path**: when
``GET /v1/models`` does not answer within the connection-check budget,
the runner must exit ``2`` and produce a scenario-report fragment with
``status: "skipped"`` so the wider E2E harness keeps going.

The tests cover:

* default URL / env-var override / model constant,
* happy-path connection check (mocked ``/v1/models`` -> 200),
* skip path (connection error, timeout, non-2xx response),
* scenario-report fragment schema and JSON-CLI exit codes,
* documentation and ``.env.example`` invariants enforced statically
  (so reviewers cannot drop the doc without breaking a test).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tests.e2e.runners import qwen

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_ROOT = REPO_ROOT / "tests" / "e2e"
ENV_EXAMPLE = E2E_ROOT / ".env.example"
README = E2E_ROOT / "README.md"


# ---------------------------------------------------------------------------
# Constants and config
# ---------------------------------------------------------------------------


def test_default_url_constant() -> None:
    assert qwen.DEFAULT_URL == "http://192.168.178.45:8080"


def test_default_model_constant() -> None:
    assert qwen.DEFAULT_MODEL == "qwen-3.5"


def test_env_var_name_constant() -> None:
    assert qwen.ENV_URL_VAR == "CROSSMEM_E2E_QWEN_URL"


def test_exit_codes() -> None:
    # Conventional UNIX-style: 0 success, 1 fail, 2 skipped (per DoD).
    assert qwen.EXIT_OK == 0
    assert qwen.EXIT_FAIL == 1
    assert qwen.EXIT_SKIPPED == 2


def test_resolve_url_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(qwen.ENV_URL_VAR, raising=False)
    assert qwen.resolve_url() == qwen.DEFAULT_URL


def test_resolve_url_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    assert qwen.resolve_url() == "http://qwen.example:9000"


def test_resolve_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``/v1/models`` is appended verbatim, so a trailing slash would
    # produce a double-slash request path.
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000/")
    assert qwen.resolve_url() == "http://qwen.example:9000"


def test_resolve_url_blank_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``.env.example`` ships the key with an empty value. If a user
    # sources that file unchanged the runner should still target the
    # documented default rather than POSTing to "".
    monkeypatch.setenv(qwen.ENV_URL_VAR, "")
    assert qwen.resolve_url() == qwen.DEFAULT_URL


# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------


def test_check_connection_ok(httpx_mock) -> None:
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        json={"data": [{"id": "qwen-3.5"}]},
        status_code=200,
    )
    result = qwen.check_connection("http://qwen.example:9000", timeout=0.5)
    assert result.reachable is True
    assert result.status_code == 200
    assert result.reason is None


def test_check_connection_non_2xx_marks_unreachable(httpx_mock) -> None:
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        status_code=503,
        json={"error": "starting up"},
    )
    result = qwen.check_connection("http://qwen.example:9000", timeout=0.5)
    assert result.reachable is False
    assert result.status_code == 503
    assert result.reason is not None
    assert "503" in result.reason


def test_check_connection_connect_error_skips(httpx_mock) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("connection refused"),
        url="http://qwen.example:9000/v1/models",
    )
    result = qwen.check_connection("http://qwen.example:9000", timeout=0.5)
    assert result.reachable is False
    assert result.status_code is None
    assert "ConnectError" in (result.reason or "") or "refused" in (result.reason or "")


def test_check_connection_timeout_skips(httpx_mock) -> None:
    httpx_mock.add_exception(
        httpx.ConnectTimeout("timed out"),
        url="http://qwen.example:9000/v1/models",
    )
    result = qwen.check_connection("http://qwen.example:9000", timeout=0.5)
    assert result.reachable is False
    assert result.status_code is None


# ---------------------------------------------------------------------------
# Scenario fragment + run wrapper
# ---------------------------------------------------------------------------


def test_skipped_fragment_shape() -> None:
    fragment = qwen.build_scenario_fragment(
        name="scenarios/happy_path/store-query.py",
        status="skipped",
        duration_s=0.0,
        log_path="reports/20260512T173045Z/store-query.log",
        reason="endpoint unreachable: ConnectError",
    )
    # Schema matches task 27.1 report ({name,status,duration_s,log_path})
    # plus an optional ``reason`` field used by skip / fail entries.
    assert fragment["name"] == "scenarios/happy_path/store-query.py"
    assert fragment["status"] == "skipped"
    assert fragment["duration_s"] == 0.0
    assert fragment["log_path"].endswith("store-query.log")
    assert fragment["reason"].startswith("endpoint unreachable")


def test_status_pass_omits_reason() -> None:
    fragment = qwen.build_scenario_fragment(
        name="scenarios/happy_path/store-query.py",
        status="pass",
        duration_s=0.123,
        log_path="reports/x/y.log",
    )
    assert "reason" not in fragment


def test_status_fail_includes_reason() -> None:
    fragment = qwen.build_scenario_fragment(
        name="scenarios/happy_path/store-query.py",
        status="fail",
        duration_s=0.5,
        log_path="reports/x/y.log",
        reason="scenario exited 7",
    )
    assert fragment["status"] == "fail"
    assert fragment["reason"] == "scenario exited 7"


def test_build_fragment_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        qwen.build_scenario_fragment(
            name="x",
            status="bogus",
            duration_s=0.0,
            log_path="y",
        )


def test_build_fragment_skipped_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        qwen.build_scenario_fragment(
            name="x", status="skipped", duration_s=0.0, log_path="y"
        )


def test_build_fragment_fail_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        qwen.build_scenario_fragment(
            name="x", status="fail", duration_s=0.0, log_path="y"
        )


def test_build_fragment_pass_rejects_reason() -> None:
    with pytest.raises(ValueError, match="must not carry a reason"):
        qwen.build_scenario_fragment(
            name="x",
            status="pass",
            duration_s=0.0,
            log_path="y",
            reason="should not be here",
        )


def test_run_with_skip_uses_exit_2(monkeypatch: pytest.MonkeyPatch, httpx_mock) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_exception(
        httpx.ConnectError("nope"),
        url="http://qwen.example:9000/v1/models",
    )

    def _scenario() -> int:  # pragma: no cover - not invoked on skip
        raise AssertionError("scenario must not run when endpoint is down")

    exit_code, fragment = qwen.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == qwen.EXIT_SKIPPED
    assert fragment["status"] == "skipped"
    assert fragment["reason"].startswith("endpoint unreachable")


def test_run_happy_path_invokes_scenario(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        json={"data": [{"id": "qwen-3.5"}]},
        status_code=200,
    )
    calls: list[str] = []

    def _scenario() -> int:
        calls.append("ran")
        return 0

    exit_code, fragment = qwen.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == qwen.EXIT_OK
    assert fragment["status"] == "pass"
    assert calls == ["ran"]


def test_run_converts_scenario_exception_to_fail(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        json={"data": []},
        status_code=200,
    )

    def _scenario() -> int:
        raise RuntimeError("network blew up mid-scenario")

    exit_code, fragment = qwen.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == qwen.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "RuntimeError" in fragment["reason"]
    assert "network blew up" in fragment["reason"]


def test_run_explicit_base_url_overrides_env(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    # An explicit ``base_url=`` argument wins over the env var so
    # tests / orchestration code can target an arbitrary endpoint
    # without mutating ``os.environ``.
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://wrong.example:1")
    httpx_mock.add_response(
        url="http://right.example:9000/v1/models",
        method="GET",
        json={"data": []},
        status_code=200,
    )

    def _scenario() -> int:
        return 0

    exit_code, fragment = qwen.run(
        scenario_name="x",
        scenario=_scenario,
        log_path="y",
        timeout=0.5,
        base_url="http://right.example:9000",
    )
    assert exit_code == qwen.EXIT_OK
    assert fragment["status"] == "pass"


def test_run_propagates_scenario_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        json={"data": []},
        status_code=200,
    )

    def _scenario() -> int:
        return 7

    exit_code, fragment = qwen.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == qwen.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "exit code 7" in fragment["reason"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_main_reachable_emits_pass_fragment(
    monkeypatch: pytest.MonkeyPatch, httpx_mock, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_response(
        url="http://qwen.example:9000/v1/models",
        method="GET",
        json={"data": [{"id": "qwen-3.5"}]},
        status_code=200,
    )
    rc = qwen.main(
        [
            "--check-only",
            "--scenario-name",
            "qwen/check",
            "--log-path",
            "reports/x/y.log",
            "--timeout",
            "0.5",
        ]
    )
    assert rc == qwen.EXIT_OK
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "qwen/check"
    assert payload["status"] == "pass"
    assert payload["log_path"] == "reports/x/y.log"


def test_main_unreachable_emits_skipped_fragment(
    monkeypatch: pytest.MonkeyPatch, httpx_mock, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(qwen.ENV_URL_VAR, "http://qwen.example:9000")
    httpx_mock.add_exception(
        httpx.ConnectError("nope"),
        url="http://qwen.example:9000/v1/models",
    )
    rc = qwen.main(["--check-only", "--timeout", "0.5"])
    assert rc == qwen.EXIT_SKIPPED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "skipped"
    assert payload["reason"].startswith("endpoint unreachable")


def test_cli_check_only_skipped_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Run the runner as a subprocess so the actual exit code is observed,
    # not just the return value of ``main``. The env var points at a
    # closed loopback port so the socket call returns immediately.
    env = {
        **dict(__import__("os").environ),
        qwen.ENV_URL_VAR: "http://127.0.0.1:1",
        "PYTHONPATH": str(REPO_ROOT / "src")
        + __import__("os").pathsep
        + str(REPO_ROOT),
    }
    proc = subprocess.run(  # noqa: S603 - controlled test invocation
        [
            sys.executable,
            "-m",
            "tests.e2e.runners.qwen",
            "--check-only",
            "--timeout",
            "0.5",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == qwen.EXIT_SKIPPED, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "skipped"
    assert payload["reason"]


# ---------------------------------------------------------------------------
# Documentation and .env.example invariants
# ---------------------------------------------------------------------------


def test_env_example_lists_qwen_url_without_default() -> None:
    assert ENV_EXAMPLE.is_file(), (
        f"{ENV_EXAMPLE} missing — task 27.3a DoD: .env.example must list "
        f"the Qwen env var without a default value"
    )
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    # ``CROSSMEM_E2E_QWEN_URL=`` on its own line, no default value.
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith(f"{qwen.ENV_URL_VAR}="):
            value = stripped.split("=", 1)[1]
            assert value == "", (
                f".env.example must not pre-fill {qwen.ENV_URL_VAR}; "
                f"users should opt in by setting their own URL"
            )
            found = True
    assert found, f".env.example does not list {qwen.ENV_URL_VAR}"


def test_readme_documents_qwen_runner() -> None:
    text = README.read_text(encoding="utf-8")
    assert qwen.ENV_URL_VAR in text, "README must mention the Qwen env var"
    assert qwen.DEFAULT_URL in text, "README must mention the default Qwen URL"
    assert "qwen-3.5" in text, "README must mention the Qwen model name"
    # Skip behaviour must be documented so users know why CI passes
    # without a reachable backend.
    assert "skip" in text.lower()
