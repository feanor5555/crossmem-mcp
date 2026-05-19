"""Tests for the Opus E2E endpoint runner (task 27.3b).

The runner targets the official Anthropic Messages API to validate that
CrossMem scenarios can drive Claude Opus end to end. CI hosts virtually
never have ``ANTHROPIC_API_KEY`` set, so the central behaviour under
test is the **skip path**: when the key is missing (or the probe call
fails) the runner must exit ``2`` and emit a scenario-report fragment
with ``status: "skipped"`` so the wider harness keeps going.

The tests cover:

* env-var name and default model / endpoint constants,
* skip when the API key is unset or blank,
* probe-call happy path (mocked ``POST /v1/messages`` -> 200),
* probe-call failure paths (auth error, network error, non-2xx),
* scenario-report fragment schema and CLI exit codes,
* documentation and ``.env.example`` invariants enforced statically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tests.e2e.runners import opus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
E2E_ROOT = REPO_ROOT / "tests" / "e2e"
ENV_EXAMPLE = E2E_ROOT / ".env.example"
README = E2E_ROOT / "README.md"


# ---------------------------------------------------------------------------
# Constants and config
# ---------------------------------------------------------------------------


def test_env_var_name_constant() -> None:
    assert opus.ENV_API_KEY_VAR == "ANTHROPIC_API_KEY"


def test_default_base_url_constant() -> None:
    # Official Anthropic Messages API endpoint.
    assert opus.DEFAULT_BASE_URL == "https://api.anthropic.com"


def test_default_model_constant() -> None:
    # Claude Opus identifier as published by Anthropic. The exact tag
    # is allowed to drift; the test pins the family so a typo in the
    # constant is caught.
    assert opus.DEFAULT_MODEL.startswith("claude-opus")


def test_api_version_constant() -> None:
    # Anthropic requires an ``anthropic-version`` header on every
    # Messages call. Pinning the constant prevents silent drift.
    assert opus.ANTHROPIC_API_VERSION == "2023-06-01"


def test_exit_codes() -> None:
    assert opus.EXIT_OK == 0
    assert opus.EXIT_FAIL == 1
    assert opus.EXIT_SKIPPED == 2


def test_resolve_api_key_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(opus.ENV_API_KEY_VAR, raising=False)
    assert opus.resolve_api_key() is None


def test_resolve_api_key_returns_none_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``.env.example`` ships the key with an empty value, so sourcing
    # that file unchanged must NOT be treated as "key present".
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "")
    assert opus.resolve_api_key() is None


def test_resolve_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-test-123")
    assert opus.resolve_api_key() == "sk-ant-test-123"


def test_resolve_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "  sk-ant-test-123  ")
    assert opus.resolve_api_key() == "sk-ant-test-123"


# ---------------------------------------------------------------------------
# Probe call
# ---------------------------------------------------------------------------


def test_check_api_ok(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )
    result = opus.check_api(api_key="sk-ant-test", timeout=0.5)
    assert result.reachable is True
    assert result.status_code == 200
    assert result.reason is None


def test_check_api_auth_failure(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        status_code=401,
        json={
            "error": {"type": "authentication_error", "message": "invalid x-api-key"}
        },
    )
    result = opus.check_api(api_key="sk-ant-bad", timeout=0.5)
    assert result.reachable is False
    assert result.status_code == 401
    assert result.reason is not None
    assert "401" in result.reason


def test_check_api_rate_limited(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        status_code=429,
        json={"error": {"type": "rate_limit_error"}},
    )
    result = opus.check_api(api_key="sk-ant-test", timeout=0.5)
    assert result.reachable is False
    assert result.status_code == 429


def test_check_api_connect_error(httpx_mock) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("dns refused"),
        url="https://api.anthropic.com/v1/messages",
    )
    result = opus.check_api(api_key="sk-ant-test", timeout=0.5)
    assert result.reachable is False
    assert result.status_code is None
    assert "ConnectError" in (result.reason or "") or "refused" in (result.reason or "")


def test_check_api_timeout(httpx_mock) -> None:
    httpx_mock.add_exception(
        httpx.ConnectTimeout("timed out"),
        url="https://api.anthropic.com/v1/messages",
    )
    result = opus.check_api(api_key="sk-ant-test", timeout=0.5)
    assert result.reachable is False
    assert result.status_code is None


def test_check_api_sends_required_headers(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )
    opus.check_api(api_key="sk-ant-test-123", timeout=0.5)
    # pytest-httpx records the request; verify the auth + version
    # headers the Anthropic Messages API mandates.
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    assert req.headers.get("x-api-key") == "sk-ant-test-123"
    assert req.headers.get("anthropic-version") == opus.ANTHROPIC_API_VERSION
    assert req.headers.get("content-type", "").startswith("application/json")


# ---------------------------------------------------------------------------
# Scenario fragment + run wrapper
# ---------------------------------------------------------------------------


def test_skipped_fragment_shape() -> None:
    fragment = opus.build_scenario_fragment(
        name="scenarios/happy_path/store-query.py",
        status="skipped",
        duration_s=0.0,
        log_path="reports/20260512T173045Z/store-query.log",
        reason="ANTHROPIC_API_KEY not set",
    )
    assert fragment["name"] == "scenarios/happy_path/store-query.py"
    assert fragment["status"] == "skipped"
    assert fragment["duration_s"] == 0.0
    assert fragment["log_path"].endswith("store-query.log")
    assert fragment["reason"] == "ANTHROPIC_API_KEY not set"


def test_status_pass_omits_reason() -> None:
    fragment = opus.build_scenario_fragment(
        name="scenarios/x",
        status="pass",
        duration_s=0.123,
        log_path="reports/x/y.log",
    )
    assert "reason" not in fragment


def test_status_fail_includes_reason() -> None:
    fragment = opus.build_scenario_fragment(
        name="scenarios/x",
        status="fail",
        duration_s=0.5,
        log_path="reports/x/y.log",
        reason="scenario exited 7",
    )
    assert fragment["status"] == "fail"
    assert fragment["reason"] == "scenario exited 7"


def test_build_fragment_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        opus.build_scenario_fragment(
            name="x", status="bogus", duration_s=0.0, log_path="y"
        )


def test_build_fragment_skipped_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        opus.build_scenario_fragment(
            name="x", status="skipped", duration_s=0.0, log_path="y"
        )


def test_build_fragment_fail_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        opus.build_scenario_fragment(
            name="x", status="fail", duration_s=0.0, log_path="y"
        )


def test_build_fragment_pass_rejects_reason() -> None:
    with pytest.raises(ValueError, match="must not carry a reason"):
        opus.build_scenario_fragment(
            name="x",
            status="pass",
            duration_s=0.0,
            log_path="y",
            reason="should not be here",
        )


def test_run_without_api_key_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(opus.ENV_API_KEY_VAR, raising=False)

    def _scenario() -> int:  # pragma: no cover - must not be invoked
        raise AssertionError("scenario must not run when API key is missing")

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_SKIPPED
    assert fragment["status"] == "skipped"
    assert "ANTHROPIC_API_KEY" in fragment["reason"]


def test_run_with_blank_api_key_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "   ")

    def _scenario() -> int:  # pragma: no cover - must not be invoked
        raise AssertionError("scenario must not run when API key is blank")

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_SKIPPED
    assert fragment["status"] == "skipped"


def test_run_skips_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-bad")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        status_code=401,
        json={"error": {"type": "authentication_error"}},
    )

    def _scenario() -> int:  # pragma: no cover - must not be invoked
        raise AssertionError("scenario must not run when probe fails")

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_SKIPPED
    assert fragment["status"] == "skipped"
    assert "401" in fragment["reason"]


def test_run_happy_path_invokes_scenario(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-test")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )
    calls: list[str] = []

    def _scenario() -> int:
        calls.append("ran")
        return 0

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_OK
    assert fragment["status"] == "pass"
    assert calls == ["ran"]


def test_run_converts_scenario_exception_to_fail(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-test")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )

    def _scenario() -> int:
        raise RuntimeError("network blew up mid-scenario")

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "RuntimeError" in fragment["reason"]
    assert "network blew up" in fragment["reason"]


def test_run_propagates_scenario_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-test")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )

    def _scenario() -> int:
        return 7

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=_scenario,
        log_path="reports/x/y.log",
        timeout=0.5,
    )
    assert exit_code == opus.EXIT_FAIL
    assert fragment["status"] == "fail"
    assert "exit code 7" in fragment["reason"]


def test_run_explicit_api_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
) -> None:
    # An explicit ``api_key=`` wins over the env var so orchestration
    # code can target an arbitrary key without mutating os.environ.
    monkeypatch.delenv(opus.ENV_API_KEY_VAR, raising=False)
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )

    def _scenario() -> int:
        return 0

    exit_code, fragment = opus.run(
        scenario_name="x",
        scenario=_scenario,
        log_path="y",
        timeout=0.5,
        api_key="sk-ant-explicit",
    )
    assert exit_code == opus.EXIT_OK
    assert fragment["status"] == "pass"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_main_unset_key_emits_skipped_fragment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(opus.ENV_API_KEY_VAR, raising=False)
    rc = opus.main(["--check-only", "--timeout", "0.5"])
    assert rc == opus.EXIT_SKIPPED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "skipped"
    assert "ANTHROPIC_API_KEY" in payload["reason"]


def test_main_reachable_emits_pass_fragment(
    monkeypatch: pytest.MonkeyPatch, httpx_mock, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-test")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": opus.DEFAULT_MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        status_code=200,
    )
    rc = opus.main(
        [
            "--check-only",
            "--scenario-name",
            "opus/check",
            "--log-path",
            "reports/x/y.log",
            "--timeout",
            "0.5",
        ]
    )
    assert rc == opus.EXIT_OK
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "opus/check"
    assert payload["status"] == "pass"
    assert payload["log_path"] == "reports/x/y.log"


def test_main_auth_failure_emits_skipped_fragment(
    monkeypatch: pytest.MonkeyPatch, httpx_mock, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(opus.ENV_API_KEY_VAR, "sk-ant-bad")
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        status_code=401,
        json={"error": {"type": "authentication_error"}},
    )
    rc = opus.main(["--check-only", "--timeout", "0.5"])
    assert rc == opus.EXIT_SKIPPED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "skipped"
    assert "401" in payload["reason"]


def test_cli_check_only_skipped_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Run the runner as a subprocess so the real exit code is observed.
    import os

    env = {k: v for k, v in os.environ.items() if k != opus.ENV_API_KEY_VAR}
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)
    proc = subprocess.run(  # noqa: S603 - controlled test invocation
        [
            sys.executable,
            "-m",
            "tests.e2e.runners.opus",
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
    assert proc.returncode == opus.EXIT_SKIPPED, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "skipped"
    assert payload["reason"]


# ---------------------------------------------------------------------------
# Documentation and .env.example invariants
# ---------------------------------------------------------------------------


def test_env_example_lists_anthropic_api_key_without_default() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith(f"{opus.ENV_API_KEY_VAR}="):
            value = stripped.split("=", 1)[1]
            assert value == "", (
                f".env.example must not pre-fill {opus.ENV_API_KEY_VAR}; "
                f"users opt in by setting their own key"
            )
            found = True
    assert found, f".env.example does not list {opus.ENV_API_KEY_VAR}"


def test_readme_documents_opus_runner() -> None:
    text = README.read_text(encoding="utf-8")
    assert opus.ENV_API_KEY_VAR in text, "README must mention the Anthropic API key var"
    assert "Opus" in text or "opus" in text, "README must mention the Opus runner"
    assert "skip" in text.lower(), "README must document skip behaviour"
