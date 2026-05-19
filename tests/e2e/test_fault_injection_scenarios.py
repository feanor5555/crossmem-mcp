"""Fault-injection scenarios for the mock runner (task 27.3c).

These scenarios are mock-only on purpose: the qwen / opus runners
target live endpoints we cannot deliberately break. Each scenario
asserts that the CrossMem client / scenario harness fails *gracefully*
when the upstream LLM behaves badly — broken JSON, HTTP-5xx, timeout,
empty tool-call response.

The scenarios are exercised here as direct callables so failure modes
are caught by ``pytest`` even when the wider matrix is not run. The
mock runner re-uses the same callables via its own ``run`` wrapper.
"""

from __future__ import annotations

import os

from tests.e2e.mock_llm import MockLLMServer, load_fixtures
from tests.e2e.runners import mock as mock_runner
from tests.e2e.scenarios.fault_injection import (
    broken_json,
    empty_tool_calls,
    http_5xx,
    timeout_slow,
)
from tests.e2e.scenarios.happy_path import store_query

FAULT_FIXTURES_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent
    / "mock_llm"
    / "fault_fixtures"
)


def _server_for(name: str) -> MockLLMServer:
    # Bind a single fixture into the server so the scenario can talk
    # to its own faulty endpoint without colliding with the others.
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    matching = [f for f in fixtures.fixtures if f.name == name]
    if not matching:
        raise LookupError(f"fault fixture {name!r} missing")
    from tests.e2e.mock_llm.server import FixtureSet

    return MockLLMServer(fixtures=FixtureSet(fixtures=matching))


# ---------------------------------------------------------------------------
# Direct scenario invocations (unit-level coverage of the assertions)
# ---------------------------------------------------------------------------


def test_broken_json_scenario_returns_zero_on_expected_break() -> None:
    with _server_for("broken_json") as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = broken_json.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc == 0


def test_http_5xx_scenario_returns_zero_on_expected_break() -> None:
    with _server_for("http_5xx") as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = http_5xx.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc == 0


def test_timeout_scenario_returns_zero_on_expected_break() -> None:
    with _server_for("timeout_slow") as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = timeout_slow.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc == 0


def test_empty_tool_calls_scenario_returns_zero_on_expected_break() -> None:
    with _server_for("empty_tool_calls") as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = empty_tool_calls.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc == 0


# ---------------------------------------------------------------------------
# Each fault scenario must fail loudly when the upstream is *not* broken
# (otherwise a regression that silently fixed the fault would go unnoticed).
# ---------------------------------------------------------------------------


def test_broken_json_scenario_fails_when_response_is_well_formed() -> None:
    # Run the scenario against the *happy-path* mock fixtures — the
    # response is valid JSON so the "expect broken JSON" assertion
    # must trip.
    with MockLLMServer() as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = broken_json.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc != 0


def test_http_5xx_scenario_fails_when_response_is_2xx() -> None:
    with MockLLMServer() as server:
        os.environ[mock_runner.ENV_URL_VAR] = server.url
        try:
            rc = http_5xx.run()
        finally:
            del os.environ[mock_runner.ENV_URL_VAR]
    assert rc != 0


# ---------------------------------------------------------------------------
# Mock-runner integration: scenarios -> mock runner -> fragment
# ---------------------------------------------------------------------------


def test_happy_path_scenario_passes_via_mock_runner() -> None:
    exit_code, fragment = mock_runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=store_query.run,
        log_path="reports/happy/store_query.log",
    )
    assert exit_code == mock_runner.EXIT_OK
    assert fragment["status"] == "pass"


def test_broken_json_scenario_via_mock_runner_with_server() -> None:
    with _server_for("broken_json") as server:
        exit_code, fragment = mock_runner.run(
            scenario_name="scenarios/fault_injection/broken_json.py",
            scenario=broken_json.run,
            log_path="reports/faults/broken_json.log",
            server=server,
        )
    assert exit_code == mock_runner.EXIT_OK
    assert fragment["status"] == "pass"
