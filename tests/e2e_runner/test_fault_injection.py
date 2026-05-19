"""Tests for the mock-LLM fault-injection layer (task 27.3c).

The mock runner is the only one that can deliberately misbehave —
``qwen``/``opus`` either work or skip. Task 27.3c therefore extends the
:class:`MockLLMServer` fixture schema with three optional knobs and
ships a dedicated fault-injection scenario module that *expects* the
upstream to break:

* ``status_code`` — non-2xx response (defaults to 200). Drives the
  "HTTP-5xx" case.
* ``raw_body`` — emit this byte string verbatim instead of JSON-
  encoding ``response``. Drives the "kaputtes JSON" case.
* ``delay_s`` — sleep before responding. Drives the "Timeout" case.

An empty ``response`` body (or ``content=[]`` / ``tool_calls=[]``)
covers the "leere Tool-Call-Response" case; no new server knob is
needed for that one, only a dedicated fixture.

These tests prove the new knobs work in isolation; the higher-level
fault scenarios in :mod:`tests.e2e.scenarios.fault_injection.*`
exercise them end-to-end (covered separately by the matrix test
module).
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.mock_llm import MockLLMServer, load_fixtures
from tests.e2e.mock_llm.server import Fixture, FixtureSet

FAULT_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "e2e" / "mock_llm" / "fault_fixtures"
)


def _post_json(
    url: str, payload: dict, *, timeout: float = 5.0
) -> tuple[int, bytes, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - loopback only, tests
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        return exc.code, body_bytes, dict(exc.headers or {})


# ---------------------------------------------------------------------------
# status_code knob — HTTP-5xx fault
# ---------------------------------------------------------------------------


def test_fixture_status_code_default_is_200() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={},
        response={"content": []},
    )
    assert fixture.status_code == 200


def test_fixture_accepts_status_code(tmp_path: Path) -> None:
    (tmp_path / "boom.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"error": {"type": "server_error"}},
                "status_code": 503,
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    assert fixtures.fixtures[0].status_code == 503


def test_mock_server_emits_5xx_when_fixture_says_so(tmp_path: Path) -> None:
    (tmp_path / "boom.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"error": {"type": "server_error"}},
                "status_code": 503,
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    with MockLLMServer(fixtures=fixtures) as server:
        status, body, _ = _post_json(
            server.url + "/v1/messages",
            {"messages": [{"role": "user", "content": "anything"}]},
        )
    assert status == 503
    assert json.loads(body.decode("utf-8"))["error"]["type"] == "server_error"


# ---------------------------------------------------------------------------
# raw_body knob — malformed JSON
# ---------------------------------------------------------------------------


def test_fixture_raw_body_default_is_none() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={},
        response={"content": []},
    )
    assert fixture.raw_body is None


def test_fixture_accepts_raw_body(tmp_path: Path) -> None:
    (tmp_path / "garbage.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {},
                "raw_body": "{not-json-at-all",
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    assert fixtures.fixtures[0].raw_body == "{not-json-at-all"


def test_mock_server_emits_raw_body_verbatim(tmp_path: Path) -> None:
    (tmp_path / "garbage.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {},
                "status_code": 200,
                "raw_body": "{not-json-at-all",
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    with MockLLMServer(fixtures=fixtures) as server:
        # Hand-rolled socket because urllib raises on the malformed
        # body if it tries to follow ``Content-Length`` blindly.
        host, port = server.host, server.port
        body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(
            "utf-8"
        )
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(
                b"POST /v1/messages HTTP/1.1\r\n"
                b"Host: " + host.encode("ascii") + b"\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        raw = b"".join(chunks)
    assert b"{not-json-at-all" in raw


# ---------------------------------------------------------------------------
# delay_s knob — timeout
# ---------------------------------------------------------------------------


def test_fixture_delay_s_default_is_zero() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={},
        response={"content": []},
    )
    assert fixture.delay_s == 0.0


def test_fixture_accepts_delay_s(tmp_path: Path) -> None:
    (tmp_path / "slow.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"content": []},
                "delay_s": 0.05,
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    assert fixtures.fixtures[0].delay_s == 0.05


def test_mock_server_delays_response(tmp_path: Path) -> None:
    (tmp_path / "slow.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"content": []},
                "delay_s": 0.1,
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    with MockLLMServer(fixtures=fixtures) as server:
        started = time.monotonic()
        status, _, _ = _post_json(
            server.url + "/v1/messages",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        elapsed = time.monotonic() - started
    assert status == 200
    # Allow generous slack for slow Windows CI: the assertion is that
    # the delay actually happened, not that it landed exactly at 100ms.
    assert elapsed >= 0.08, elapsed


def test_mock_server_delay_can_be_timed_out(tmp_path: Path) -> None:
    (tmp_path / "slow.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"content": []},
                "delay_s": 0.5,
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    with (
        MockLLMServer(fixtures=fixtures) as server,
        pytest.raises((urllib.error.URLError, TimeoutError)),
    ):
        _post_json(
            server.url + "/v1/messages",
            {"messages": [{"role": "user", "content": "x"}]},
            timeout=0.1,
        )


# ---------------------------------------------------------------------------
# Empty tool-call response (no new server knob, just a fixture)
# ---------------------------------------------------------------------------


def test_empty_tool_call_response_returns_empty_content(tmp_path: Path) -> None:
    (tmp_path / "empty.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {
                    "id": "msg_empty",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": "end_turn",
                },
            }
        ),
        encoding="utf-8",
    )
    fixtures = load_fixtures(tmp_path)
    with MockLLMServer(fixtures=fixtures) as server:
        status, body, _ = _post_json(
            server.url + "/v1/messages",
            {"messages": [{"role": "user", "content": "x"}]},
        )
    assert status == 200
    parsed = json.loads(body.decode("utf-8"))
    assert parsed["content"] == []


# ---------------------------------------------------------------------------
# Ship-with fault fixtures
# ---------------------------------------------------------------------------


def test_fault_fixture_directory_exists() -> None:
    assert FAULT_FIXTURES_DIR.is_dir(), FAULT_FIXTURES_DIR


def test_fault_fixture_set_loads() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    names = {f.name for f in fixtures.fixtures}
    # One fixture per fault class spelled out in the task DoD.
    assert "broken_json" in names
    assert "timeout_slow" in names
    assert "empty_tool_calls" in names
    assert "http_5xx" in names


def test_fault_fixture_http_5xx_has_status_code() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    by_name = {f.name: f for f in fixtures.fixtures}
    assert by_name["http_5xx"].status_code >= 500


def test_fault_fixture_broken_json_has_raw_body() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    by_name = {f.name: f for f in fixtures.fixtures}
    assert by_name["broken_json"].raw_body is not None


def test_fault_fixture_timeout_has_delay() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    by_name = {f.name: f for f in fixtures.fixtures}
    assert by_name["timeout_slow"].delay_s > 0


def test_fault_fixture_empty_tool_calls_has_empty_response() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    by_name = {f.name: f for f in fixtures.fixtures}
    fixture = by_name["empty_tool_calls"]
    if fixture.endpoint == "/v1/messages":
        assert fixture.response.get("content") == []
    else:
        choice = fixture.response["choices"][0]
        assert choice["message"].get("tool_calls") in (None, [])


# ---------------------------------------------------------------------------
# Fixture-loading rejects bad knob types
# ---------------------------------------------------------------------------


def test_load_rejects_non_int_status_code(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {},
                "status_code": "five hundred",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status_code"):
        load_fixtures(tmp_path)


def test_load_rejects_non_string_raw_body(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {},
                "raw_body": ["not", "a", "string"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw_body"):
        load_fixtures(tmp_path)


def test_load_rejects_negative_delay(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {},
                "delay_s": -1.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="delay_s"):
        load_fixtures(tmp_path)


# ---------------------------------------------------------------------------
# FixtureSet helper used by the runner — passes through new knobs
# ---------------------------------------------------------------------------


def test_fixtureset_with_fault_fixtures_finds_match() -> None:
    fixtures = load_fixtures(FAULT_FIXTURES_DIR)
    # Each fixture has a ``match: user_contains: <keyword>`` predicate;
    # we use the fixture's own predicate to ensure the find pipeline
    # still works with the new knobs in place.
    by_name = {f.name: f for f in fixtures.fixtures}
    fixture = by_name["http_5xx"]
    if "user_contains" in fixture.match:
        keyword = fixture.match["user_contains"]
        found = fixtures.find(
            fixture.endpoint,
            {"messages": [{"role": "user", "content": keyword}]},
        )
        assert found is fixture
    else:
        # If the fixture matches all (empty match dict), find() should
        # still return *some* fixture for the right endpoint.
        found = fixtures.find(
            fixture.endpoint, {"messages": [{"role": "user", "content": "x"}]}
        )
        assert isinstance(found, Fixture)


def test_fixtureset_for_endpoint_classifies_by_endpoint() -> None:
    fs = FixtureSet(
        fixtures=[
            Fixture(name="a", endpoint="/v1/messages", match={}, response={}),
            Fixture(
                name="b",
                endpoint="/v1/chat/completions",
                match={},
                response={},
            ),
        ]
    )
    assert [f.name for f in fs.for_endpoint("/v1/messages")] == ["a"]
    assert [f.name for f in fs.for_endpoint("/v1/chat/completions")] == ["b"]
