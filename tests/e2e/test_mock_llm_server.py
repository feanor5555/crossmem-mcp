"""Unit tests for the e2e mock LLM HTTP server (task 27.3).

Covers:

* fixture loading (skip ``*.request.json`` samples, reject unsupported
  endpoints / malformed shapes),
* request matching (``user_contains`` predicate + forward-compatible
  unknown keys),
* the live HTTP surface for both ``/v1/messages`` (Anthropic) and
  ``/v1/chat/completions`` (OpenAI), including error paths.

The server is started on an ephemeral loopback port per test so the
suite parallelises cleanly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.mock_llm import (
    DEFAULT_FIXTURES_DIR,
    FixtureSet,
    MockLLMServer,
    load_fixtures,
)
from tests.e2e.mock_llm.server import (
    SUPPORTED_ENDPOINTS,
    Fixture,
    _join_user_messages,
)

FIXTURES_DIR = DEFAULT_FIXTURES_DIR


def _post_json(url: str, payload: dict, *, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - loopback only, tests
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def _post_raw(url: str, body: bytes, *, timeout: float = 5.0) -> tuple[int, dict]:
    req = urllib.request.Request(  # noqa: S310 - loopback only, tests
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        text = body_bytes.decode("utf-8") if body_bytes else ""
        return exc.code, json.loads(text) if text else {}


def _load_request_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Fixture-loading layer
# --------------------------------------------------------------------------- #


def test_default_fixtures_dir_exists() -> None:
    assert DEFAULT_FIXTURES_DIR.is_dir(), DEFAULT_FIXTURES_DIR


def test_load_fixtures_picks_up_both_endpoints() -> None:
    fixtures = load_fixtures()
    endpoints = {f.endpoint for f in fixtures.fixtures}
    assert endpoints == set(SUPPORTED_ENDPOINTS)


def test_load_fixtures_skips_request_samples(tmp_path: Path) -> None:
    """``*.request.json`` files are caller-side request samples, not fixtures."""
    fixture_file = tmp_path / "only.json"
    fixture_file.write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": {"content": []},
            }
        ),
        encoding="utf-8",
    )
    sample_request = tmp_path / "only.request.json"
    sample_request.write_text(json.dumps({"messages": []}), encoding="utf-8")

    fixtures = load_fixtures(tmp_path)
    assert [f.name for f in fixtures.fixtures] == ["only"]


def test_load_fixtures_rejects_unsupported_endpoint(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v2/something",
                "match": {},
                "response": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported endpoint"):
        load_fixtures(tmp_path)


def test_load_fixtures_rejects_non_dict_match(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": ["not", "a", "dict"],
                "response": {"content": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'match' must be an object"):
        load_fixtures(tmp_path)


def test_load_fixtures_rejects_non_dict_response(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "endpoint": "/v1/messages",
                "match": {},
                "response": "not-a-dict",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'response' must be an object"):
        load_fixtures(tmp_path)


def test_load_fixtures_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fixtures(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------- #
# Request-matching layer
# --------------------------------------------------------------------------- #


def test_join_user_messages_handles_string_and_block_contents() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ignored"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "block-A"},
                    {"type": "image", "source": {"data": "..."}},
                    {"type": "text", "text": "block-B"},
                ],
            },
        ]
    }
    joined = _join_user_messages(body)
    assert "hello" in joined
    assert "block-A" in joined
    assert "block-B" in joined
    assert "ignored" not in joined


def test_join_user_messages_returns_empty_for_missing_messages() -> None:
    assert _join_user_messages({}) == ""
    assert _join_user_messages({"messages": "not-a-list"}) == ""


def test_join_user_messages_skips_non_dict_entries() -> None:
    body = {
        "messages": [
            "garbage-not-a-dict",
            {"role": "user", "content": "kept"},
        ]
    }
    assert _join_user_messages(body) == "kept"


def test_fixture_matches_user_contains_case_insensitive() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={"user_contains": "STORE"},
        response={"content": []},
    )
    assert fixture.matches(
        {"messages": [{"role": "user", "content": "please Store now"}]}
    )
    assert not fixture.matches(
        {"messages": [{"role": "user", "content": "query only"}]}
    )


def test_fixture_matches_unknown_match_key_is_forward_compatible() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={"future_key": "anything"},
        response={"content": []},
    )
    assert fixture.matches({"messages": []})


def test_fixture_matches_user_contains_non_string_returns_false() -> None:
    fixture = Fixture(
        name="f",
        endpoint="/v1/messages",
        match={"user_contains": 42},
        response={"content": []},
    )
    assert not fixture.matches({"messages": [{"role": "user", "content": "x"}]})


def test_fixtureset_find_returns_none_when_no_match() -> None:
    fixtures = FixtureSet(
        fixtures=[
            Fixture(
                name="store",
                endpoint="/v1/messages",
                match={"user_contains": "store"},
                response={"content": []},
            )
        ]
    )
    assert fixtures.find("/v1/messages", {"messages": []}) is None


# --------------------------------------------------------------------------- #
# Live HTTP surface — Anthropic
# --------------------------------------------------------------------------- #


def test_anthropic_messages_returns_tool_use_block() -> None:
    request = _load_request_fixture("anthropic_store.request.json")
    with MockLLMServer() as server:
        status, body = _post_json(server.url + "/v1/messages", request)

    assert status == 200
    assert body["type"] == "message"
    content = body["content"]
    assert isinstance(content, list) and content, body
    tool_use = content[0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "crossmem.store"
    assert tool_use["input"]["source_url"].startswith("https://example.invalid/")


# --------------------------------------------------------------------------- #
# Live HTTP surface — OpenAI
# --------------------------------------------------------------------------- #


def test_openai_chat_completions_returns_tool_calls() -> None:
    request = _load_request_fixture("openai_query.request.json")
    with MockLLMServer() as server:
        status, body = _post_json(server.url + "/v1/chat/completions", request)

    assert status == 200
    choices = body["choices"]
    assert isinstance(choices, list) and choices, body
    message = choices[0]["message"]
    tool_calls = message["tool_calls"]
    assert isinstance(tool_calls, list) and tool_calls, message
    call = tool_calls[0]
    assert call["function"]["name"] == "crossmem.query"
    args = json.loads(call["function"]["arguments"])
    assert args["query"] == "mock document"


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_unknown_endpoint_returns_404() -> None:
    with MockLLMServer() as server:
        status, body = _post_json(server.url + "/v1/unknown", {"messages": []})
    assert status == 404
    assert body["error"]["type"] == "not_found"


def test_endpoint_with_trailing_slash_is_normalised() -> None:
    request = _load_request_fixture("anthropic_store.request.json")
    with MockLLMServer() as server:
        status, body = _post_json(server.url + "/v1/messages/", request)
    assert status == 200
    assert body["content"][0]["type"] == "tool_use"


def test_invalid_json_body_returns_400() -> None:
    with MockLLMServer() as server:
        status, body = _post_raw(server.url + "/v1/messages", b"{not-json")
    assert status == 400
    assert body["error"]["type"] == "invalid_json"


def test_non_object_request_body_returns_400() -> None:
    with MockLLMServer() as server:
        status, body = _post_raw(
            server.url + "/v1/messages", b'["not", "an", "object"]'
        )
    assert status == 400
    assert body["error"]["type"] == "invalid_request"


def test_no_fixture_match_returns_404_with_diagnostic() -> None:
    with MockLLMServer() as server:
        status, body = _post_json(
            server.url + "/v1/messages",
            {"messages": [{"role": "user", "content": "no-keyword-here"}]},
        )
    assert status == 404
    assert body["error"]["type"] == "no_fixture_match"
    assert "/v1/messages" in body["error"]["message"]


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def test_port_and_url_unavailable_before_start() -> None:
    server = MockLLMServer()
    with pytest.raises(RuntimeError):
        _ = server.port
    with pytest.raises(RuntimeError):
        _ = server.url


def test_double_start_raises() -> None:
    server = MockLLMServer()
    server.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            server.start()
    finally:
        server.stop()


def test_stop_is_idempotent() -> None:
    server = MockLLMServer()
    server.start()
    server.stop()
    server.stop()  # second call must not raise


def test_can_pass_fixtureset_directly() -> None:
    fixtures = load_fixtures()
    server = MockLLMServer(fixtures=fixtures)
    server.start()
    try:
        assert server.fixtures is fixtures
        assert server.host == "127.0.0.1"
        assert server.port > 0
    finally:
        server.stop()


def test_supported_endpoints_constant_matches_handler() -> None:
    assert set(SUPPORTED_ENDPOINTS) == {"/v1/messages", "/v1/chat/completions"}


# --------------------------------------------------------------------------- #
# Packaging: the ``e2e`` extra must declare this code path
# --------------------------------------------------------------------------- #


def test_pyproject_declares_e2e_optional_dependency_group() -> None:
    """``pyproject.toml`` must expose an ``e2e`` extra (may be empty).

    The task DoD explicitly says the mock server is a dev dependency in
    ``[project.optional-dependencies].e2e``; downstream tasks (27.4+)
    will append CLI-specific helpers to the same group.
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - py3.10 fallback
        import tomli as tomllib

    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]
    assert "e2e" in optional, optional
    # Must be a list (possibly empty) so future tasks can append to it.
    assert isinstance(optional["e2e"], list)
