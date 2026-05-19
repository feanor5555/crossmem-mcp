"""Threaded stdlib HTTP server that emulates Anthropic + OpenAI chat APIs.

Design notes
------------

* **Stdlib only.** The ``e2e`` extra in ``pyproject.toml`` must not pull
  in a web framework. We rely on :mod:`http.server` + a worker thread so
  the e2e harness can spin the mock up in-process or as a sidecar.
* **Fixture-driven.** Every response is loaded from a JSON file under
  ``tests/e2e/mock_llm/fixtures/``. Each fixture declares the
  ``endpoint`` it serves and a ``match`` block. The first fixture whose
  match predicate is satisfied wins; if none match, the server returns
  HTTP 404 with a JSON error so test failures stay loud.
* **Two endpoints.**

  * ``POST /v1/messages`` — Anthropic-style. Response body is the
    fixture's ``response`` JSON (must contain ``content`` blocks with at
    least one ``{"type": "tool_use", ...}`` entry, per the Anthropic API
    contract; the mock does not enforce that, the caller's fixtures do).
  * ``POST /v1/chat/completions`` — OpenAI-style. Response body is the
    fixture's ``response`` JSON (typically
    ``choices[0].message.tool_calls``).

Matching
--------

The match predicate currently supports a single key,
``user_contains``: a case-insensitive substring check against the
concatenation of every ``user``-role message in the incoming request.
That is enough to map "please store ..." -> the ``crossmem.store``
fixture and "please query ..." -> the ``crossmem.query`` fixture without
forcing test authors to hand-craft exact-equality request payloads.

Extra match keys can be added later (e.g. ``model_equals``,
``tool_name``) without breaking existing fixtures because unknown keys
in the fixture's ``match`` block fall through to a no-op match (always
True) — keeping forward compatibility for future fixture authors.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SUPPORTED_ENDPOINTS = frozenset({"/v1/messages", "/v1/chat/completions"})


@dataclass(frozen=True)
class Fixture:
    """A single request -> response mapping loaded from disk.

    Beyond the mandatory ``endpoint`` / ``match`` / ``response`` triple
    the fixture carries three optional fault-injection knobs (task
    27.3c):

    * ``status_code`` — non-2xx response body. Drives the "HTTP-5xx"
      fault case in :mod:`tests.e2e.scenarios.fault_injection.http_5xx`.
    * ``raw_body`` — emit this byte / string body verbatim instead of
      JSON-encoding ``response``. Drives the "broken JSON" case
      (:mod:`tests.e2e.scenarios.fault_injection.broken_json`).
    * ``delay_s`` — sleep before responding. Drives the "timeout" case
      (:mod:`tests.e2e.scenarios.fault_injection.timeout_slow`).

    Fault fixtures live in ``tests/e2e/mock_llm/fault_fixtures/`` —
    deliberately *outside* :data:`DEFAULT_FIXTURES_DIR` so that a
    plain ``MockLLMServer()`` (used by the happy-path scenarios) never
    accidentally loads them.

    The defaults mean any fixture that doesn't opt in to a fault knob
    behaves exactly like the pre-27.3c fixtures (200 OK, JSON body,
    no delay) — so existing fixtures keep working unchanged.
    """

    name: str
    endpoint: str
    match: dict[str, Any]
    response: dict[str, Any]
    status_code: int = 200
    raw_body: str | None = None
    delay_s: float = 0.0

    def matches(self, request_body: dict[str, Any]) -> bool:
        """Return ``True`` iff *request_body* satisfies ``self.match``.

        Unknown match keys are ignored (treated as satisfied) so older
        fixtures keep working when new predicate keys are added.
        """
        for key, expected in self.match.items():
            if key == "user_contains":
                joined = _join_user_messages(request_body).lower()
                if not isinstance(expected, str):
                    return False
                if expected.lower() not in joined:
                    return False
            # Unknown keys: forward-compatibility, treat as match.
        return True


@dataclass
class FixtureSet:
    """All fixtures grouped by endpoint."""

    fixtures: list[Fixture]

    def for_endpoint(self, endpoint: str) -> list[Fixture]:
        return [f for f in self.fixtures if f.endpoint == endpoint]

    def find(self, endpoint: str, request_body: dict[str, Any]) -> Fixture | None:
        for fixture in self.for_endpoint(endpoint):
            if fixture.matches(request_body):
                return fixture
        return None


def _join_user_messages(request_body: dict[str, Any]) -> str:
    """Concatenate all ``user``-role message contents into a flat string.

    Both Anthropic and OpenAI nest user prompts under ``messages``. The
    Anthropic ``content`` field may be a string or a list of content
    blocks; OpenAI's is always a string. This helper papers over both
    shapes so the ``user_contains`` matcher works for either endpoint.
    """
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def load_fixtures(directory: Path | None = None) -> FixtureSet:
    """Load every ``*.json`` fixture in *directory* (recursively).

    Files whose name ends in ``.request.json`` are sample REQUESTS used
    by callers (e.g. the curl smoke-checks in the task DoD) and are
    skipped here.
    """
    target = directory or DEFAULT_FIXTURES_DIR
    fixtures: list[Fixture] = []
    if not target.is_dir():
        raise FileNotFoundError(f"fixtures directory not found: {target}")
    for path in sorted(target.rglob("*.json")):
        if path.name.endswith(".request.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        endpoint = data.get("endpoint")
        if endpoint not in SUPPORTED_ENDPOINTS:
            raise ValueError(
                f"fixture {path.name!r} declares unsupported endpoint "
                f"{endpoint!r}; expected one of {sorted(SUPPORTED_ENDPOINTS)}"
            )
        match = data.get("match") or {}
        if not isinstance(match, dict):
            raise ValueError(
                f"fixture {path.name!r}: 'match' must be an object, got "
                f"{type(match).__name__}"
            )
        response = data.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"fixture {path.name!r}: 'response' must be an object")
        status_code = data.get("status_code", 200)
        # ``bool`` is a subclass of ``int`` — reject it explicitly so a
        # stray ``"status_code": true`` does not silently coerce to 1.
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise ValueError(
                f"fixture {path.name!r}: 'status_code' must be an integer, "
                f"got {type(status_code).__name__}"
            )
        raw_body = data.get("raw_body")
        if raw_body is not None and not isinstance(raw_body, str):
            raise ValueError(
                f"fixture {path.name!r}: 'raw_body' must be a string or null, "
                f"got {type(raw_body).__name__}"
            )
        delay_s = data.get("delay_s", 0.0)
        if isinstance(delay_s, bool) or not isinstance(delay_s, int | float):
            raise ValueError(
                f"fixture {path.name!r}: 'delay_s' must be a number, "
                f"got {type(delay_s).__name__}"
            )
        if delay_s < 0:
            raise ValueError(
                f"fixture {path.name!r}: 'delay_s' must be non-negative, "
                f"got {delay_s!r}"
            )
        fixtures.append(
            Fixture(
                name=path.stem,
                endpoint=endpoint,
                match=match,
                response=response,
                status_code=status_code,
                raw_body=raw_body,
                delay_s=float(delay_s),
            )
        )
    return FixtureSet(fixtures=fixtures)


class _MockLLMRequestHandler(BaseHTTPRequestHandler):
    """Per-request handler bound to a parent :class:`MockLLMServer`."""

    server: ThreadingHTTPServer  # type: ignore[assignment]

    # ``BaseHTTPRequestHandler`` logs to stderr for every request which
    # would spam pytest output. Silence it.
    def log_message(  # noqa: D401 - stdlib override
        self,
        format: str,  # noqa: A002 - stdlib signature
        *args: Any,
    ) -> None:
        return

    def _send_json(self, status: HTTPStatus | int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        fixtures: FixtureSet = self.server.fixtures  # type: ignore[attr-defined]
        endpoint = self.path.split("?", 1)[0].rstrip("/")
        if endpoint == "":  # pragma: no cover - HTTP requires non-empty path
            endpoint = "/"
        if endpoint not in SUPPORTED_ENDPOINTS:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": {
                        "type": "not_found",
                        "message": (
                            f"unknown endpoint {endpoint!r}; supported: "
                            f"{sorted(SUPPORTED_ENDPOINTS)}"
                        ),
                    }
                },
            )
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            request_body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"type": "invalid_json", "message": str(exc)}},
            )
            return

        if not isinstance(request_body, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "request body must be a JSON object",
                    }
                },
            )
            return

        fixture = fixtures.find(endpoint, request_body)
        if fixture is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": {
                        "type": "no_fixture_match",
                        "message": (
                            f"no fixture matched POST {endpoint}; "
                            f"loaded {len(fixtures.for_endpoint(endpoint))} "
                            f"fixture(s) for this endpoint"
                        ),
                    }
                },
            )
            return

        # Fault knobs (task 27.3c): an opt-in fixture can ask the
        # server to delay, emit a non-2xx status, or send a raw
        # (possibly malformed) body. Defaults preserve pre-27.3c
        # behaviour (200 OK, JSON body, no delay).
        if fixture.delay_s > 0:
            time.sleep(fixture.delay_s)
        if fixture.raw_body is not None:
            self._send_raw(fixture.status_code, fixture.raw_body)
            return
        self._send_json(fixture.status_code, fixture.response)

    def _send_raw(self, status: int, body: str) -> None:
        """Send *body* verbatim — used for malformed-JSON fixtures."""
        encoded = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class MockLLMServer:
    """Threaded :class:`http.server.ThreadingHTTPServer` wrapper.

    Typical usage in tests::

        with MockLLMServer(fixtures_dir) as server:
            url = server.url + "/v1/messages"
            ...

    The server picks an ephemeral port by default (``port=0``); read the
    bound port back via :attr:`port` after entering the context manager.
    """

    def __init__(
        self,
        fixtures: FixtureSet | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if isinstance(fixtures, FixtureSet):
            self._fixtures = fixtures
        else:
            self._fixtures = load_fixtures(fixtures)
        self._host = host
        self._requested_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def fixtures(self) -> FixtureSet:
        return self._fixtures

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("MockLLMServer not started")
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        if self._httpd is not None:
            raise RuntimeError("MockLLMServer already started")
        httpd = ThreadingHTTPServer(
            (self._host, self._requested_port), _MockLLMRequestHandler
        )
        httpd.fixtures = self._fixtures  # type: ignore[attr-defined]
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="mock-llm-server",
            daemon=True,
        )
        thread.start()
        self._httpd = httpd
        self._thread = thread

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> MockLLMServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def _list_endpoints() -> Iterable[str]:  # pragma: no cover - debug helper
    return SUPPORTED_ENDPOINTS
