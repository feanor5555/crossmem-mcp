"""Mock LLM HTTP server for end-to-end CLI roundtrip tests.

The server emulates two upstream chat APIs the supported CLIs talk to:

* Anthropic ``POST /v1/messages`` -> response with ``content`` blocks
  containing one or more ``{"type": "tool_use", ...}`` entries.
* OpenAI ``POST /v1/chat/completions`` -> response with
  ``choices[0].message.tool_calls`` listing tool invocations.

Mapping from request -> canned response is driven by JSON fixtures in
``tests/e2e/mock_llm/fixtures/``. The server is intentionally built on the
Python stdlib (no FastAPI / Starlette) so the ``e2e`` extra stays
runtime-dependency-free.

Public surface:

* :class:`MockLLMServer` — context-manager wrapper that binds the
  threaded HTTP server to ``127.0.0.1:<port>`` (port 0 by default for
  test isolation).
* :func:`load_fixtures` — load a directory of fixture files for the
  request-matching layer.
"""

from __future__ import annotations

from .server import (
    DEFAULT_FIXTURES_DIR,
    FixtureSet,
    MockLLMServer,
    load_fixtures,
)

__all__ = [
    "DEFAULT_FIXTURES_DIR",
    "FixtureSet",
    "MockLLMServer",
    "load_fixtures",
]
