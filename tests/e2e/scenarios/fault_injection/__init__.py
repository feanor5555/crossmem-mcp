"""Mock-only fault-injection scenarios (task 27.3c).

These scenarios deliberately drive the mock LLM server into one of
the four failure modes spelled out in the task DoD:

* :mod:`broken_json` — server emits malformed JSON. The scenario
  asserts the upstream client treats it as an error.
* :mod:`http_5xx` — server returns a 5xx status code. The scenario
  asserts the upstream client raises / surfaces the HTTP error.
* :mod:`timeout_slow` — server delays the response past the client's
  timeout budget. The scenario asserts the client raises a timeout.
* :mod:`empty_tool_calls` — server returns a syntactically valid but
  semantically empty response (no ``content`` blocks / no
  ``tool_calls``). The scenario asserts the client handles "nothing
  to do" without crashing.

Each scenario reads the mock URL from ``CROSSMEM_E2E_MOCK_URL`` (the
env var the mock runner exports) and returns ``0`` if the expected
fault was observed, non-zero otherwise.
"""
