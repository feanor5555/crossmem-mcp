"""End-to-end scenario runners for the LLM-matrix tests (task 27.3a+).

Each submodule wraps **one** upstream LLM endpoint family so the
scenario harness in :mod:`tests.e2e` can swap implementations without
threading provider-specific knobs through the test code:

* :mod:`tests.e2e.runners.qwen` — real Qwen 3.5 backend over an
  OpenAI-compatible REST endpoint (task 27.3a). Skips with exit
  code ``2`` when the endpoint is unreachable so CI without a local
  Qwen keeps going.
* :mod:`tests.e2e.runners.opus` — Anthropic Claude Opus over the
  official Messages API (task 27.3b). Skips with exit code ``2``
  when ``ANTHROPIC_API_KEY`` is unset or the API is unreachable.
* :mod:`tests.e2e.runners.mock` — in-process
  :class:`tests.e2e.mock_llm.MockLLMServer` (task 27.3c). Always
  available; the offline-CI guarantee of the matrix. Also the only
  runner that can deliberately inject faults (broken JSON, HTTP-5xx,
  timeout, empty tool-call response) via the fixtures under
  ``tests/e2e/mock_llm/fixtures/faults/``.

Runners share the report fragment shape mandated by task 27.1:
``{"name", "status", "duration_s", "log_path"}`` plus an optional
``reason`` field for skip / fail entries.
"""
