"""Scenario callables consumed by the LLM-matrix runners (task 27.3c).

Each subpackage groups one *category* of scenarios:

* :mod:`tests.e2e.scenarios.happy_path` — store -> query roundtrip and
  any other basic flow that must work against every runner.
* :mod:`tests.e2e.scenarios.fault_injection` — mock-only scenarios
  that *expect* the upstream LLM to misbehave (broken JSON,
  HTTP-5xx, timeout, empty tool-call response).

Every scenario module exposes a top-level ``run() -> int`` callable.
Return ``0`` on success, non-zero on failure — the matrix runners
treat the return value as a process-style exit code.
"""
