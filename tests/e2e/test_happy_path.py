"""End-to-end happy-path scenarios parametrised over the LLM matrix.

This module is the single consumer of the ``runner`` fixture declared
in :mod:`tests.e2e.conftest`. Each test exercises one happy-path
scenario callable against each runner in ``{qwen, opus, mock}`` —
``qwen``/``opus`` skip cleanly when their endpoints are unreachable
(see ``tests/e2e/README.md``), ``mock`` always runs because the
in-process :class:`MockLLMServer` has no external dependency.

Adding a new happy-path scenario:

1. Drop the callable in ``tests/e2e/scenarios/happy_path/<name>.py``
   with a top-level ``run() -> int``.
2. Reference it from a new ``test_<name>`` function here that delegates
   to the ``runner`` fixture's ``run`` callable.

The matrix runs in the regular ``pytest`` invocation, but every
scenario is short — no live LLM round-trips happen unless the
operator deliberately exports ``CROSSMEM_E2E_QWEN_URL`` or
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from tests.e2e.scenarios.happy_path import store_query


def test_store_query_roundtrip(runner) -> None:  # noqa: ANN001 - pytest fixture
    """Store -> query roundtrip through the in-process MCP store.

    The scenario callable is identical for every runner; the matrix
    layer decides whether to actually exercise the LLM (live for
    qwen/opus, in-process MockLLMServer for mock) or skip.
    """
    exit_code, fragment = runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=store_query.run,
        log_path="reports/happy_path/store_query.log",
    )
    # Either the scenario passed, or the runner skipped because its
    # endpoint is unreachable on this host. A failure means a real
    # regression — surface it loudly.
    assert exit_code in (runner.EXIT_OK, runner.EXIT_SKIPPED), fragment
    if exit_code == runner.EXIT_OK:
        assert fragment["status"] == "pass"
    else:
        assert fragment["status"] == "skipped"
        assert fragment["reason"]
