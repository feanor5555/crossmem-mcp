"""Mock-endpoint E2E runner (task 27.3c).

Unlike the Qwen and Opus runners — which target live endpoints and
*skip* when those endpoints are unreachable — the mock runner is
always available. It owns (or borrows) an in-process
:class:`tests.e2e.mock_llm.MockLLMServer` that emulates Anthropic's
``/v1/messages`` and OpenAI's ``/v1/chat/completions`` from fixture
files. CI hosts without any network access still see the matrix
exercise this column green; live-endpoint regressions therefore
cannot mask a regression in the offline path.

Behaviour
---------

The runner exports the mock-server URL via the ``CROSSMEM_E2E_MOCK_URL``
env var before invoking the scenario and restores the previous value
afterwards. The scenario reads the URL from the same env var the
live runners use (each scenario can decide which knob it wants, but
``ENV_URL_VAR`` is the documented contract).

Two usage modes:

* Implicit server — :func:`run` spins a fresh :class:`MockLLMServer`
  on an ephemeral loopback port for the duration of the scenario.
  Convenient for one-off happy-path scenarios.
* Caller-supplied server — pass ``server=...`` (already started) when
  several scenarios share fixtures or the test author wants to point
  at a faulty fixture set (see ``tests/e2e/mock_llm/fixtures/faults/``).

The fragment schema mirrors the other runners exactly:
``{"name", "status", "duration_s", "log_path"[, "reason"]}``.

Usage
-----

Programmatic (preferred from the scenario harness)::

    from tests.e2e.runners import mock as mock_runner

    exit_code, fragment = mock_runner.run(
        scenario_name="scenarios/happy_path/store_query.py",
        scenario=my_scenario_callable,
        log_path="reports/<ts>/store_query.log",
    )

Command-line wrapper (used by the bash / PowerShell ``run_all``
scripts)::

    python -m tests.e2e.runners.mock --check-only

prints the fragment JSON on stdout and exits ``0`` (the mock is
always reachable, so ``--check-only`` never emits ``skipped``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import TYPE_CHECKING

from tests.e2e.mock_llm import MockLLMServer

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Env-var the mock runner exports for scenarios. Documented in
#: ``tests/e2e/.env.example`` and ``tests/e2e/README.md``. Scenarios
#: are expected to fall back to this var when the live ones are unset
#: (see ``tests/e2e/scenarios/happy_path/store_query.py``).
ENV_URL_VAR = "CROSSMEM_E2E_MOCK_URL"

#: Exit codes shared with the bash / PowerShell ``run_all`` scripts
#: and matched against the qwen / opus runners so the harness can
#: compare codes without provider-specific branches.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SKIPPED = 2  # never produced by this runner; kept for parity


# ---------------------------------------------------------------------------
# Scenario report fragment
# ---------------------------------------------------------------------------


def build_scenario_fragment(
    *,
    name: str,
    status: str,
    duration_s: float,
    log_path: str,
    reason: str | None = None,
) -> dict[str, object]:
    """Return a single ``scenarios[]`` entry per task 27.1's schema.

    Matches the qwen / opus runner fragment shape exactly so the
    matrix report does not need provider-specific normalisation.
    """
    if status not in {"pass", "fail", "skipped"}:
        raise ValueError(f"invalid status {status!r}")
    fragment: dict[str, object] = {
        "name": name,
        "status": status,
        "duration_s": round(float(duration_s), 3),
        "log_path": log_path,
    }
    if status in {"skipped", "fail"}:
        if not reason:
            raise ValueError(f"status={status!r} requires a non-empty reason")
        fragment["reason"] = reason
    elif reason is not None:
        # Defensive: a passing run must not carry a reason — that
        # would either mask test-author confusion or smuggle
        # diagnostics into the happy-path report.
        raise ValueError("status='pass' must not carry a reason")
    return fragment


# ---------------------------------------------------------------------------
# Run wrapper
# ---------------------------------------------------------------------------


def _restore_env_var(previous: str | None) -> None:
    """Restore ``ENV_URL_VAR`` to *previous* (deleting it when ``None``)."""
    if previous is None:
        os.environ.pop(ENV_URL_VAR, None)
    else:
        os.environ[ENV_URL_VAR] = previous


def run(
    *,
    scenario_name: str,
    scenario: Callable[[], int],
    log_path: str,
    server: MockLLMServer | None = None,
) -> tuple[int, dict[str, object]]:
    """Execute *scenario* against an in-process mock LLM server.

    When *server* is ``None`` a fresh :class:`MockLLMServer` is
    started for the duration of the call and stopped afterwards.
    When it is supplied the caller is responsible for the lifecycle —
    typical for fault-injection tests that pre-configure a fixture
    set (see :func:`tests.e2e.test_fault_injection_scenarios._server_for`).

    Returns ``(exit_code, fragment)`` so callers can both feed the
    process exit and append the fragment to a matrix report.
    """
    previous_env = os.environ.get(ENV_URL_VAR)
    owns_server = server is None
    active_server = server if server is not None else MockLLMServer()
    if owns_server:
        active_server.start()
    try:
        os.environ[ENV_URL_VAR] = active_server.url
        started = time.monotonic()
        try:
            scenario_exit = int(scenario())
        except Exception as exc:  # noqa: BLE001 - convert to fail fragment
            duration = time.monotonic() - started
            fragment = build_scenario_fragment(
                name=scenario_name,
                status="fail",
                duration_s=duration,
                log_path=log_path,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return EXIT_FAIL, fragment
        duration = time.monotonic() - started

        if scenario_exit == 0:
            fragment = build_scenario_fragment(
                name=scenario_name,
                status="pass",
                duration_s=duration,
                log_path=log_path,
            )
            return EXIT_OK, fragment

        fragment = build_scenario_fragment(
            name=scenario_name,
            status="fail",
            duration_s=duration,
            log_path=log_path,
            reason=f"scenario exited with exit code {scenario_exit}",
        )
        return EXIT_FAIL, fragment
    finally:
        _restore_env_var(previous_env)
        if owns_server:
            active_server.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.e2e.runners.mock",
        description=(
            "Emit a scenario-report fragment for the offline mock LLM "
            "runner. Always exits 0 — the mock is always available."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Emit a pass fragment without running a scenario. Mirrors "
            "the qwen / opus runners so the shell harness can call "
            "every runner the same way."
        ),
    )
    parser.add_argument(
        "--scenario-name",
        default="mock/connection-check",
        help="Value for the fragment's ``name`` field.",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Value for the fragment's ``log_path`` field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    fragment = build_scenario_fragment(
        name=args.scenario_name,
        status="pass",
        duration_s=0.0,
        log_path=args.log_path,
    )
    sys.stdout.write(json.dumps(fragment) + "\n")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
