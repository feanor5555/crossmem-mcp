"""Qwen-endpoint E2E runner (task 27.3a).

Targets a *real* Qwen 3.5 backend exposed over an OpenAI-compatible
REST endpoint. Default URL is ``http://192.168.178.45:8080`` (the
developer's lab box); the env var ``CROSSMEM_E2E_QWEN_URL`` overrides
that for everyone else.

Behaviour
---------

Before running any scenario the runner pings ``GET /v1/models`` with a
short timeout. Two outcomes:

* **Reachable** (HTTP 2xx) — scenario runs normally; the report
  fragment carries ``status="pass"`` or ``"fail"`` plus the wall-clock
  duration. Exit codes follow the harness convention from task 27.1
  (``0`` = pass, ``1`` = fail).
* **Unreachable** (connect error, timeout, non-2xx) — the scenario is
  *skipped*. The runner emits a fragment with ``status="skipped"`` and
  a ``reason`` string and exits ``2``. CI hosts without a local Qwen
  see a clean skip instead of a hard failure.

The fragment schema is the one fixed by task 27.1's report
(``{"name", "status", "duration_s", "log_path"}``) plus an optional
``"reason"`` field for skip / fail entries — task 27.3c will collect
fragments across runners into the matrix report.

Usage
-----

Programmatic (preferred from the scenario harness in 27.3c)::

    from tests.e2e.runners import qwen

    exit_code, fragment = qwen.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=my_scenario_callable,
        log_path="reports/<ts>/store-query.log",
    )

Command-line wrapper (used by the bash + PowerShell ``run_all`` scripts
that don't host a Python process)::

    python -m tests.e2e.runners.qwen --check-only

prints the fragment JSON on stdout and exits ``0`` / ``1`` / ``2`` so a
shell ``run_all.sh`` loop can record it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Env-var that overrides :data:`DEFAULT_URL`. Empty string falls back
#: to the default so users can ``source .env.example`` unchanged
#: without accidentally pointing at an empty host.
ENV_URL_VAR = "CROSSMEM_E2E_QWEN_URL"

#: Developer-lab default (per task 27.3a DoD). Documented so reviewers
#: don't have to grep for the literal.
DEFAULT_URL = "http://192.168.178.45:8080"

#: OpenAI-compatible model identifier. ``qwen-3.5`` is the model name
#: the running Qwen instance advertises in ``/v1/models``.
DEFAULT_MODEL = "qwen-3.5"

#: Connection-check timeout (seconds). Short on purpose — the goal is
#: to skip cleanly on unreachable hosts, not to retry forever.
DEFAULT_TIMEOUT_S = 2.0

#: Exit codes shared with the bash / PowerShell ``run_all`` scripts.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SKIPPED = 2


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionCheckResult:
    """Outcome of :func:`check_connection`.

    ``reachable`` is the only field callers care about for the skip
    decision; ``status_code`` and ``reason`` are surfaced so the
    fragment's ``reason`` string can spell out *why* the runner
    skipped (an unreachable port looks different from a 503 from a
    half-booted backend).
    """

    reachable: bool
    status_code: int | None
    reason: str | None


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def resolve_url() -> str:
    """Return the Qwen base URL the runner should use.

    Empty env var falls back to :data:`DEFAULT_URL` (see comment on
    :data:`ENV_URL_VAR`). Trailing slashes are stripped so callers can
    join ``/v1/models`` without producing a double-slash path.
    """
    value = os.environ.get(ENV_URL_VAR, "").strip()
    if not value:
        return DEFAULT_URL
    return value.rstrip("/")


# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------


def check_connection(
    base_url: str, *, timeout: float = DEFAULT_TIMEOUT_S
) -> ConnectionCheckResult:
    """Probe ``GET {base_url}/v1/models`` and classify the response.

    Any connect-level error, timeout or non-2xx response is treated as
    "unreachable" — the test harness should *skip* rather than fail
    because the developer's lab Qwen is intentionally optional.
    """
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        # ``HTTPError`` is httpx's umbrella for connect / read /
        # write / timeout exceptions. The class name plus the
        # message is enough context for the report's ``reason``.
        return ConnectionCheckResult(
            reachable=False,
            status_code=None,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if 200 <= response.status_code < 300:
        return ConnectionCheckResult(
            reachable=True, status_code=response.status_code, reason=None
        )
    return ConnectionCheckResult(
        reachable=False,
        status_code=response.status_code,
        reason=f"HTTP {response.status_code} from /v1/models",
    )


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

    ``reason`` is omitted from the dict for ``status="pass"`` so the
    happy-path entries stay identical to the bash runner's output.
    Skip and fail entries always include ``reason`` (callers must pass
    it).
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
        # Defensive: never silently drop a reason on a passing run —
        # that would mask test-author confusion.
        raise ValueError("status='pass' must not carry a reason")
    return fragment


# ---------------------------------------------------------------------------
# Run wrapper
# ---------------------------------------------------------------------------


def run(
    *,
    scenario_name: str,
    scenario: Callable[[], int],
    log_path: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    base_url: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Execute one scenario gated on a Qwen-endpoint reachability check.

    The ``scenario`` callable is expected to return a process-style
    exit code (``0`` pass, non-zero fail). It is only invoked when the
    connection check succeeds; otherwise the runner short-circuits to
    a skip fragment.

    Returns ``(exit_code, fragment)`` so callers can both feed the
    process exit and append the fragment to a matrix report.
    """
    resolved = base_url if base_url is not None else resolve_url()
    check = check_connection(resolved, timeout=timeout)
    if not check.reachable:
        fragment = build_scenario_fragment(
            name=scenario_name,
            status="skipped",
            duration_s=0.0,
            log_path=log_path,
            reason=f"endpoint unreachable: {check.reason}",
        )
        return EXIT_SKIPPED, fragment

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.e2e.runners.qwen",
        description=(
            "Probe a Qwen 3.5 OpenAI-compatible endpoint and emit a "
            "scenario-report fragment on stdout."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Only run the reachability check; emit a JSON fragment with "
            "status pass/skipped to stdout and exit 0 / 2 accordingly."
        ),
    )
    parser.add_argument(
        "--scenario-name",
        default="qwen/connection-check",
        help="Value for the fragment's ``name`` field.",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Value for the fragment's ``log_path`` field.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Connection-check timeout in seconds (default %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    base_url = resolve_url()
    check = check_connection(base_url, timeout=args.timeout)
    if check.reachable:
        fragment = build_scenario_fragment(
            name=args.scenario_name,
            status="pass",
            duration_s=0.0,
            log_path=args.log_path,
        )
        sys.stdout.write(json.dumps(fragment) + "\n")
        return EXIT_OK
    fragment = build_scenario_fragment(
        name=args.scenario_name,
        status="skipped",
        duration_s=0.0,
        log_path=args.log_path,
        reason=f"endpoint unreachable: {check.reason}",
    )
    sys.stdout.write(json.dumps(fragment) + "\n")
    return EXIT_SKIPPED


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
