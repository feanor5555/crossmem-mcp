"""Opus-endpoint E2E runner (task 27.3b).

Targets the official Anthropic Messages API (Claude Opus). Authentication
uses the ``ANTHROPIC_API_KEY`` env var; the runner does **not** ship a
fallback default — without a key it always skips so CI hosts and
contributors without an Anthropic account see a clean ``status:
"skipped"`` fragment instead of a hard failure.

Behaviour
---------

Before running any scenario the runner sends a minimal probe request to
``POST /v1/messages`` (model ``claude-opus``, ``max_tokens=1``) with a
short timeout. Outcomes:

* **2xx response** — scenario runs normally; the report fragment uses
  ``status="pass"`` or ``"fail"`` plus the wall-clock duration. Exit
  codes follow the harness convention from task 27.1 (``0`` pass,
  ``1`` fail).
* **No key / blank key** — short-circuit to ``status="skipped"`` with
  ``reason="ANTHROPIC_API_KEY not set"`` and exit ``2``. Network is
  never touched.
* **Auth error, rate limit, connect error, timeout, non-2xx** — the
  scenario is skipped with ``status="skipped"`` and an explanatory
  ``reason`` string. Exit code ``2``. The harness keeps going.

The fragment schema matches task 27.1's report
(``{"name", "status", "duration_s", "log_path"}``) plus an optional
``"reason"`` field for skip / fail entries — task 27.3c collects
fragments across runners into the matrix report.

Usage
-----

Programmatic (preferred from the scenario harness in 27.3c)::

    from tests.e2e.runners import opus

    exit_code, fragment = opus.run(
        scenario_name="scenarios/happy_path/store-query.py",
        scenario=my_scenario_callable,
        log_path="reports/<ts>/store-query.log",
    )

Command-line wrapper (used by the bash + PowerShell ``run_all`` scripts
that don't host a Python process)::

    python -m tests.e2e.runners.opus --check-only

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

#: Env-var holding the Anthropic API key. Documented in
#: ``tests/e2e/.env.example`` and ``tests/e2e/README.md``.
ENV_API_KEY_VAR = "ANTHROPIC_API_KEY"

#: Official Anthropic Messages API base URL. Hardcoded on purpose —
#: there is no developer-lab fallback for Opus, unlike the Qwen runner.
DEFAULT_BASE_URL = "https://api.anthropic.com"

#: Claude Opus model identifier. Anthropic publishes dated tags
#: (``claude-opus-4-5``, ``claude-opus-4-7``...); the prefix
#: ``claude-opus`` is the stable family identifier we pin here so the
#: matrix report stays meaningful even if the dated tag rolls.
DEFAULT_MODEL = "claude-opus-4-7"

#: Required ``anthropic-version`` header value. Pinned per Anthropic's
#: published Messages API stability contract.
ANTHROPIC_API_VERSION = "2023-06-01"

#: Probe-call timeout (seconds). Short on purpose — the goal is to skip
#: cleanly on unreachable APIs, not to retry forever.
DEFAULT_TIMEOUT_S = 5.0

#: Exit codes shared with the bash / PowerShell ``run_all`` scripts.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SKIPPED = 2


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiCheckResult:
    """Outcome of :func:`check_api`.

    ``reachable`` is the only field callers care about for the skip
    decision; ``status_code`` and ``reason`` are surfaced so the
    fragment's ``reason`` string can spell out *why* the runner
    skipped (a 401 auth error looks different from a connect error).
    """

    reachable: bool
    status_code: int | None
    reason: str | None


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def resolve_api_key() -> str | None:
    """Return the Anthropic API key from the environment, or ``None``.

    A blank / whitespace-only value is treated as "unset" so users who
    ``source .env.example`` unchanged don't accidentally send an empty
    ``x-api-key`` header to Anthropic.
    """
    value = os.environ.get(ENV_API_KEY_VAR, "").strip()
    if not value:
        return None
    return value


# ---------------------------------------------------------------------------
# API probe
# ---------------------------------------------------------------------------


def check_api(
    *,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> ApiCheckResult:
    """Probe ``POST {base_url}/v1/messages`` with a minimal payload.

    The probe asks Claude for one token of output to keep the cost
    floor as low as possible while still exercising auth + routing.
    Any connect-level error, timeout, or non-2xx response is treated
    as "unreachable" so the harness skips the scenario instead of
    failing.
    """
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        return ApiCheckResult(
            reachable=False,
            status_code=None,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if 200 <= response.status_code < 300:
        return ApiCheckResult(
            reachable=True, status_code=response.status_code, reason=None
        )
    return ApiCheckResult(
        reachable=False,
        status_code=response.status_code,
        reason=f"HTTP {response.status_code} from /v1/messages",
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

    ``reason`` is omitted from the dict for ``status="pass"`` so happy-
    path entries stay identical to the bash runner's output. Skip and
    fail entries always include ``reason`` (callers must pass it).
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
        # Defensive: never silently drop a reason on a passing run.
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
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> tuple[int, dict[str, object]]:
    """Execute one scenario gated on an Anthropic-API probe.

    The ``scenario`` callable is expected to return a process-style
    exit code (``0`` pass, non-zero fail). It is only invoked when the
    probe succeeds; otherwise the runner short-circuits to a skip
    fragment.

    Returns ``(exit_code, fragment)`` so callers can both feed the
    process exit and append the fragment to a matrix report.
    """
    resolved_key = api_key.strip() if api_key is not None else resolve_api_key()
    if not resolved_key:
        fragment = build_scenario_fragment(
            name=scenario_name,
            status="skipped",
            duration_s=0.0,
            log_path=log_path,
            reason=f"{ENV_API_KEY_VAR} not set",
        )
        return EXIT_SKIPPED, fragment

    check = check_api(
        api_key=resolved_key, timeout=timeout, base_url=base_url, model=model
    )
    if not check.reachable:
        fragment = build_scenario_fragment(
            name=scenario_name,
            status="skipped",
            duration_s=0.0,
            log_path=log_path,
            reason=f"api unreachable: {check.reason}",
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
        prog="python -m tests.e2e.runners.opus",
        description=(
            "Probe the Anthropic Claude Opus API and emit a "
            "scenario-report fragment on stdout."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Only run the API probe; emit a JSON fragment with "
            "status pass/skipped to stdout and exit 0 / 2 accordingly."
        ),
    )
    parser.add_argument(
        "--scenario-name",
        default="opus/connection-check",
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
        help="Probe-call timeout in seconds (default %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    api_key = resolve_api_key()
    if not api_key:
        fragment = build_scenario_fragment(
            name=args.scenario_name,
            status="skipped",
            duration_s=0.0,
            log_path=args.log_path,
            reason=f"{ENV_API_KEY_VAR} not set",
        )
        sys.stdout.write(json.dumps(fragment) + "\n")
        return EXIT_SKIPPED

    check = check_api(api_key=api_key, timeout=args.timeout)
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
        reason=f"api unreachable: {check.reason}",
    )
    sys.stdout.write(json.dumps(fragment) + "\n")
    return EXIT_SKIPPED


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
