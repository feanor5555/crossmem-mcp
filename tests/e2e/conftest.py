"""Pytest fixtures for the LLM-matrix tests (task 27.3c).

Exposes a single fixture, ``runner``, parametrised across
:data:`tests.e2e.matrix.RUNNER_NAMES`. Any test in :mod:`tests.e2e`
that takes a ``runner`` argument therefore runs once per backend in
``{qwen, opus, mock}``. Each parametrised case yields the runner
*module* (not just the name) so the test body can compare exit codes
against ``runner.EXIT_OK`` / ``EXIT_SKIPPED`` etc. without importing
the modules individually.

Skip semantics live inside the runner modules themselves — the
fixture does not pre-skip cases. The test body inspects the
``(exit_code, fragment)`` tuple the runner returns and decides
whether a ``skipped`` outcome is acceptable (it is, for happy-path
scenarios).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.matrix import RUNNER_NAMES, get_runner

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(params=RUNNER_NAMES, ids=list(RUNNER_NAMES))
def runner(request: pytest.FixtureRequest) -> ModuleType:
    """Yield each runner module in turn for the parametrised test."""
    return get_runner(request.param)
