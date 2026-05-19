"""LLM-runner matrix definition (task 27.3c).

A single source of truth for the runner names the happy-path scenarios
parametrise over: ``qwen``, ``opus`` and ``mock``. The pytest conftest
in :mod:`tests.e2e.conftest` consumes :data:`RUNNER_NAMES` to drive
:func:`pytest.fixture.params`, and shell scripts (``run_all.sh``,
``run_all.ps1``) import or shell out to the same name list so the
matrix shape stays consistent across the harness.

Keeping the runner-name -> module mapping in one helper means new
runners only need to be registered here; the conftest, the README and
the shell scripts pick the change up automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.e2e.runners import mock as _mock_runner
from tests.e2e.runners import opus as _opus_runner
from tests.e2e.runners import qwen as _qwen_runner

if TYPE_CHECKING:
    from types import ModuleType

#: Ordered tuple of runner names the matrix parametrises over. The
#: order is the order rows appear in the matrix report — keep
#: ``qwen`` first so the developer-lab default is the most visible
#: column on a green run.
RUNNER_NAMES: tuple[str, ...] = ("qwen", "opus", "mock")

_RUNNERS: dict[str, ModuleType] = {
    "qwen": _qwen_runner,
    "opus": _opus_runner,
    "mock": _mock_runner,
}


def get_runner(name: str) -> ModuleType:
    """Resolve a runner name to its module.

    Raises :class:`ValueError` on an unknown name so test-authors get
    a loud error rather than a silent skip when they typo a runner.
    """
    try:
        return _RUNNERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown runner {name!r}; expected one of {sorted(_RUNNERS)}"
        ) from exc


__all__ = ["RUNNER_NAMES", "get_runner"]
