"""Tests for the LLM-matrix scenario harness (task 27.3c).

The matrix layer lives in :mod:`tests.e2e.conftest` (the pytest
``runner`` parametrisation fixture) and :mod:`tests.e2e.scenarios.\
happy_path.store_query` (one shared scenario callable consumed by every
runner). These tests assert:

* the conftest exposes the parametrisation across ``{qwen, opus, mock}``
  so :mod:`tests.e2e.test_happy_path` and any future happy-path module
  picks the matrix up automatically,
* the happy-path callable performs a real store -> query roundtrip
  through the in-process MCP store (so unit-level coverage of the
  scenario stays honest even when no live LLM is reachable),
* the runner-resolution helper maps name -> runner module and refuses
  unknown names so authors get a loud error rather than a silent skip.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------


def test_matrix_module_exposes_runner_names() -> None:
    """The matrix definition is the single source of truth for the runners."""
    from tests.e2e import matrix

    assert tuple(matrix.RUNNER_NAMES) == ("qwen", "opus", "mock")


def test_matrix_get_runner_returns_module() -> None:
    from tests.e2e import matrix

    qwen = matrix.get_runner("qwen")
    opus = matrix.get_runner("opus")
    mock = matrix.get_runner("mock")
    # Each runner exposes the same ``run`` surface so the harness can
    # call them interchangeably.
    for runner in (qwen, opus, mock):
        assert hasattr(runner, "run")
        assert hasattr(runner, "EXIT_OK")
        assert hasattr(runner, "EXIT_FAIL")
        assert hasattr(runner, "EXIT_SKIPPED")


def test_matrix_get_runner_rejects_unknown_name() -> None:
    from tests.e2e import matrix

    with pytest.raises(ValueError, match="unknown runner"):
        matrix.get_runner("gemini")


# ---------------------------------------------------------------------------
# Happy-path scenario
# ---------------------------------------------------------------------------


def test_happy_path_scenario_is_callable() -> None:
    scenario_module = importlib.import_module(
        "tests.e2e.scenarios.happy_path.store_query"
    )
    assert callable(scenario_module.run)


def test_happy_path_scenario_returns_zero_on_success() -> None:
    scenario_module = importlib.import_module(
        "tests.e2e.scenarios.happy_path.store_query"
    )
    # ``run()`` exercises the in-process MCP server / KnowledgeStore.
    # No external network, no live LLM — the matrix layer is what
    # decides which runner the scenario *also* runs against.
    assert scenario_module.run() == 0


def test_happy_path_scenario_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the roundtrip silently drops the doc, ``run()`` must return non-zero.

    We force the failure by monkeypatching the store factory to return
    a backend whose ``query`` always yields ``[]`` regardless of input.
    """
    scenario_module = importlib.import_module(
        "tests.e2e.scenarios.happy_path.store_query"
    )

    class _BrokenStore:
        def store(self, **kwargs: object) -> list[str]:  # noqa: ARG002
            return ["x"]

        def query(self, *args: object, **kwargs: object) -> list[object]:  # noqa: ARG002
            return []

    monkeypatch.setattr(scenario_module, "_build_store", lambda: _BrokenStore())
    assert scenario_module.run() != 0


# ---------------------------------------------------------------------------
# Pytest conftest hook
# ---------------------------------------------------------------------------


def test_conftest_provides_runner_fixture() -> None:
    """``tests/e2e/conftest.py`` parametrises a ``runner`` fixture across the matrix."""
    conftest_path = REPO_ROOT / "tests" / "e2e" / "conftest.py"
    assert conftest_path.is_file(), conftest_path
    text = conftest_path.read_text(encoding="utf-8")
    assert "RUNNER_NAMES" in text
    # The fixture is consumed via pytest indirection; just check that
    # the conftest declares the parametrisation rather than running a
    # full pytest sub-invocation here.
    assert "parametrize" in text or "params=" in text


def test_test_happy_path_module_picks_up_matrix() -> None:
    """The happy-path test module must use the ``runner`` fixture from conftest."""
    test_path = REPO_ROOT / "tests" / "e2e" / "test_happy_path.py"
    assert test_path.is_file(), test_path
    text = test_path.read_text(encoding="utf-8")
    assert "runner" in text  # signature uses the fixture
    assert "store_query" in text  # scenario is exercised
