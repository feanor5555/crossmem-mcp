"""Smoke test that verifies ``pytest-httpx`` is loaded as a pytest plugin.

The plugin is required for upcoming HTTP-mocking refactors (tasks 0.5.2a-d).
Declaring it as a dev dep is only useful if pytest actually picks it up, so
this smoke test guards against missing-from-environment regressions.
"""

from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version


def test_pytest_httpx_plugin_loaded(pytestconfig) -> None:
    assert pytestconfig.pluginmanager.hasplugin("pytest_httpx"), (
        "pytest_httpx plugin not registered; ensure pytest-httpx is installed"
    )


def test_httpx_mock_fixture_available(pytestconfig) -> None:
    fixture_names = pytestconfig.pluginmanager.get_plugin(
        "pytest_httpx"
    ).__dict__.keys()
    # Plugin module exposes the fixture function ``httpx_mock``.
    assert "httpx_mock" in fixture_names, (
        "httpx_mock fixture not exposed by pytest_httpx"
    )


def test_pytest_httpx_version_meets_minimum() -> None:
    installed = Version(version("pytest-httpx"))
    assert installed >= Version("0.30"), f"pytest-httpx {installed} < required 0.30"
