"""Smoke tests for runtime dependencies declared in pyproject.toml.

These tests verify that runtime deps used by ``src/crossmem/sources/`` are
both importable and meet the minimum versions declared in ``pyproject.toml``.
This guards against transitive-only installs that disappear when an upstream
dep drops them.
"""

from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version


def test_beautifulsoup4_importable() -> None:
    import bs4  # noqa: F401  (smoke import)


def test_httpx_importable() -> None:
    import httpx  # noqa: F401  (smoke import)


def test_pyyaml_importable() -> None:
    import yaml

    assert yaml.safe_load("a: 1") == {"a": 1}


def test_beautifulsoup4_version_meets_minimum() -> None:
    installed = Version(version("beautifulsoup4"))
    assert installed >= Version("4.12"), f"beautifulsoup4 {installed} < required 4.12"


def test_httpx_version_meets_minimum() -> None:
    installed = Version(version("httpx"))
    assert installed >= Version("0.27"), f"httpx {installed} < required 0.27"


def test_pyyaml_version_meets_minimum() -> None:
    installed = Version(version("pyyaml"))
    assert installed >= Version("6.0"), f"pyyaml {installed} < required 6.0"
