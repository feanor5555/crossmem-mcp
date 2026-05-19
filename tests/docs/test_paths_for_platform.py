"""Pure-function platform-path resolution for doc rendering.

The renderer used to flip ``sys.platform`` and ``os.environ['APPDATA']``
globally inside a contextmanager to coax each connector's
``config_path()`` into returning the per-platform variant. That made the
renderer thread-hostile — under ``pytest -n auto`` a parallel test that
ran concurrently with a render could observe the mutated globals and
fail intermittently.

Task 21.4 replaces the global-mutation trick with a pure
``connector.paths_for_platform(platform, *, appdata) -> Path`` method
plus a renderer that calls it three times (linux, darwin, win32) without
touching globals. These tests pin that contract:

* the renderer must not read or write ``sys.platform`` while building
  the triplet,
* the renderer must not read or write ``os.environ['APPDATA']`` while
  building the triplet,
* calling ``paths_for_platform`` on each shipped connector returns the
  same path that the legacy ``sys.platform``-monkey-patch route used
  to produce, and
* rendering an install doc from one thread does not perturb a parallel
  thread's view of ``sys.platform`` / ``APPDATA``.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from crossmem.connectors.amazonq import AmazonQConnector
from crossmem.connectors.claude_code import ClaudeCodeConnector
from crossmem.connectors.cline import ClineConnector
from crossmem.connectors.continuedev import ContinueDevConnector
from crossmem.connectors.cursor import CursorConnector
from crossmem.connectors.gemini import GeminiConnector
from crossmem.connectors.goose import GooseConnector
from crossmem.connectors.kilocode import KiloCodeConnector
from crossmem.connectors.opencode import OpenCodeConnector
from crossmem.connectors.pi import PiConnector
from crossmem.connectors.windsurf import WindsurfConnector
from crossmem.connectors.zed import ZedConnector
from crossmem.docs.install_template import (
    CONNECTOR_REGISTRY,
    render_install_doc,
)

if TYPE_CHECKING:
    from crossmem.connectors.base import CLIConnector


# ---------------------------------------------------------------------------
# paths_for_platform contract
# ---------------------------------------------------------------------------


def test_base_class_exposes_paths_for_platform() -> None:
    """Every connector inherits a ``paths_for_platform`` callable."""
    for cls in CONNECTOR_REGISTRY.values():
        assert hasattr(cls, "paths_for_platform")
        assert callable(cls.paths_for_platform)


@pytest.mark.parametrize(
    ("platform", "appdata"),
    [("linux", None), ("darwin", None), ("win32", r"%APPDATA%")],
)
def test_paths_for_platform_is_pure(
    platform: str, appdata: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling ``paths_for_platform`` does not mutate process globals."""
    saved_platform = sys.platform
    saved_appdata = os.environ.get("APPDATA")
    for cls in CONNECTOR_REGISTRY.values():
        cls().paths_for_platform(platform, appdata=appdata)
    assert sys.platform == saved_platform
    assert os.environ.get("APPDATA") == saved_appdata


def test_render_install_doc_does_not_mutate_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``render_install_doc`` must not touch ``sys.platform``/``APPDATA``."""
    saved_platform = sys.platform
    monkeypatch.setenv("APPDATA", "SENTINEL_APPDATA_VALUE")
    saved_appdata = os.environ["APPDATA"]
    for cli in CONNECTOR_REGISTRY:
        render_install_doc(cli)
    assert sys.platform == saved_platform
    assert os.environ.get("APPDATA") == saved_appdata


# ---------------------------------------------------------------------------
# Per-connector path expectations
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _both_unix(connector: CLIConnector, expected_rel: tuple[str, ...]) -> None:
    """Assert linux + darwin both resolve to ``home / expected_rel``."""
    home = Path.home()
    expected = home.joinpath(*expected_rel)
    assert connector.paths_for_platform("linux", appdata=None) == expected
    assert connector.paths_for_platform("darwin", appdata=None) == expected


def test_claude_code_paths(fake_home: Path) -> None:
    _both_unix(ClaudeCodeConnector(), (".claude.json",))
    assert (
        ClaudeCodeConnector().paths_for_platform("win32", appdata=r"X:\\AppData")
        == fake_home / ".claude.json"
    )


def test_cursor_paths(fake_home: Path) -> None:
    _both_unix(CursorConnector(), (".cursor", "mcp.json"))
    assert (
        CursorConnector().paths_for_platform("win32", appdata=r"X:\\AppData")
        == fake_home / ".cursor" / "mcp.json"
    )


def test_gemini_paths(fake_home: Path) -> None:
    _both_unix(GeminiConnector(), (".gemini", "settings.json"))


def test_amazonq_paths(fake_home: Path) -> None:
    _both_unix(AmazonQConnector(), (".aws", "amazonq", "mcp.json"))


def test_pi_paths(fake_home: Path) -> None:
    _both_unix(PiConnector(), (".pi", "agent", "mcp.json"))


def test_windsurf_paths(fake_home: Path) -> None:
    _both_unix(WindsurfConnector(), (".codeium", "windsurf", "mcp_config.json"))


def test_continuedev_paths(fake_home: Path) -> None:
    _both_unix(ContinueDevConnector(), (".continue", "config.json"))


def test_opencode_paths(fake_home: Path) -> None:
    home = Path.home()
    assert OpenCodeConnector().paths_for_platform("linux", appdata=None) == (
        home / ".config" / "opencode" / "opencode.json"
    )
    assert OpenCodeConnector().paths_for_platform("darwin", appdata=None) == (
        home / ".config" / "opencode" / "opencode.json"
    )
    assert (
        OpenCodeConnector().paths_for_platform("win32", appdata=r"X:\\AppData")
        == Path(r"X:\\AppData") / "opencode" / "opencode.json"
    )


def test_opencode_paths_windows_no_appdata(fake_home: Path) -> None:
    assert OpenCodeConnector().paths_for_platform("win32", appdata=None) == (
        fake_home / "AppData" / "Roaming" / "opencode" / "opencode.json"
    )


def test_zed_paths(fake_home: Path) -> None:
    home = Path.home()
    expected_unix = home / ".config" / "zed" / "settings.json"
    assert ZedConnector().paths_for_platform("linux", appdata=None) == expected_unix
    assert ZedConnector().paths_for_platform("darwin", appdata=None) == expected_unix
    assert ZedConnector().paths_for_platform("win32", appdata=r"X:\\AppData") == (
        Path(r"X:\\AppData") / "Zed" / "settings.json"
    )


def test_goose_paths(fake_home: Path) -> None:
    home = Path.home()
    expected_unix = home / ".config" / "goose" / "config.yaml"
    assert GooseConnector().paths_for_platform("linux", appdata=None) == expected_unix
    assert GooseConnector().paths_for_platform("darwin", appdata=None) == expected_unix
    assert GooseConnector().paths_for_platform("win32", appdata=r"X:\\AppData") == (
        Path(r"X:\\AppData") / "goose" / "config.yaml"
    )


def test_goose_paths_windows_no_appdata(fake_home: Path) -> None:
    assert GooseConnector().paths_for_platform("win32", appdata=None) == (
        fake_home / "AppData" / "Roaming" / "goose" / "config.yaml"
    )


def test_cline_paths_linux(fake_home: Path) -> None:
    home = Path.home()
    expected = (
        home
        / ".config"
        / "Code"
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
        / "cline_mcp_settings.json"
    )
    assert ClineConnector().paths_for_platform("linux", appdata=None) == expected


def test_cline_paths_mac(fake_home: Path) -> None:
    home = Path.home()
    expected = (
        home
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
        / "cline_mcp_settings.json"
    )
    assert ClineConnector().paths_for_platform("darwin", appdata=None) == expected


def test_cline_paths_windows(fake_home: Path) -> None:
    expected = (
        Path(r"X:\\AppData")
        / "Code"
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
        / "cline_mcp_settings.json"
    )
    assert (
        ClineConnector().paths_for_platform("win32", appdata=r"X:\\AppData") == expected
    )


def test_kilocode_paths_linux(fake_home: Path) -> None:
    home = Path.home()
    expected = (
        home
        / ".config"
        / "Code"
        / "User"
        / "globalStorage"
        / "kilocode.kilo-code"
        / "settings"
        / "mcp_settings.json"
    )
    assert KiloCodeConnector().paths_for_platform("linux", appdata=None) == expected


def test_kilocode_paths_windows(fake_home: Path) -> None:
    expected = (
        Path(r"X:\\AppData")
        / "Code"
        / "User"
        / "globalStorage"
        / "kilocode.kilo-code"
        / "settings"
        / "mcp_settings.json"
    )
    assert (
        KiloCodeConnector().paths_for_platform("win32", appdata=r"X:\\AppData")
        == expected
    )


# ---------------------------------------------------------------------------
# Concurrency: parallel renders must not interfere
# ---------------------------------------------------------------------------


def test_parallel_renders_do_not_interfere() -> None:
    """Two threads rendering different CLIs concurrently must succeed.

    Before Task 21.4 this could fail because thread A would flip
    ``sys.platform`` to ``win32`` while thread B was about to read
    ``Path.home()`` for a Linux render. Pure-function path resolution
    eliminates the shared mutable state.
    """
    errors: list[BaseException] = []

    def render_many(cli: str) -> None:
        try:
            for _ in range(50):
                render_install_doc(cli)
        except BaseException as exc:  # noqa: BLE001 — propagate to caller
            errors.append(exc)

    threads = [
        threading.Thread(target=render_many, args=(cli,))
        for cli in ("claude_code", "opencode", "zed", "goose", "cline")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
