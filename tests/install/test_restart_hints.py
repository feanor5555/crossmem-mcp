"""Tests for restart hints in installer output (Task 16.2).

After ``connector.register()`` succeeds, the installer prints a
``Restart <name> to pick up changes`` line — but only for GUI-style
CLIs (``is_gui_app=True``). CLI-style tools pick up the new config on
their next invocation and need no restart hint.

The exact phrasing comes from :meth:`CLIConnector.restart_hint` when the
connector overrides it, falling back to a generic ``Restart <name> to
pick up changes`` string. Undetected connectors emit nothing on the
restart-hint channel at all.

This module pins the behaviour with an exact stdout snapshot covering
the three relevant connector shapes (CLI, GUI, undetected) — a single
regression will diff visibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from crossmem import installer
from crossmem.connectors.base import CLIConnector


@dataclass
class _StubConnector(CLIConnector):
    """Mock connector with explicit ``is_gui_app`` toggle.

    Mirrors the ``_RecordingConnector`` helpers in the sibling test
    files but exposes ``is_gui`` so we can flip the GUI flag per-test
    without subclassing for every variant. Does NOT override
    :meth:`restart_hint` — the installer's restart-hint helper detects
    that and falls back to the short ``Restart <name> to pick up
    changes`` string we want to assert against.
    """

    cli_name: str
    detected: bool
    config_file: Path
    is_gui: bool = False
    register_calls: list[str] = field(default_factory=list)

    @property
    def is_gui_app(self) -> bool:  # type: ignore[override]
        # Mirror the class-attribute slot on the base so the installer can
        # branch on ``connector.is_gui_app`` regardless of subclass shape.
        return self.is_gui

    def name(self) -> str:
        return self.cli_name

    def detect(self) -> bool:
        return self.detected

    def config_path(self) -> Path:
        return self.config_file

    def register(self, server_cmd: str) -> None:
        self.register_calls.append(server_cmd)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text("{}", encoding="utf-8")

    def unregister(self) -> None:  # pragma: no cover - not exercised here
        return None


@dataclass
class _StubConnectorWithCustomHint(_StubConnector):
    """Variant that overrides :meth:`restart_hint` with a fixed phrase.

    Used to assert that a connector's own (tool-specific) hint takes
    precedence over the installer's short fallback. Kept as a separate
    class so the default stub does NOT trip the "override detected"
    branch of ``installer._restart_hint_line``.
    """

    custom_hint: str = ""

    def restart_hint(self) -> str:
        return self.custom_hint


class _MockEmbedder:
    """Embedder stand-in — same pattern as ``test_installer._MockEmbedder``."""

    def __init__(self) -> None:
        self.model_name = "mock-model"


class _ExplodingEmbedder:
    """Embedder that fails if instantiated — used in dry-run assertions."""

    def __init__(self) -> None:  # pragma: no cover - must not be called
        msg = "embedder must not be built during dry-run"
        raise AssertionError(msg)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch ``Path.home`` to a tmp dir so the installer writes there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _ok_doctor() -> list:
    """Doctor stub returning a single ``ok`` result — keeps install going."""
    from crossmem.doctor import CheckResult

    return [CheckResult(name="stub", status="ok", detail="stub ok")]


def test_gui_connector_emits_restart_hint_after_register(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """GUI CLI -> a ``Restart <name>`` line appears, anchored to that name.

    The hint must reference the connector's ``name()`` so a user with
    multiple GUIs detected can tell which apps need a restart.
    """
    gui = _StubConnector(
        cli_name="cursor",
        detected=True,
        config_file=fake_home / ".cursor" / "config.json",
        is_gui=True,
    )

    installer.install(
        connectors=[gui],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    out = capsys.readouterr().out
    restart_lines = [ln for ln in out.splitlines() if "Restart" in ln]
    assert len(restart_lines) == 1
    assert "cursor" in restart_lines[0]


def test_cli_connector_emits_no_restart_hint(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-GUI CLI -> no ``Restart`` line at all.

    CLI-driven tools re-read their config on every invocation, so
    surfacing a restart hint would just add noise.
    """
    cli = _StubConnector(
        cli_name="claude_code",
        detected=True,
        config_file=fake_home / ".claude" / "config.json",
        is_gui=False,
    )

    installer.install(
        connectors=[cli],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    out = capsys.readouterr().out
    restart_lines = [ln for ln in out.splitlines() if "Restart" in ln]
    assert restart_lines == []


def test_undetected_connector_emits_no_restart_hint(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Undetected connector -> only the ``not detected`` line, no restart hint.

    Even GUI-flagged connectors that fail ``detect()`` skip the restart
    hint because ``register()`` never ran.
    """
    missing_gui = _StubConnector(
        cli_name="zed",
        detected=False,
        config_file=fake_home / ".zed" / "config.json",
        is_gui=True,
    )

    installer.install(
        connectors=[missing_gui],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    out = capsys.readouterr().out
    restart_lines = [ln for ln in out.splitlines() if "Restart" in ln]
    assert restart_lines == []
    # The standard "not detected, skipping" line is still expected.
    assert "zed" in out
    assert "not detected" in out


def test_install_stdout_snapshot_three_connector_shapes(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exact stdout snapshot — CLI / GUI / undetected, in order.

    Three connectors cover the three relevant cases:

    * ``cli_app`` (CLI, detected): registers, no restart hint.
    * ``gui_app`` (GUI, detected): registers, restart hint follows.
    * ``missing`` (undetected): prints skip line only.

    The snapshot pins the precise wording. A reformat that drops the
    restart hint or moves it before ``register()`` will diff visibly.
    """
    cli_app = _StubConnector(
        cli_name="cli_app",
        detected=True,
        config_file=fake_home / ".cli_app" / "config.json",
        is_gui=False,
    )
    gui_app = _StubConnector(
        cli_name="gui_app",
        detected=True,
        config_file=fake_home / ".gui_app" / "config.json",
        is_gui=True,
    )
    missing = _StubConnector(
        cli_name="missing",
        detected=False,
        config_file=fake_home / ".missing" / "config.json",
        is_gui=True,  # Even GUI-flagged: no hint when undetected.
    )

    installer.install(
        connectors=[cli_app, gui_app, missing],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    out = capsys.readouterr().out
    # Pin only the three lines that this task introduces — surrounding
    # progress output (doctor preflight banner, DB-init, embedding model)
    # is covered by ``test_installer`` and is allowed to drift here.
    assert "  - cli_app: registering MCP server" in out
    assert "  - gui_app: registering MCP server" in out
    assert "  - missing: not detected, skipping" in out
    # Restart hint anchored to GUI app only.
    expected_hint = "    Restart gui_app to pick up changes"
    assert expected_hint in out
    # No restart line for the CLI app or the missing connector.
    assert "Restart cli_app" not in out
    assert "Restart missing" not in out

    # Order matters: the GUI restart hint must appear AFTER the GUI
    # register line and BEFORE the next connector's line (or any later
    # phase). Slicing the output by line index pins this.
    lines = out.splitlines()
    gui_register_idx = next(
        i for i, ln in enumerate(lines) if "gui_app: registering" in ln
    )
    gui_restart_idx = next(i for i, ln in enumerate(lines) if "Restart gui_app" in ln)
    missing_idx = next(i for i, ln in enumerate(lines) if "missing: not detected" in ln)
    assert gui_register_idx < gui_restart_idx < missing_idx


def test_gui_connector_with_custom_restart_hint(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Connectors may override :meth:`restart_hint` — the installer honours it.

    When a connector returns a tool-specific phrasing (e.g. "Cmd-Q then
    relaunch Cursor"), the installer must print that exact string rather
    than the generic ``Restart <name> ...`` fallback.
    """
    custom = "Cmd-Q Cursor, then relaunch from Spotlight."
    gui = _StubConnectorWithCustomHint(
        cli_name="cursor",
        detected=True,
        config_file=fake_home / ".cursor" / "config.json",
        is_gui=True,
        custom_hint=custom,
    )

    installer.install(
        connectors=[gui],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    out = capsys.readouterr().out
    assert custom in out
    # The generic fallback must not also be printed when the connector
    # supplies its own phrasing.
    assert "Restart cursor to pick up changes" not in out


def test_dry_run_emits_restart_hint_for_detected_gui(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` previews the restart hint for detected GUI connectors.

    Dry-run is a full preview: an LLM caller deciding whether to commit
    to the install should already see which apps will need a restart.
    """
    gui = _StubConnector(
        cli_name="windsurf",
        detected=True,
        config_file=fake_home / ".windsurf" / "config.json",
        is_gui=True,
    )
    cli_app = _StubConnector(
        cli_name="claude_code",
        detected=True,
        config_file=fake_home / ".claude" / "config.json",
        is_gui=False,
    )
    missing = _StubConnector(
        cli_name="absent",
        detected=False,
        config_file=fake_home / ".absent" / "config.json",
        is_gui=True,
    )

    installer.install(
        connectors=[gui, cli_app, missing],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "would" in out  # dry-run still emits diff entries
    restart_lines = [ln for ln in out.splitlines() if "Restart" in ln]
    assert len(restart_lines) == 1
    assert "windsurf" in restart_lines[0]
    assert "Restart claude_code" not in out
    assert "Restart absent" not in out
