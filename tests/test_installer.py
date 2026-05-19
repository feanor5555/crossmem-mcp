"""Tests for ``crossmem install`` (installer.py).

The installer wires three things together:

1. CLI detection — loop over registered :class:`CLIConnector` classes and
   call their ``register()`` only when ``detect()`` is True.
2. DB initialization — create ``<home>/.crossmem/knowledge.db`` via the
   :class:`SQLiteBackend` so the schema migration runs.
3. Embedding model — instantiate the embedder so the on-disk model is
   downloaded with a progress note printed for the user.

Tests use a fake ``$HOME`` (``Path.home`` monkeypatched) and inject mock
connectors + a mock embedder factory so no network call or real
fastembed download happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from crossmem import installer
from crossmem.connectors.base import CLIConnector
from crossmem.doctor import CheckResult


@dataclass
class _RecordingConnector(CLIConnector):
    """Mock connector that records ``register``/``unregister`` calls."""

    cli_name: str
    detected: bool
    config_file: Path
    register_calls: list[str] = field(default_factory=list)
    unregister_calls: int = 0

    def name(self) -> str:
        return self.cli_name

    def detect(self) -> bool:
        return self.detected

    def config_path(self) -> Path:
        return self.config_file

    def register(self, server_cmd: str) -> None:
        self.register_calls.append(server_cmd)
        # Touch the config file so subsequent detect() reads can see it.
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text("{}", encoding="utf-8")

    def unregister(self) -> None:
        self.unregister_calls += 1


class _MockEmbedder:
    """Stand-in for :class:`EmbeddingService` — records that it was built."""

    instances: list[_MockEmbedder] = []  # noqa: RUF012 - simple class-level registry

    def __init__(self) -> None:
        self._model_name = "mock-model"
        type(self).instances.append(self)

    @property
    def model_name(self) -> str:
        return self._model_name


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch ``Path.home`` to a tmp dir so the installer writes there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_mock_embedder() -> None:
    _MockEmbedder.instances.clear()


def _make_connectors(
    fake_home: Path,
) -> tuple[_RecordingConnector, _RecordingConnector]:
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    missing = _RecordingConnector(
        cli_name="beta",
        detected=False,
        config_file=fake_home / ".beta" / "config.json",
    )
    return detected, missing


def test_install_runs_on_fake_home(fake_home: Path) -> None:
    """Detected connector gets register(), missing one is skipped, DB exists."""
    detected, missing = _make_connectors(fake_home)

    result = installer.install(
        connectors=[detected, missing],
        embedder_factory=_MockEmbedder,
    )

    assert detected.register_calls == [installer.DEFAULT_SERVER_CMD]
    assert missing.register_calls == []
    assert result.detected_clis == ["alpha"]
    assert result.db_path == fake_home / ".crossmem" / "knowledge.db"
    assert result.db_path.exists()
    assert result.embedding_model == "mock-model"
    assert len(_MockEmbedder.instances) == 1


def test_install_creates_backup_before_register(fake_home: Path) -> None:
    """Backup behaviour is delegated to the connector — verify the call.

    The actual file backup is the connector's responsibility (covered by
    its own tests). The installer just needs to invoke ``register`` once
    per detected CLI.
    """
    detected, _missing = _make_connectors(fake_home)
    detected.config_file.parent.mkdir(parents=True, exist_ok=True)
    detected.config_file.write_text('{"existing": true}', encoding="utf-8")

    installer.install(connectors=[detected], embedder_factory=_MockEmbedder)

    assert len(detected.register_calls) == 1


def test_install_idempotent(fake_home: Path) -> None:
    """Running install twice still calls register twice (connector dedupes)."""
    detected, _missing = _make_connectors(fake_home)

    installer.install(connectors=[detected], embedder_factory=_MockEmbedder)
    installer.install(connectors=[detected], embedder_factory=_MockEmbedder)

    # Installer always invokes register on detected connectors. The
    # connector itself is responsible for being idempotent (write the
    # same key twice -> identical config). We assert the call count so
    # a regression that skips re-registration is caught.
    assert len(detected.register_calls) == 2
    assert all(cmd == installer.DEFAULT_SERVER_CMD for cmd in detected.register_calls)
    # DB file should still exist (and not be wiped by re-install).
    assert (fake_home / ".crossmem" / "knowledge.db").exists()


def test_install_uses_default_connector_registry(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no ``connectors`` arg is given, fall back to ALL_CONNECTORS."""
    detected, missing = _make_connectors(fake_home)
    monkeypatch.setattr(
        installer,
        "ALL_CONNECTORS",
        [lambda d=detected: d, lambda m=missing: m],
    )

    result = installer.install(embedder_factory=_MockEmbedder)

    assert "alpha" in result.detected_clis
    assert "beta" not in result.detected_clis


def test_install_prints_progress(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Install prints a human-readable progress trace to stdout."""
    detected, _missing = _make_connectors(fake_home)

    installer.install(connectors=[detected], embedder_factory=_MockEmbedder)

    out = capsys.readouterr().out
    # Three rough phases: detect, register, embedding model. Wording can
    # change but each line should mention the relevant noun.
    assert "alpha" in out
    assert "knowledge.db" in out or "database" in out.lower()
    assert "embedding" in out.lower() or "model" in out.lower()


def test_install_default_embedder_factory(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``embedder_factory`` the installer builds an EmbeddingService.

    We monkeypatch the symbol the installer imports so no real model is
    downloaded; this proves the default wiring exists.
    """
    detected, _missing = _make_connectors(fake_home)
    built: list[object] = []

    def _factory() -> object:
        built.append(object())

        class _Stub:
            model_name = "stub"

        return _Stub()

    monkeypatch.setattr(installer, "EmbeddingService", _factory)

    result = installer.install(connectors=[detected])

    assert built  # the patched factory was used
    assert result.embedding_model == "stub"


def test_install_returns_dataclass(fake_home: Path) -> None:
    """:class:`InstallResult` exposes the three expected fields."""
    detected, _missing = _make_connectors(fake_home)

    result = installer.install(connectors=[detected], embedder_factory=_MockEmbedder)

    assert isinstance(result, installer.InstallResult)
    assert isinstance(result.detected_clis, list)
    assert isinstance(result.db_path, Path)
    assert isinstance(result.embedding_model, str)


def _ok_doctor() -> list[CheckResult]:
    """Doctor stub returning a single ``ok`` result — for tests that don't
    care about preflight behaviour and just want install to proceed."""
    return [CheckResult(name="stub", status="ok", detail="stub ok")]


def test_install_aborts_when_doctor_reports_fail(fake_home: Path) -> None:
    """A single ``fail`` result aborts install with :class:`InstallAbortedError`.

    The exception message must include the failed check's name and detail
    so the user can act (e.g. install a missing extra).
    """
    detected, _missing = _make_connectors(fake_home)

    def doctor_factory() -> list[CheckResult]:
        return [
            CheckResult(
                name="module_fastembed",
                status="fail",
                detail=(
                    "cannot import fastembed: not installed. "
                    "Install with: pip install crossmem[default]"
                ),
            ),
        ]

    with pytest.raises(installer.InstallAbortedError) as excinfo:
        installer.install(
            connectors=[detected],
            embedder_factory=_MockEmbedder,
            doctor_factory=doctor_factory,
        )

    msg = str(excinfo.value)
    assert "module_fastembed" in msg
    assert "pip install crossmem[default]" in msg
    # Connector must NOT have been touched, DB must NOT exist.
    assert detected.register_calls == []
    assert not (fake_home / ".crossmem" / "knowledge.db").exists()
    assert _MockEmbedder.instances == []


def test_install_proceeds_when_doctor_only_warns(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``warn`` results print but do not abort the install."""
    detected, _missing = _make_connectors(fake_home)

    def doctor_factory() -> list[CheckResult]:
        return [
            CheckResult(
                name="optional_chromadb",
                status="warn",
                detail="chromadb not installed (optional).",
            ),
            CheckResult(
                name="python_version",
                status="ok",
                detail="Python 3.12 is fine",
            ),
        ]

    result = installer.install(
        connectors=[detected],
        embedder_factory=_MockEmbedder,
        doctor_factory=doctor_factory,
    )

    out = capsys.readouterr().out
    assert "optional_chromadb" in out
    assert detected.register_calls == [installer.DEFAULT_SERVER_CMD]
    assert result.detected_clis == ["alpha"]


def test_install_proceeds_when_doctor_all_ok(fake_home: Path) -> None:
    """All ``ok`` -> install proceeds normally, no special output needed."""
    detected, _missing = _make_connectors(fake_home)

    result = installer.install(
        connectors=[detected],
        embedder_factory=_MockEmbedder,
        doctor_factory=_ok_doctor,
    )

    assert result.detected_clis == ["alpha"]
    assert result.db_path.exists()
    assert len(_MockEmbedder.instances) == 1


def test_doctor_check_runs_before_register(fake_home: Path) -> None:
    """On ``fail``: connector.register, DB-init, embedder are all skipped.

    Guarantees doctor really is a *preflight* — no side effects past it
    when checks fail.
    """
    detected, _missing = _make_connectors(fake_home)

    def doctor_factory() -> list[CheckResult]:
        return [
            CheckResult(name="db_dir_writable", status="fail", detail="oops"),
        ]

    with pytest.raises(installer.InstallAbortedError):
        installer.install(
            connectors=[detected],
            embedder_factory=_MockEmbedder,
            doctor_factory=doctor_factory,
        )

    assert detected.register_calls == []
    assert not (fake_home / ".crossmem").exists()
    assert _MockEmbedder.instances == []


def test_install_default_doctor_factory_is_run_checks(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``doctor_factory`` the installer wires up doctor.run_checks.

    Patching the symbol the installer imports proves the default wiring.
    """
    detected, _missing = _make_connectors(fake_home)
    calls: list[int] = []

    def _stub_run_checks() -> list[CheckResult]:
        calls.append(1)
        return [CheckResult(name="stub", status="ok", detail="ok")]

    monkeypatch.setattr(installer, "run_checks", _stub_run_checks)

    installer.install(connectors=[detected], embedder_factory=_MockEmbedder)

    assert calls == [1]


def test_install_aborts_on_any_fail_among_oks_and_warns(fake_home: Path) -> None:
    """A single ``fail`` aborts even if other checks are ok/warn."""
    detected, _missing = _make_connectors(fake_home)

    def doctor_factory() -> list[CheckResult]:
        return [
            CheckResult(name="python_version", status="ok", detail="3.12"),
            CheckResult(name="optional_chromadb", status="warn", detail="missing"),
            CheckResult(
                name="module_sqlite_vec",
                status="fail",
                detail="cannot import sqlite_vec",
            ),
        ]

    with pytest.raises(installer.InstallAbortedError) as excinfo:
        installer.install(
            connectors=[detected],
            embedder_factory=_MockEmbedder,
            doctor_factory=doctor_factory,
        )

    msg = str(excinfo.value)
    assert "module_sqlite_vec" in msg
    # Other check names should not pollute the failure summary.
    assert "python_version" not in msg


def test_install_aborted_exposes_failed_checks(fake_home: Path) -> None:
    """:class:`InstallAbortedError` carries the failed check list for callers."""
    detected, _missing = _make_connectors(fake_home)
    failures = [
        CheckResult(name="module_fastmcp", status="fail", detail="missing"),
        CheckResult(name="db_dir_writable", status="fail", detail="readonly"),
    ]

    def doctor_factory() -> list[CheckResult]:
        return [
            CheckResult(name="python_version", status="ok", detail="3.12"),
            *failures,
        ]

    with pytest.raises(installer.InstallAbortedError) as excinfo:
        installer.install(
            connectors=[detected],
            embedder_factory=_MockEmbedder,
            doctor_factory=doctor_factory,
        )

    assert list(excinfo.value.failed_checks) == failures


def test_default_server_cmd_is_python_module() -> None:
    """server_cmd default points at the in-tree MCP server module.

    ``crossmem`` (the console-script) is not yet a separate entry point at
    install time on every platform, but ``python -m crossmem.server`` is
    always callable once the wheel is on PYTHONPATH.
    """
    assert installer.DEFAULT_SERVER_CMD.endswith("crossmem.server") or (
        installer.DEFAULT_SERVER_CMD == "crossmem"
    )


def test_doctor_install_doc_connectors_delegates_to_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``doctor._install_doc_connectors`` funnels through the canonical factory.

    Patching ``installer.instantiate_connectors`` must change what doctor
    sees — proves there is no parallel instantiation path that bypasses
    the canonical factory.
    """
    from crossmem import doctor

    sentinel: list[object] = [object()]
    monkeypatch.setattr(installer, "instantiate_connectors", lambda: sentinel)

    assert doctor._install_doc_connectors() is sentinel


def test_cli_status_connectors_factory_delegates_to_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cli._status_connectors_factory`` funnels through the canonical factory."""
    from crossmem import cli

    sentinel: list[object] = [object()]
    monkeypatch.setattr(installer, "instantiate_connectors", lambda: sentinel)

    assert cli._status_connectors_factory() is sentinel


def test_uninstaller_materialise_connectors_delegates_to_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``uninstaller._materialise_connectors(None)`` uses the canonical factory."""
    from crossmem import uninstaller

    sentinel: list[object] = [object()]
    monkeypatch.setattr(installer, "instantiate_connectors", lambda: sentinel)

    assert uninstaller._materialise_connectors(None) is sentinel


def test_instantiate_connectors_walks_all_connectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``instantiate_connectors()`` calls every factory in ``ALL_CONNECTORS``.

    Canonical factory introduced by task 26.20: the three other sites
    (``doctor._install_doc_connectors``, ``cli._status_connectors_factory``,
    ``uninstaller._materialise_connectors`` no-arg branch) all funnel
    through this single function so the connector set never drifts.
    """
    calls: list[str] = []

    def _factory_a() -> object:
        calls.append("a")
        return object()

    def _factory_b() -> object:
        calls.append("b")
        return object()

    monkeypatch.setattr(installer, "ALL_CONNECTORS", [_factory_a, _factory_b])

    result = installer.instantiate_connectors()

    assert calls == ["a", "b"]
    assert len(result) == 2


def test_all_connectors_registers_every_shipped_connector() -> None:
    """``ALL_CONNECTORS`` must list every shipped :class:`CLIConnector` class.

    The project ships 12 connectors (see CLAUDE.md "Unterstuetzte CLIs"). If
    a connector class exists on disk but is missing from ``ALL_CONNECTORS``,
    ``crossmem install`` silently skips it — the user runs install but the
    MCP entry is never written for that CLI. This test guards against that
    regression by asserting the full set is registered.
    """
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

    expected = {
        ClaudeCodeConnector,
        CursorConnector,
        ClineConnector,
        GooseConnector,
        ContinueDevConnector,
        PiConnector,
        OpenCodeConnector,
        KiloCodeConnector,
        GeminiConnector,
        WindsurfConnector,
        AmazonQConnector,
        ZedConnector,
    }
    assert set(installer.ALL_CONNECTORS) == expected
    assert len(installer.ALL_CONNECTORS) == 12
