"""Tests for ``crossmem install --dry-run``.

Dry-run mode shows what ``install`` *would* do without writing anything:

* No config-file mutation (no write, no ``.bak``).
* No ``~/.crossmem/knowledge.db`` creation.
* No embedder instantiation (no model download).

The stdout output is a human-readable diff line per detected connector
(``would add`` / ``would update`` / ``already present``) so an LLM (or
human) caller can preview the install before committing to it.

We use the same ``_RecordingConnector`` pattern as ``test_installer`` so
no real CLI configs or fastembed downloads are touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from crossmem import installer
from crossmem.connectors.base import CLIConnector


@dataclass
class _RecordingConnector(CLIConnector):
    """Mock connector that records ``register``/``unregister`` calls.

    Mirrors the helper in ``tests/test_installer.py``. ``register_calls``
    must remain empty after a dry-run, which is the central invariant of
    this test module.
    """

    cli_name: str
    detected: bool
    config_file: Path
    existing_entry: dict | None = None
    register_calls: list[str] = field(default_factory=list)
    unregister_calls: int = 0

    def name(self) -> str:
        return self.cli_name

    def detect(self) -> bool:
        return self.detected

    def config_path(self) -> Path:
        return self.config_file

    def current_entry(self) -> dict | None:
        return self.existing_entry

    def register(self, server_cmd: str) -> None:
        self.register_calls.append(server_cmd)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text("{}", encoding="utf-8")

    def unregister(self) -> None:
        self.unregister_calls += 1


class _ExplodingEmbedder:
    """Embedder stand-in that fails loudly if instantiated.

    Dry-run must never build the embedder; using this factory turns any
    accidental call into an immediate test failure.
    """

    def __init__(self) -> None:  # pragma: no cover - must not be called
        msg = "embedder must not be built during dry-run"
        raise AssertionError(msg)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch ``Path.home`` to a tmp dir so the installer writes there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_dry_run_writes_nothing(fake_home: Path) -> None:
    """Dry-run never touches the connector's config file or the DB.

    The installer must not:
    * call ``connector.register()`` (the side-effecting path),
    * create ``~/.crossmem/knowledge.db``,
    * leave any ``*.bak`` file behind for the detected connector.
    """
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )

    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    assert detected.register_calls == []
    # No config file created.
    assert not detected.config_file.exists()
    # No backup files anywhere under fake_home.
    backups = list(fake_home.rglob("*.bak"))
    backups_ts = list(fake_home.rglob("*.bak.*"))
    assert backups == []
    assert backups_ts == []
    # No SQLite DB.
    assert not (fake_home / ".crossmem" / "knowledge.db").exists()


def test_dry_run_outputs_one_line_per_detected_connector(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run prints exactly one diff entry per *detected* connector.

    Undetected connectors are still mentioned (consistent with the
    non-dry-run flow's "skipping" line) but each detected connector
    must produce one — and only one — diff entry tagged with its name.
    """
    alpha = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    beta = _RecordingConnector(
        cli_name="beta",
        detected=True,
        config_file=fake_home / ".beta" / "config.json",
    )
    gamma = _RecordingConnector(
        cli_name="gamma",
        detected=False,
        config_file=fake_home / ".gamma" / "config.json",
    )

    installer.install(
        connectors=[alpha, beta, gamma],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    # One diff entry per detected connector.
    alpha_entries = [ln for ln in out.splitlines() if "alpha" in ln and "would" in ln]
    beta_entries = [ln for ln in out.splitlines() if "beta" in ln and "would" in ln]
    assert len(alpha_entries) == 1
    assert len(beta_entries) == 1
    # The undetected connector must NOT get a diff entry.
    gamma_entries = [ln for ln in out.splitlines() if "gamma" in ln and "would" in ln]
    assert gamma_entries == []


def test_dry_run_reports_would_add_for_missing_entry(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the connector has no existing entry, status is ``would add``."""
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
        existing_entry=None,
    )

    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "would add" in out
    assert "alpha" in out


def test_dry_run_reports_already_present_when_entry_matches(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the current entry equals the planned snippet -> ``already present``."""
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    # Pre-fill the recording connector's reported current_entry with the
    # exact snippet the base class will produce for DEFAULT_SERVER_CMD.
    detected.existing_entry = detected.mcp_snippet(installer.DEFAULT_SERVER_CMD)

    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "already present" in out


def test_dry_run_reports_would_update_when_entry_differs(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When an entry exists but differs from the planned snippet -> ``would update``."""
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
        existing_entry={"command": "old-binary", "args": [], "env": {}},
    )

    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "would update" in out


def test_dry_run_skips_doctor_preflight(fake_home: Path) -> None:
    """Dry-run does not abort on doctor failures — it is a pure preview.

    A doctor failure during dry-run would force the user to fix the
    environment just to preview the diff, which defeats the point of
    a preview. Dry-run therefore short-circuits before the doctor
    check ever runs.
    """
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )
    doctor_calls: list[int] = []

    def exploding_doctor() -> list:
        doctor_calls.append(1)
        from crossmem.doctor import CheckResult

        return [
            CheckResult(name="never_run", status="fail", detail="should be skipped"),
        ]

    # No raise expected — dry-run must short-circuit the doctor preflight.
    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        doctor_factory=exploding_doctor,
        dry_run=True,
    )
    assert doctor_calls == []


def test_dry_run_returns_install_result_marked_as_dry(fake_home: Path) -> None:
    """The returned :class:`InstallResult` reflects that nothing was written.

    ``dry_run=True`` is exposed on the result so CLI callers can render
    a different summary line ("would install N CLIs" vs "installed N").
    """
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )

    result = installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    assert result.detected_clis == ["alpha"]
    assert result.dry_run is True
    # db_path still points at the would-be location, but the file must
    # NOT exist on disk after a dry-run.
    assert result.db_path == fake_home / ".crossmem" / "knowledge.db"
    assert not result.db_path.exists()


def test_dry_run_includes_planned_snippet_in_output(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The diff output references the planned MCP snippet (command/args).

    The snippet is what would land in the connector's config; surfacing
    it in dry-run lets the user verify ``python -m crossmem.server`` is
    the wired command before any file is touched.
    """
    detected = _RecordingConnector(
        cli_name="alpha",
        detected=True,
        config_file=fake_home / ".alpha" / "config.json",
    )

    installer.install(
        connectors=[detected],
        embedder_factory=_ExplodingEmbedder,
        dry_run=True,
    )

    out = capsys.readouterr().out
    # The default snippet contains "crossmem.server" — surfaces it so the
    # user sees what is going to be written.
    assert "crossmem.server" in out


def test_cli_install_dry_run_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``crossmem install --dry-run`` calls ``installer.install(dry_run=True)``.

    Wires the argparse flag through to the installer call and proves no
    side effects happen on the install path (we record the kwargs).
    """
    from crossmem import cli
    from crossmem.installer import InstallResult

    captured_kwargs: list[dict] = []

    def fake_install(*, dry_run: bool = False) -> InstallResult:
        captured_kwargs.append({"dry_run": dry_run})
        return InstallResult(
            detected_clis=["alpha"],
            db_path=Path("/tmp/knowledge.db"),
            embedding_model="mock-model",
            dry_run=dry_run,
        )

    monkeypatch.setattr(cli, "install", fake_install)
    exit_code = cli.main(["install", "--dry-run"])

    assert exit_code == 0
    assert captured_kwargs == [{"dry_run": True}]
    # CLI summary makes the dry-run explicit so the user is not confused
    # into thinking install succeeded.
    out = capsys.readouterr().out
    assert "dry" in out.lower() or "would" in out.lower()


def test_base_class_current_entry_default_reads_json_config(
    tmp_path: Path,
) -> None:
    """``CLIConnector.current_entry`` default reads JSON and returns the entry.

    The base class implementation supports the 10 connectors that share
    the ``mcpServers`` JSON layout; goose/continuedev override.
    """
    from crossmem.connectors.base import CLIConnector as Base

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "crossmem": {
                        "command": "python",
                        "args": ["-m", "crossmem.server"],
                        "env": {},
                    },
                    "other": {"command": "x", "args": [], "env": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    class _Stub(Base):
        def name(self) -> str:
            return "stub"

        def detect(self) -> bool:
            return True

        def config_path(self) -> Path:
            return cfg

        def register(self, server_cmd: str) -> None:  # pragma: no cover
            return None

        def unregister(self) -> None:  # pragma: no cover
            return None

    entry = _Stub().current_entry()
    assert entry == {
        "command": "python",
        "args": ["-m", "crossmem.server"],
        "env": {},
    }


def test_base_class_current_entry_returns_none_when_file_missing(
    tmp_path: Path,
) -> None:
    """Missing config -> ``None`` (treated as ``would add`` by the installer)."""
    from crossmem.connectors.base import CLIConnector as Base

    cfg = tmp_path / "missing.json"

    class _Stub(Base):
        def name(self) -> str:
            return "stub"

        def detect(self) -> bool:
            return False

        def config_path(self) -> Path:
            return cfg

        def register(self, server_cmd: str) -> None:  # pragma: no cover
            return None

        def unregister(self) -> None:  # pragma: no cover
            return None

    assert _Stub().current_entry() is None
