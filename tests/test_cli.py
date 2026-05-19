"""Tests for the ``crossmem`` CLI entry point.

Covers the dispatcher, the doctor/configure subcommands, the install/export/
import subcommands wired up here, and the default behaviour of starting the
MCP server when invoked with no arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossmem import cli
from crossmem.doctor import CheckResult
from crossmem.installer import InstallAbortedError, InstallResult
from crossmem.uninstaller import UninstallResult
from crossmem.uninstaller import UnknownConnectorError as UninstallUnknownError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _all_ok() -> list[CheckResult]:
    return [
        CheckResult(name="python_version", status="ok", detail="Python 3.12.5"),
        CheckResult(name="module_fastembed", status="ok", detail="fastembed ok"),
        CheckResult(
            name="optional_chromadb",
            status="warn",
            detail=(
                "chromadb not installed (optional). "
                "Install with: pip install crossmem[chroma]"
            ),
        ),
    ]


def _has_fail() -> list[CheckResult]:
    return [
        CheckResult(name="python_version", status="ok", detail="Python 3.12.5"),
        CheckResult(
            name="db_dir_writable",
            status="fail",
            detail="cannot create /home/x/.crossmem: permission denied",
        ),
    ]


# ---------------------------------------------------------------------------
# Dispatcher behavior
# ---------------------------------------------------------------------------


def test_main_no_args_starts_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling ``main()`` with no args starts the MCP server.

    Direct invocation of ``crossmem`` (no subcommand) is the supported way
    for MCP-capable CLIs to spawn the server over stdio. Tests stub
    ``server.main`` to avoid actually running FastMCP.
    """
    called: list[bool] = []

    def fake_server_main() -> None:
        called.append(True)

    monkeypatch.setattr(cli, "server_main", fake_server_main)
    exit_code = cli.main([])
    assert exit_code == 0
    assert called == [True]


def test_help_flag_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` prints the help text and exits with code 0."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "crossmem" in captured.out
    # Help should mention every subcommand.
    for sub in ("doctor", "configure", "install", "uninstall", "export", "import"):
        assert sub in captured.out


def test_unknown_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown subcommand should make argparse exit non-zero."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["bogus-subcommand"])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# doctor: human-readable output
# ---------------------------------------------------------------------------


def test_doctor_exit_zero_when_all_ok(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fail results -> exit code 0 (warns are tolerated)."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    exit_code = cli.main(["doctor"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[ok]" in captured.out
    assert "[warn]" in captured.out
    assert "python_version" in captured.out


def test_doctor_exit_one_when_any_fail(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single fail result must produce exit code 1."""
    monkeypatch.setattr(cli, "run_checks", _has_fail)
    exit_code = cli.main(["doctor"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[fail]" in captured.out
    assert "db_dir_writable" in captured.out


def test_doctor_prints_summary_line(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final line summarises counts in the form ``N ok, M warn, K fail``."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    cli.main(["doctor"])
    captured = capsys.readouterr()
    # _all_ok: 2 ok, 1 warn, 0 fail
    assert "2 ok, 1 warn, 0 fail" in captured.out


def test_doctor_marker_format_per_line(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each result is printed on its own line with ``name: detail`` after the marker."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    # First non-summary line should be the python_version result.
    assert any(
        line.startswith("[ok]") and "python_version" in line and "Python 3.12.5" in line
        for line in lines
    )


# ---------------------------------------------------------------------------
# doctor: --json
# ---------------------------------------------------------------------------


def test_doctor_json_output_is_valid_json(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    exit_code = cli.main(["doctor", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    # Stable top-level shape: object with version + checks + summary.
    assert isinstance(payload, dict)
    assert payload["version"] == "1"
    assert payload["checks"][0] == {
        "name": "python_version",
        "status": "ok",
        "detail": "Python 3.12.5",
    }
    assert payload["summary"] == {"ok": 2, "warn": 1, "fail": 0, "total": 3}
    # No human-readable summary line should be appended in JSON mode.
    assert "2 ok, 1 warn" not in captured.out
    # No marker brackets either.
    assert "[ok]" not in captured.out


def test_doctor_json_no_color_codes(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON output never contains ANSI escape codes, even on a TTY."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli.main(["doctor", "--json"])
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out


def test_doctor_json_exit_code_one_on_fail(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_checks", _has_fail)
    exit_code = cli.main(["doctor", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out)
    assert any(item["status"] == "fail" for item in payload["checks"])
    assert payload["summary"]["fail"] >= 1


# ---------------------------------------------------------------------------
# doctor: color handling
# ---------------------------------------------------------------------------


def test_doctor_tty_emits_ansi(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a TTY, output contains ANSI color codes for ok/warn/fail markers."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "\x1b[" in out


def test_doctor_non_tty_has_no_ansi(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a TTY, output is plain text (no ANSI codes)."""
    monkeypatch.setattr(cli, "run_checks", _all_ok)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_doctor_tty_uses_distinct_colors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ok/warn/fail use different ANSI codes when colorized."""
    monkeypatch.setattr(
        cli,
        "run_checks",
        lambda: [
            CheckResult(name="a", status="ok", detail="d"),
            CheckResult(name="b", status="warn", detail="d"),
            CheckResult(name="c", status="fail", detail="d"),
        ],
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    # Green, yellow, red — standard ANSI 32/33/31.
    assert "\x1b[32m" in out  # green for ok
    assert "\x1b[33m" in out  # yellow for warn
    assert "\x1b[31m" in out  # red for fail


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def test_install_subcommand_invokes_installer(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``crossmem install`` calls ``installer.install`` and prints the summary."""
    db_path = tmp_path / "knowledge.db"
    result = InstallResult(
        detected_clis=["claude-code", "cursor"],
        db_path=db_path,
        embedding_model="mock-model",
    )
    captured_calls: list[bool] = []

    def fake_install(*, dry_run: bool = False) -> InstallResult:
        captured_calls.append(dry_run)
        return result

    monkeypatch.setattr(cli, "install", fake_install)
    exit_code = cli.main(["install"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured_calls == [False]
    assert "claude-code" in out
    assert "cursor" in out
    assert "mock-model" in out
    assert str(db_path) in out


def test_install_subcommand_no_clis(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty detected list still succeeds and prints a clear summary."""
    result = InstallResult(
        detected_clis=[],
        db_path=tmp_path / "knowledge.db",
        embedding_model="mock-model",
    )
    monkeypatch.setattr(cli, "install", lambda *, dry_run=False: result)
    exit_code = cli.main(["install"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "0" in out  # count surfaces in the summary


def test_install_subcommand_aborted(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A doctor failure surfaces as non-zero exit + readable error on stderr."""
    failed = [
        CheckResult(name="db_dir_writable", status="fail", detail="permission denied"),
    ]

    def fake_install(*, dry_run: bool = False) -> InstallResult:
        raise InstallAbortedError(failed)

    monkeypatch.setattr(cli, "install", fake_install)
    exit_code = cli.main(["install"])
    captured = capsys.readouterr()
    assert exit_code == 1
    # The error message must reach the user — argparse-level errors go to
    # stderr, so we accept it on either stream.
    combined = captured.out + captured.err
    assert "db_dir_writable" in combined
    assert "permission denied" in combined


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_subcommand_invokes_uninstaller(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``crossmem uninstall`` calls ``uninstaller.uninstall`` and prints summary."""
    result = UninstallResult(
        unregistered_clis=["claude_code", "cursor"],
        skipped_clis=["zed"],
        purged_home=False,
        home_path=tmp_path / ".crossmem",
    )
    captured_kwargs: list[dict[str, object]] = []

    def fake_uninstall(
        *, cli: str | None = None, purge: bool = False, confirm: bool = False
    ) -> UninstallResult:
        captured_kwargs.append({"cli": cli, "purge": purge, "confirm": confirm})
        return result

    monkeypatch.setattr(cli, "uninstall", fake_uninstall)
    exit_code = cli.main(["uninstall"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured_kwargs == [{"cli": None, "purge": False, "confirm": False}]
    assert "claude_code" in out
    assert "cursor" in out


def test_uninstall_subcommand_cli_filter(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--cli NAME`` is forwarded to ``uninstaller.uninstall``."""
    result = UninstallResult(
        unregistered_clis=["claude_code"],
        skipped_clis=[],
        purged_home=False,
        home_path=tmp_path / ".crossmem",
    )
    captured_kwargs: list[dict[str, object]] = []

    def fake_uninstall(
        *, cli: str | None = None, purge: bool = False, confirm: bool = False
    ) -> UninstallResult:
        captured_kwargs.append({"cli": cli, "purge": purge, "confirm": confirm})
        return result

    monkeypatch.setattr(cli, "uninstall", fake_uninstall)
    exit_code = cli.main(["uninstall", "--cli", "claude_code"])
    assert exit_code == 0
    assert captured_kwargs == [{"cli": "claude_code", "purge": False, "confirm": False}]
    out = capsys.readouterr().out
    assert "claude_code" in out


def test_uninstall_subcommand_purge_yes_forwarded(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--purge --yes`` forwards both flags and the summary mentions the path."""
    crossmem_dir = tmp_path / ".crossmem"
    result = UninstallResult(
        unregistered_clis=["alpha"],
        skipped_clis=[],
        purged_home=True,
        home_path=crossmem_dir,
    )
    captured_kwargs: list[dict[str, object]] = []

    def fake_uninstall(
        *, cli: str | None = None, purge: bool = False, confirm: bool = False
    ) -> UninstallResult:
        captured_kwargs.append({"cli": cli, "purge": purge, "confirm": confirm})
        return result

    monkeypatch.setattr(cli, "uninstall", fake_uninstall)
    exit_code = cli.main(["uninstall", "--purge", "--yes"])
    assert exit_code == 0
    assert captured_kwargs == [
        {"cli": None, "purge": True, "confirm": True},
    ]
    out = capsys.readouterr().out
    assert "Purged" in out
    assert str(crossmem_dir) in out


def test_uninstall_subcommand_purge_without_yes_hints(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--purge`` without ``--yes`` exits 0 and surfaces the missing flag."""
    crossmem_dir = tmp_path / ".crossmem"
    result = UninstallResult(
        unregistered_clis=["alpha"],
        skipped_clis=[],
        purged_home=False,
        home_path=crossmem_dir,
    )

    def fake_uninstall(
        *, cli: str | None = None, purge: bool = False, confirm: bool = False
    ) -> UninstallResult:
        return result

    monkeypatch.setattr(cli, "uninstall", fake_uninstall)
    exit_code = cli.main(["uninstall", "--purge"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "--yes" in out


def test_uninstall_subcommand_unknown_cli_exits_two(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown ``--cli`` -> exit 2 + helpful stderr listing known CLIs."""

    def fake_uninstall(
        *, cli: str | None = None, purge: bool = False, confirm: bool = False
    ) -> UninstallResult:
        raise UninstallUnknownError("nope", ["alpha", "beta"])

    monkeypatch.setattr(cli, "uninstall", fake_uninstall)
    exit_code = cli.main(["uninstall", "--cli", "nope"])
    captured = capsys.readouterr()
    assert exit_code == 2
    combined = captured.out + captured.err
    assert "nope" in combined
    assert "alpha" in combined
    assert "beta" in combined


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------


class _StubStore:
    """Minimal stand-in for KnowledgeStore — tracks calls without touching IO."""

    def __init__(self) -> None:
        self.export_calls: list[tuple[Path, str]] = []
        self.import_calls: list[Path] = []
        self.export_return = 7
        self.import_return = 3

    def export(self, path: Path, format: str = "zip") -> int:  # noqa: A002
        self.export_calls.append((path, format))
        return self.export_return

    def import_data(self, path: Path) -> int:
        self.import_calls.append(path)
        return self.import_return


def test_export_subcommand_writes_zip_by_default(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``crossmem export --path P`` defaults to ZIP and prints the count."""
    stub = _StubStore()
    monkeypatch.setattr(cli, "build_default_store", lambda: stub)
    target = tmp_path / "out.zip"
    exit_code = cli.main(["export", "--path", str(target)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert stub.export_calls == [(target, "zip")]
    assert "7" in out  # documents exported
    assert str(target) in out


def test_export_subcommand_jsonl_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--format jsonl`` is forwarded verbatim."""
    stub = _StubStore()
    monkeypatch.setattr(cli, "build_default_store", lambda: stub)
    target = tmp_path / "out.jsonl"
    exit_code = cli.main(["export", "--path", str(target), "--format", "jsonl"])
    assert exit_code == 0
    assert stub.export_calls == [(target, "jsonl")]


def test_export_subcommand_invalid_format(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Unknown format -> argparse exits non-zero before any store call."""
    target = tmp_path / "out.txt"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["export", "--path", str(target), "--format", "txt"])
    assert excinfo.value.code != 0


def test_import_subcommand_reads_file(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``crossmem import --path P`` forwards to ``import_data`` + prints count."""
    stub = _StubStore()
    monkeypatch.setattr(cli, "build_default_store", lambda: stub)
    src = tmp_path / "in.zip"
    exit_code = cli.main(["import", "--path", str(src)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert stub.import_calls == [src]
    assert "3" in out
    assert str(src) in out


def test_import_subcommand_value_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ``ValueError`` from import_data surfaces with exit code 1, not a traceback."""

    class _BoomStore:
        def import_data(self, path: Path) -> int:
            raise ValueError("EOF sha256 does not match content lines")

    monkeypatch.setattr(cli, "build_default_store", lambda: _BoomStore())
    exit_code = cli.main(["import", "--path", str(tmp_path / "broken.jsonl")])
    captured = capsys.readouterr()
    assert exit_code == 1
    combined = captured.out + captured.err
    assert "EOF sha256" in combined


# ---------------------------------------------------------------------------
# build_default_store
# ---------------------------------------------------------------------------


def test_build_default_store_returns_knowledge_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The default store factory wires SQLite + EmbeddingService into KnowledgeStore."""
    # Redirect DB path so the test doesn't touch ~/.crossmem.
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(tmp_path / "knowledge.db"))
    # Redirect $HOME so load_config() returns the default sqlite config and
    # does not pick up a real ~/.crossmem/config.toml that might point at an
    # optional backend.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Avoid downloading the fastembed model in the test environment.
    class _StubEmbedder:
        model_name = "mock"

        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 384

        def embed_passage(self, text: str) -> list[float]:
            return [0.0] * 384

        def embed_passage_batch(
            self, texts: list[str], batch_size: int = 32
        ) -> list[list[float]]:
            del batch_size
            return [[0.0] * 384 for _ in texts]

    monkeypatch.setattr(cli, "EmbeddingService", _StubEmbedder)
    store = cli.build_default_store()
    # We don't import KnowledgeStore here to keep the test focused on the
    # contract — it must expose export + import_data.
    assert hasattr(store, "export")
    assert hasattr(store, "import_data")


# ---------------------------------------------------------------------------
# trash subcommand (list / restore / empty)
# ---------------------------------------------------------------------------


def test_trash_list_prints_rows_from_cleanup_module(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crossmem trash list`` prints one row per entry returned by list_trash."""
    from crossmem.cleanup import TrashEntry

    entries = [
        TrashEntry(
            doc_id="abc123",
            deleted_at="2026-01-01T00:00:00+00:00",
            source_url="https://example.com/a",
            title="A title",
        ),
        TrashEntry(
            doc_id="def456",
            deleted_at="2026-01-02T00:00:00+00:00",
            source_url="https://example.com/b",
            title="B title",
        ),
    ]
    monkeypatch.setattr(cli, "list_trash", lambda: entries)

    exit_code = cli.main(["trash", "list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    # Every entry's id/url/title appears in the output.
    for e in entries:
        assert e.doc_id in out
        assert e.source_url in out
        assert e.title in out


def test_trash_list_empty_prints_zero_count(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty trash -> exit 0 + a clear "0" or "empty" hint in stdout."""
    monkeypatch.setattr(cli, "list_trash", lambda: [])
    exit_code = cli.main(["trash", "list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    # The wording is implementation-defined; tests only require a clear signal.
    assert "0" in out or "empty" in out.lower()


def test_trash_restore_known_id(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``trash restore --id X`` calls restore_from_trash and prints the id."""
    stub_store = object()
    monkeypatch.setattr(cli, "build_default_store", lambda: stub_store)

    captured: dict[str, object] = {}

    def fake_restore(store, doc_id):
        captured["store"] = store
        captured["doc_id"] = doc_id

        class _Restored:
            id = doc_id
            content = "x"

        return _Restored()

    monkeypatch.setattr(cli, "restore_from_trash", fake_restore)

    exit_code = cli.main(["trash", "restore", "--id", "abc123"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured == {"store": stub_store, "doc_id": "abc123"}
    assert "abc123" in out


def test_trash_restore_unknown_id_prints_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown id -> exit 1, clear error message on stderr."""
    monkeypatch.setattr(cli, "build_default_store", lambda: object())

    def boom(store, doc_id):
        raise ValueError(f"doc_id not found in trash: {doc_id!r}")

    monkeypatch.setattr(cli, "restore_from_trash", boom)

    exit_code = cli.main(["trash", "restore", "--id", "missing"])
    captured = capsys.readouterr()
    assert exit_code == 1
    combined = captured.out + captured.err
    assert "missing" in combined
    assert "not found" in combined.lower() or "trash" in combined.lower()


def test_trash_empty_default_ttl_30(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``trash empty`` defaults ttl_days=30 and prints the removed count."""
    captured: dict[str, object] = {}

    def fake_empty(trash_path=None, *, ttl_days=30):
        captured["trash_path"] = trash_path
        captured["ttl_days"] = ttl_days
        return 4

    monkeypatch.setattr(cli, "empty_trash", fake_empty)

    exit_code = cli.main(["trash", "empty"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured["ttl_days"] == 30
    assert "4" in out


def test_trash_empty_explicit_ttl(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``trash empty --ttl-days 0`` forwards ttl_days=0 (DSGVO purge)."""
    captured: dict[str, int] = {}

    def fake_empty(trash_path=None, *, ttl_days=30):
        captured["ttl_days"] = ttl_days
        return 7

    monkeypatch.setattr(cli, "empty_trash", fake_empty)

    exit_code = cli.main(["trash", "empty", "--ttl-days", "0"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured["ttl_days"] == 0
    assert "7" in out


def test_trash_no_subcommand_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crossmem trash`` with no sub-subcommand prints help and exits non-zero."""
    exit_code = cli.main(["trash"])
    captured = capsys.readouterr()
    assert exit_code == 2
    combined = captured.out + captured.err
    assert "list" in combined
    assert "restore" in combined
    assert "empty" in combined
