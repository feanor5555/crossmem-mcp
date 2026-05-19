"""Tests for ``crossmem configure`` — backend switching.

The configure subcommand writes the active backend choice (and optional
``url`` / ``api_key``) to ``~/.crossmem/config.toml``. When switching between
backends, the user is offered a migration via export/import; the actual
migration is mocked in these tests.

This module also exercises the runtime wiring: ``build_backend`` dispatches
on ``[backend].name`` and is called by both ``server.main()`` and
``cli.build_default_store()`` so the config file actually drives behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tomllib

from crossmem import cli, configure

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to a tmp dir so configure writes there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# write + read roundtrip
# ---------------------------------------------------------------------------


def test_configure_writes_config_toml(fake_home: Path) -> None:
    """``configure`` writes ~/.crossmem/config.toml with the backend choice."""
    result = configure.configure(backend="chroma", url="http://host:8000")
    config_path = fake_home / ".crossmem" / "config.toml"
    assert config_path.exists()
    assert result.path == config_path
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert data["backend"]["name"] == "chroma"
    assert data["backend"]["url"] == "http://host:8000"


def test_configure_is_readable_after_reload(fake_home: Path) -> None:
    """Written config is reloadable through :func:`load_config`."""
    configure.configure(backend="qdrant", url="https://q.example", api_key="xyz")
    loaded = configure.load_config()
    assert loaded["backend"]["name"] == "qdrant"
    assert loaded["backend"]["url"] == "https://q.example"
    assert loaded["backend"]["api_key"] == "xyz"


def test_load_config_missing_returns_default(fake_home: Path) -> None:
    """Without a config file, the default sqlite backend is returned."""
    loaded = configure.load_config()
    assert loaded["backend"]["name"] == "sqlite"


def test_configure_omits_none_fields(fake_home: Path) -> None:
    """Optional fields left as ``None`` are not written to disk."""
    configure.configure(backend="chroma")
    data = tomllib.loads(
        (fake_home / ".crossmem" / "config.toml").read_text(encoding="utf-8")
    )
    assert "url" not in data["backend"]
    assert "api_key" not in data["backend"]


# ---------------------------------------------------------------------------
# Backend switching
# ---------------------------------------------------------------------------


def test_switch_from_default_sqlite_to_chroma(fake_home: Path) -> None:
    """Default backend is sqlite; switching to chroma records the new choice."""
    assert configure.load_config()["backend"]["name"] == "sqlite"
    configure.configure(backend="chroma", url="http://h:8000")
    assert configure.load_config()["backend"]["name"] == "chroma"


def test_switch_chroma_to_qdrant_updates_file(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    configure.configure(backend="chroma", url="http://h:8000")
    configure.configure(backend="qdrant", url="https://q", api_key="k")
    loaded = configure.load_config()
    assert loaded["backend"]["name"] == "qdrant"
    assert loaded["backend"]["url"] == "https://q"
    assert loaded["backend"]["api_key"] == "k"
    # Old chroma url must be gone.
    assert loaded["backend"].get("url") != "http://h:8000"


def test_invalid_backend_rejected(fake_home: Path) -> None:
    """Only sqlite/chroma/qdrant are accepted."""
    with pytest.raises(ValueError, match="unsupported backend"):
        configure.configure(backend="bogus")


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_no_migration_offered_on_first_configure(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No existing config -> nothing to migrate, never call the migrator."""
    called: list[tuple[str, str]] = []

    def fake_migrate(src: str, dst: str) -> None:
        called.append((src, dst))

    monkeypatch.setattr(configure, "_run_migration", fake_migrate)
    configure.configure(backend="chroma")
    assert called == []


def test_migration_skipped_when_user_says_no(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User answering 'n' at the prompt skips migration."""
    configure.configure(backend="sqlite")
    called: list[tuple[str, str]] = []
    monkeypatch.setattr(
        configure,
        "_run_migration",
        lambda src, dst: called.append((src, dst)),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    configure.configure(backend="chroma", url="http://h")
    assert called == []


def test_migration_runs_when_user_says_yes(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User answering 'y' triggers the (mocked) migration helper."""
    configure.configure(backend="sqlite")
    called: list[tuple[str, str]] = []
    monkeypatch.setattr(
        configure,
        "_run_migration",
        lambda src, dst: called.append((src, dst)),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    configure.configure(backend="chroma", url="http://h")
    assert called == [("sqlite", "chroma")]


def test_migration_flag_skips_prompt(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--migrate`` (=migrate=True) bypasses the interactive prompt."""
    configure.configure(backend="sqlite")
    called: list[tuple[str, str]] = []
    monkeypatch.setattr(
        configure,
        "_run_migration",
        lambda src, dst: called.append((src, dst)),
    )

    def boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called with --migrate")

    monkeypatch.setattr("builtins.input", boom)
    configure.configure(backend="chroma", url="http://h", migrate=True)
    assert called == [("sqlite", "chroma")]


def test_same_backend_no_migration_prompt(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running configure with the same backend never prompts for migration."""
    configure.configure(backend="chroma", url="http://h")
    called: list[tuple[str, str]] = []
    monkeypatch.setattr(
        configure,
        "_run_migration",
        lambda src, dst: called.append((src, dst)),
    )

    def boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called for same-backend reconfigure")

    monkeypatch.setattr("builtins.input", boom)
    configure.configure(backend="chroma", url="http://other")
    assert called == []


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_configure_subcommand(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crossmem configure --backend chroma --url ...`` works end-to-end."""
    monkeypatch.setattr(configure, "_run_migration", lambda *_a: None)
    exit_code = cli.main(["configure", "--backend", "chroma", "--url", "http://h:8000"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "chroma" in out
    loaded = configure.load_config()
    assert loaded["backend"]["name"] == "chroma"


def test_cli_configure_rejects_unknown_backend(
    fake_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Argparse rejects backend values outside the choices list."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["configure", "--backend", "bogus"])
    assert exc.value.code != 0


def test_cli_configure_passes_migrate_flag(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--migrate`` reaches the configure() function."""
    configure.configure(backend="sqlite")
    seen: list[bool] = []

    def fake_configure(
        backend: str,
        url: str | None = None,
        api_key: str | None = None,
        migrate: bool = False,
    ) -> configure.ConfigureResult:
        seen.append(migrate)
        return configure.ConfigureResult(
            path=fake_home / ".crossmem" / "config.toml",
            previous_backend="sqlite",
            new_backend=backend,
            migrated=migrate,
        )

    monkeypatch.setattr(cli, "configure_backend", fake_configure)
    exit_code = cli.main(
        ["configure", "--backend", "chroma", "--url", "http://h", "--migrate"]
    )
    assert exit_code == 0
    assert seen == [True]


# ---------------------------------------------------------------------------
# Sanity: writer is tomllib-readable
# ---------------------------------------------------------------------------


def test_written_toml_parses_with_tomllib(fake_home: Path) -> None:
    """Manual TOML serialization must roundtrip through tomllib."""
    configure.configure(
        backend="qdrant",
        url="https://example.invalid:6333",
        api_key='quote " and = signs',
    )
    raw = (fake_home / ".crossmem" / "config.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(raw)
    assert parsed["backend"]["api_key"] == 'quote " and = signs'


def test_module_runs_via_python_m(fake_home: Path) -> None:
    """``python -m crossmem.configure`` is not a user-facing entry point.

    We don't expose ``__main__``; this test simply asserts the configure
    module is importable as a script-style module (covers ``__all__`` shape).
    """
    assert "configure" in configure.__all__
    assert "load_config" in configure.__all__
    assert sys.modules["crossmem.configure"] is configure


# ---------------------------------------------------------------------------
# _dump_toml hardening — only string values are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        1,
        1.5,
        True,
        None,
        ["a"],
        {"x": "y"},
        b"bytes",
    ],
)
def test_dump_toml_rejects_non_string_value(bad_value: object) -> None:
    """``_dump_toml`` raises ``TypeError`` for any non-string field value.

    The writer is intentionally narrow (flat ``[section]`` with string
    fields). Silent ``str()``-coercion would emit syntactically valid but
    semantically wrong TOML (e.g. ``key = "True"`` instead of ``key = true``);
    a typed error tells the caller to switch types up the stack.
    """
    with pytest.raises(TypeError, match="only string values supported"):
        configure._dump_toml({"backend": {"name": bad_value}})  # type: ignore[dict-item]


def test_dump_toml_rejects_non_string_key() -> None:
    """Non-string keys are rejected the same way as non-string values."""
    with pytest.raises(TypeError, match="only string"):
        configure._dump_toml({"backend": {123: "x"}})  # type: ignore[dict-item]


def test_dump_toml_rejects_non_string_section_name() -> None:
    """Non-string section names are rejected the same way."""
    with pytest.raises(TypeError, match="only string"):
        configure._dump_toml({42: {"name": "sqlite"}})  # type: ignore[dict-item]


def test_dump_toml_accepts_string_values() -> None:
    """Happy path stays intact: string values serialize as before."""
    rendered = configure._dump_toml({"backend": {"name": "sqlite", "url": "http://h"}})
    assert "[backend]" in rendered
    assert 'name = "sqlite"' in rendered
    assert 'url = "http://h"' in rendered


# ---------------------------------------------------------------------------
# Real migration semantics — _run_migration raises NotImplementedError
# ---------------------------------------------------------------------------


def test_run_migration_raises_notimplementederror() -> None:
    """The unpatched migration helper points users to the manual fallback.

    A real cross-backend migration would force both optional clients into
    the install graph; the export/import roundtrip already supports the
    manual workflow, so the helper raises with explicit guidance instead.
    """
    with pytest.raises(NotImplementedError, match="crossmem export"):
        configure._run_migration("sqlite", "chroma")


def test_configure_propagates_migration_notimplementederror(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpatched ``--migrate`` surfaces ``NotImplementedError`` to the caller.

    Migration runs *before* config.toml is written, so a failed migration
    leaves the previous backend active. That way the user can follow the
    manual-fallback hint (``crossmem export``) against the backend that
    still holds the data instead of an empty newly-switched target.
    """
    configure.configure(backend="sqlite")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    with pytest.raises(NotImplementedError):
        configure.configure(backend="chroma", url="http://h", migrate=True)
    # Config update did NOT happen — the previous sqlite backend stays active
    # so the user's export step in the fallback hint sees the real data.
    assert configure.load_config()["backend"]["name"] == "sqlite"


def test_cli_configure_reports_migration_not_implemented(
    fake_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crossmem configure --migrate`` surfaces the manual fallback hint."""
    configure.configure(backend="sqlite")
    exit_code = cli.main(
        ["configure", "--backend", "chroma", "--url", "http://h", "--migrate"]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Backend switch not applied" in err
    assert "crossmem export" in err
    # Config file is unchanged — previous backend stays active so the
    # export step in the hint runs against the data-holding backend.
    assert configure.load_config()["backend"]["name"] == "sqlite"


def test_configure_result_migrated_true_when_helper_returns_cleanly(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``migrated`` only flips to True after _run_migration returns cleanly.

    Tests previously used the helper as a no-op stub; we keep that contract
    explicit: a successful return == real migration ran, ``migrated=True``.
    """
    configure.configure(backend="sqlite")
    monkeypatch.setattr(configure, "_run_migration", lambda *_a: None)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    result = configure.configure(backend="chroma", url="http://h")
    assert result.migrated is True


# ---------------------------------------------------------------------------
# build_backend — central factory
# ---------------------------------------------------------------------------


def test_build_backend_defaults_to_sqlite(fake_home: Path, tmp_path: Path) -> None:
    """No config -> SQLite backend at the supplied path."""
    from crossmem.backends.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "knowledge.db"
    backend = configure.build_backend(sqlite_path=db_path)
    assert isinstance(backend, SQLiteBackend)


def test_build_backend_uses_default_sqlite_path(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No sqlite_path supplied -> honour CROSSMEM_DB_PATH then ~/.crossmem."""
    from crossmem.backends.sqlite_backend import SQLiteBackend

    monkeypatch.setenv("CROSSMEM_DB_PATH", str(tmp_path / "from-env.db"))
    backend = configure.build_backend()
    assert isinstance(backend, SQLiteBackend)
    monkeypatch.delenv("CROSSMEM_DB_PATH", raising=False)
    backend2 = configure.build_backend()
    assert isinstance(backend2, SQLiteBackend)


def test_build_backend_reads_config_when_none(fake_home: Path, tmp_path: Path) -> None:
    """When called without a config, the function loads ``config.toml``."""
    from crossmem.backends.sqlite_backend import SQLiteBackend

    configure.configure(backend="sqlite")
    backend = configure.build_backend(sqlite_path=tmp_path / "knowledge.db")
    assert isinstance(backend, SQLiteBackend)


def test_build_backend_dispatches_chroma(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backend = chroma`` instantiates ChromaBackend with the URL from config."""
    fake_chroma = MagicMock(name="ChromaBackend")
    import crossmem.backends.chroma_backend as chroma_module

    monkeypatch.setattr(chroma_module, "ChromaBackend", fake_chroma)
    cfg = {"backend": {"name": "chroma", "url": "http://h:8000"}}
    configure.build_backend(cfg)
    fake_chroma.assert_called_once_with(path="http://h:8000")


def test_build_backend_dispatches_qdrant(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backend = qdrant`` forwards URL and api_key to QdrantBackend."""
    fake_qdrant = MagicMock(name="QdrantBackend")
    import crossmem.backends.qdrant_backend as qdrant_module

    monkeypatch.setattr(qdrant_module, "QdrantBackend", fake_qdrant)
    cfg = {
        "backend": {
            "name": "qdrant",
            "url": "https://q.example",
            "api_key": "secret",
        }
    }
    configure.build_backend(cfg)
    fake_qdrant.assert_called_once_with(url="https://q.example", api_key="secret")


def test_build_backend_rejects_unknown_name(fake_home: Path) -> None:
    """Unknown ``[backend].name`` raises BackendConfigError with the path."""
    cfg = {"backend": {"name": "bogus"}}
    with pytest.raises(configure.BackendConfigError, match="bogus"):
        configure.build_backend(cfg)


def test_build_backend_chroma_missing_extra_raises_backendconfigerror(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing chromadb -> BackendConfigError mentioning the extras name."""
    # Setting ``sys.modules[name] = None`` makes ``import name`` raise
    # ``ImportError`` immediately, which is the standard stdlib hook for
    # simulating a missing optional dependency.
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "crossmem.backends.chroma_backend", None)
    cfg = {"backend": {"name": "chroma", "url": "http://h"}}
    with pytest.raises(configure.BackendConfigError, match="crossmem\\[chroma\\]"):
        configure.build_backend(cfg)


def test_build_backend_qdrant_missing_extra_raises_backendconfigerror(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing qdrant_client -> BackendConfigError mentioning the extras name."""
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "crossmem.backends.qdrant_backend", None)
    cfg = {"backend": {"name": "qdrant", "url": "https://q"}}
    with pytest.raises(configure.BackendConfigError, match="crossmem\\[qdrant\\]"):
        configure.build_backend(cfg)


# ---------------------------------------------------------------------------
# Runtime wiring — server.main() and cli.build_default_store() honour config
# ---------------------------------------------------------------------------


def test_server_main_uses_build_backend(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``server.main()`` routes the active backend through build_backend."""
    from crossmem import server as server_module

    configure.configure(backend="qdrant", url="https://q", api_key="k")

    fake_backend = MagicMock(name="backend")
    seen_cfg: list[dict[str, object]] = []

    def fake_build_backend(cfg: dict[str, object], *, sqlite_path: Path) -> object:
        seen_cfg.append(cfg)
        return fake_backend

    monkeypatch.setattr(server_module, "build_backend", fake_build_backend)
    monkeypatch.setattr(server_module, "EmbeddingService", MagicMock())
    monkeypatch.setattr(server_module, "KnowledgeStore", MagicMock())
    fake_app = MagicMock()
    monkeypatch.setattr(
        server_module, "create_server", MagicMock(return_value=fake_app)
    )
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(tmp_path / "knowledge.db"))

    server_module.main()

    assert seen_cfg, "build_backend was not called"
    assert seen_cfg[0]["backend"]["name"] == "qdrant"
    assert seen_cfg[0]["backend"]["url"] == "https://q"


def test_cli_build_default_store_uses_build_backend(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``cli.build_default_store()`` honours the configured backend."""
    configure.configure(backend="chroma", url="http://h:8000")

    fake_backend = MagicMock(name="backend")
    seen_cfg: list[dict[str, object]] = []

    def fake_build_backend(cfg: dict[str, object], *, sqlite_path: Path) -> object:
        seen_cfg.append(cfg)
        return fake_backend

    monkeypatch.setattr(cli, "build_backend", fake_build_backend)
    monkeypatch.setattr(cli, "EmbeddingService", MagicMock())
    monkeypatch.setenv("CROSSMEM_DB_PATH", str(tmp_path / "knowledge.db"))

    cli.build_default_store()

    assert seen_cfg, "build_backend was not called"
    assert seen_cfg[0]["backend"]["name"] == "chroma"
    assert seen_cfg[0]["backend"]["url"] == "http://h:8000"


def test_server_main_aborts_when_backend_unavailable(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing optional backend surfaces a friendly error, not a traceback."""
    from crossmem import server as server_module

    def boom(_cfg: dict[str, object], *, sqlite_path: Path) -> object:
        raise configure.BackendConfigError(
            "backend 'chroma' is configured but chromadb is not installed. "
            "Install it with: pip install crossmem[chroma]"
        )

    monkeypatch.setattr(server_module, "build_backend", boom)
    monkeypatch.setattr(server_module, "EmbeddingService", MagicMock())
    monkeypatch.setattr(server_module, "KnowledgeStore", MagicMock())
    monkeypatch.setattr(server_module, "create_server", MagicMock())

    with pytest.raises(SystemExit) as exc:
        server_module.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "crossmem[chroma]" in err


def test_cli_export_reports_backend_config_error(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``crossmem export`` surfaces BackendConfigError to stderr, exit 1."""

    def boom() -> object:
        raise configure.BackendConfigError(
            "backend 'qdrant' is configured but qdrant-client is not installed. "
            "Install it with: pip install crossmem[qdrant]"
        )

    monkeypatch.setattr(cli, "build_default_store", boom)
    exit_code = cli.main(["export", "--path", str(tmp_path / "out.zip")])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "crossmem[qdrant]" in err
