"""Tests for ``crossmem.doctor`` preflight checks.

The doctor module is invoked by end-users via ``crossmem doctor`` (CLI wiring
in a separate task). These tests target each individual check function with
mocks so the suite never depends on the real environment, plus a roundtrip
test for ``run_checks()`` to verify ordering and overall composition.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from crossmem import doctor
from crossmem.doctor import CheckResult

# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------


def test_checkresult_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    result = CheckResult(name="x", status="ok", detail="d")
    with pytest.raises(FrozenInstanceError):
        result.name = "y"  # type: ignore[misc]


def test_checkresult_fields() -> None:
    r = CheckResult(name="python_version", status="ok", detail="3.12.0")
    assert r.name == "python_version"
    assert r.status == "ok"
    assert r.detail == "3.12.0"


# ---------------------------------------------------------------------------
# _check_python_version
# ---------------------------------------------------------------------------


class _FakeVersion(tuple):
    """Tuple that also exposes major/minor/micro attributes."""

    def __new__(cls, major: int, minor: int, micro: int = 0):
        instance = super().__new__(cls, (major, minor, micro, "final", 0))
        instance.major = major  # type: ignore[attr-defined]
        instance.minor = minor  # type: ignore[attr-defined]
        instance.micro = micro  # type: ignore[attr-defined]
        return instance


def test_python_version_ok_for_current_runtime() -> None:
    # The interpreter running the tests is >= 3.10 (project floor).
    result = doctor._check_python_version()
    assert result.name == "python_version"
    assert result.status == "ok"
    assert str(sys.version_info.major) in result.detail


def test_python_version_fails_below_310(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", _FakeVersion(3, 9, 7))
    result = doctor._check_python_version()
    assert result.name == "python_version"
    assert result.status == "fail"
    assert "3.9" in result.detail


def test_python_version_ok_at_exactly_310(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", _FakeVersion(3, 10, 0))
    result = doctor._check_python_version()
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Required module checks
# ---------------------------------------------------------------------------


def _make_failing_import(blocked: set[str]):
    """Return an ``__import__`` shim that raises ImportError for blocked names."""
    real_import = builtins.__import__

    def shim(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        top = name.split(".")[0]
        if top in blocked:
            raise ImportError(f"simulated missing module: {top}")
        return real_import(name, globals, locals, fromlist, level)

    return shim


@pytest.mark.parametrize(
    ("module_name", "check_func_name"),
    [
        ("fastembed", "_check_module_fastembed"),
        ("sqlite_vec", "_check_module_sqlite_vec"),
        ("fastmcp", "_check_module_fastmcp"),
        ("httpx", "_check_module_httpx"),
        ("bs4", "_check_module_bs4"),
        ("yaml", "_check_module_yaml"),
    ],
)
def test_required_module_ok_when_present(
    module_name: str, check_func_name: str
) -> None:
    # All required modules ARE installed in the dev environment.
    func = getattr(doctor, check_func_name)
    result = func()
    assert result.status == "ok", f"{module_name}: {result.detail}"
    assert result.name == f"module_{module_name}"


@pytest.mark.parametrize(
    ("module_name", "check_func_name"),
    [
        ("fastembed", "_check_module_fastembed"),
        ("sqlite_vec", "_check_module_sqlite_vec"),
        ("fastmcp", "_check_module_fastmcp"),
        ("httpx", "_check_module_httpx"),
        ("bs4", "_check_module_bs4"),
        ("yaml", "_check_module_yaml"),
    ],
)
def test_required_module_fails_when_missing(
    module_name: str,
    check_func_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drop any cached import so the shim runs.
    for mod in list(sys.modules):
        if mod == module_name or mod.startswith(module_name + "."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", _make_failing_import({module_name}))
    func = getattr(doctor, check_func_name)
    result = func()
    assert result.status == "fail"
    assert result.name == f"module_{module_name}"
    assert module_name in result.detail


# ---------------------------------------------------------------------------
# Optional module checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "check_func_name", "extra"),
    [
        ("chromadb", "_check_optional_chromadb", "chroma"),
        ("qdrant_client", "_check_optional_qdrant_client", "qdrant"),
    ],
)
def test_optional_module_warn_when_missing(
    module_name: str,
    check_func_name: str,
    extra: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mod in list(sys.modules):
        if mod == module_name or mod.startswith(module_name + "."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", _make_failing_import({module_name}))
    func = getattr(doctor, check_func_name)
    result = func()
    assert result.status == "warn"
    # Detail should mention the optional install command.
    assert f"crossmem[{extra}]" in result.detail


def test_optional_chromadb_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate "installed" by injecting a stub into sys.modules so import works.
    import types

    stub = types.ModuleType("chromadb")
    monkeypatch.setitem(sys.modules, "chromadb", stub)
    result = doctor._check_optional_chromadb()
    assert result.status == "ok"
    assert result.name == "optional_chromadb"


def test_optional_qdrant_client_ok_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    stub = types.ModuleType("qdrant_client")
    monkeypatch.setitem(sys.modules, "qdrant_client", stub)
    result = doctor._check_optional_qdrant_client()
    assert result.status == "ok"
    assert result.name == "optional_qdrant_client"


# ---------------------------------------------------------------------------
# _check_db_dir_writable
# ---------------------------------------------------------------------------


def test_db_dir_writable_creates_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Redirect Path.home() to a fresh tmp dir; ".crossmem" does not yet exist.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = tmp_path / ".crossmem"
    assert not target.exists()

    result = doctor._check_db_dir_writable()
    assert result.status == "ok"
    assert result.name == "db_dir_writable"
    assert target.exists() and target.is_dir()


def test_db_dir_writable_ok_when_dir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".crossmem").mkdir()
    result = doctor._check_db_dir_writable()
    assert result.status == "ok"


def test_db_dir_writable_fails_when_path_is_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Create a regular file at ~/.crossmem to break mkdir.
    (tmp_path / ".crossmem").write_text("not a dir", encoding="utf-8")
    result = doctor._check_db_dir_writable()
    assert result.status == "fail"
    assert result.name == "db_dir_writable"


def test_db_dir_writable_fails_when_write_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the OS rejecting a write inside the directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".crossmem").mkdir()

    real_write_bytes = Path.write_bytes

    def fail_write(self: Path, data: bytes) -> int:
        if ".crossmem" in self.parts:
            raise PermissionError("simulated read-only fs")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    result = doctor._check_db_dir_writable()
    assert result.status == "fail"
    assert "simulated read-only fs" in result.detail or "Permission" in result.detail


# ---------------------------------------------------------------------------
# _check_embedding_cache_reachable
# ---------------------------------------------------------------------------


def test_embedding_cache_reachable_creates_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = tmp_path / ".cache"
    assert not cache.exists()
    result = doctor._check_embedding_cache_reachable()
    assert result.status == "ok"
    assert result.name == "embedding_cache_reachable"
    assert cache.exists()


def test_embedding_cache_reachable_ok_when_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".cache").mkdir()
    result = doctor._check_embedding_cache_reachable()
    assert result.status == "ok"


def test_embedding_cache_reachable_warn_when_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    real_mkdir = Path.mkdir

    def fail_mkdir(self: Path, *args, **kwargs):
        if self.name == ".cache":
            raise PermissionError("simulated cannot create")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    result = doctor._check_embedding_cache_reachable()
    assert result.status == "warn"
    assert result.name == "embedding_cache_reachable"


# ---------------------------------------------------------------------------
# _check_embedding_model_supported
# ---------------------------------------------------------------------------


def test_embedding_model_supported_ok_for_real_default() -> None:
    """The shipping default must be in fastembed's supported list.

    Regression guard for 0.6.12: an unsupported default raises only at
    runtime; this check makes that visible during ``crossmem doctor``.
    """
    result = doctor._check_embedding_model_supported()
    assert result.name == "embedding_model_supported"
    assert result.status == "ok", result.detail


def test_embedding_model_supported_fails_for_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the configured default isn't supported, the check must fail loudly."""
    monkeypatch.setattr(
        "crossmem.core.embedding.EMBEDDING_MODEL",
        "intfloat/multilingual-e5-small",
    )
    # Make list_supported_models return a list that excludes the bad default.
    fake_models = [{"model": "sentence-transformers/all-MiniLM-L6-v2"}]
    monkeypatch.setattr(
        "fastembed.TextEmbedding.list_supported_models",
        classmethod(lambda _cls: fake_models),
    )
    result = doctor._check_embedding_model_supported()
    assert result.status == "fail"
    assert "intfloat/multilingual-e5-small" in result.detail


def test_embedding_model_supported_handles_listing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure to enumerate fastembed models must not crash the check."""

    def boom(_cls):
        raise RuntimeError("fastembed registry unavailable")

    monkeypatch.setattr(
        "fastembed.TextEmbedding.list_supported_models",
        classmethod(boom),
    )
    result = doctor._check_embedding_model_supported()
    assert result.status == "fail"
    assert "fastembed registry unavailable" in result.detail


def test_embedding_model_supported_fails_when_fastembed_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without fastembed installed, the check fails with a clear message."""
    monkeypatch.delitem(sys.modules, "fastembed", raising=False)
    monkeypatch.setattr(builtins, "__import__", _make_failing_import({"fastembed"}))
    result = doctor._check_embedding_model_supported()
    assert result.status == "fail"
    assert "fastembed" in result.detail


# ---------------------------------------------------------------------------
# _check_backend_dim_matches_model
# ---------------------------------------------------------------------------


def test_backend_dim_matches_model_ok_when_db_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an on-disk DB yet, the check is a no-op (ok)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CROSSMEM_DB_PATH", raising=False)
    result = doctor._check_backend_dim_matches_model()
    assert result.name == "backend_dim_matches_model"
    assert result.status == "ok"
    assert "no local SQLite DB" in result.detail


def test_backend_dim_matches_model_ok_when_db_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh DB with the documents table but no rows -> ok (nothing to compare)."""
    from crossmem.backends.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "empty.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    monkeypatch.setenv("CROSSMEM_DB_PATH", str(db_path))
    result = doctor._check_backend_dim_matches_model()
    assert result.status == "ok"
    assert result.name == "backend_dim_matches_model"


def test_backend_dim_matches_model_ok_when_all_rows_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB populated with EMBEDDING_DIM-correct rows -> ok."""
    from crossmem.backends.sqlite_backend import SQLiteBackend
    from crossmem.core.embedding import EMBEDDING_DIM
    from crossmem.core.models import Document, Metadata

    db_path = tmp_path / "match.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(
            Document(
                id="a" * 32,
                content="ok",
                embedding=[0.1] * EMBEDDING_DIM,
                metadata=Metadata(
                    source_url="https://example.com/a",
                    title="A",
                    source_type="web",
                    stored_at="2024-01-01T00:00:00Z",
                    embedding_model="model",
                    embedding_dim=EMBEDDING_DIM,
                    namespace="default",
                    tags=[],
                    content_hash="h",
                ),
            )
        )
    finally:
        backend.close()

    monkeypatch.setenv("CROSSMEM_DB_PATH", str(db_path))
    result = doctor._check_backend_dim_matches_model()
    assert result.status == "ok"
    assert str(EMBEDDING_DIM) in result.detail


def test_backend_dim_matches_model_fails_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored row with a non-matching embedding_dim -> fail."""
    import sqlite3

    from crossmem.backends.sqlite_backend import SQLiteBackend
    from crossmem.core.embedding import EMBEDDING_DIM

    db_path = tmp_path / "mismatch.db"
    # Create the schema via SQLiteBackend, then close and write a row with
    # a legacy embedding_dim (768) using a raw connection so we bypass the
    # backend's own dimension enforcement.
    SQLiteBackend(db_path).close()

    legacy_dim = EMBEDDING_DIM + 384
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO documents
            (id, content, source_url, title, source_type, stored_at,
             embedding_model, embedding_dim, namespace, tags, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy" * 5,
                "legacy",
                "https://example.com/legacy",
                "L",
                "web",
                "2024-01-01T00:00:00Z",
                "legacy-model",
                legacy_dim,
                "default",
                "[]",
                "h",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("CROSSMEM_DB_PATH", str(db_path))
    result = doctor._check_backend_dim_matches_model()
    assert result.status == "fail"
    assert result.name == "backend_dim_matches_model"
    assert str(legacy_dim) in result.detail
    assert str(EMBEDDING_DIM) in result.detail


# ---------------------------------------------------------------------------
# _check_pi_mcp_adapter
# ---------------------------------------------------------------------------


def _make_pi_config(home: Path) -> Path:
    """Create a minimal Pi config so ``PiConnector.detect()`` returns True."""
    cfg = home / ".pi" / "agent" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    return cfg


def test_pi_mcp_adapter_skipped_when_pi_not_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Pi config -> check is not emitted at all."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ensure no executable on PATH would be detected.
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = doctor._check_pi_mcp_adapter()
    assert result is None


def test_pi_mcp_adapter_warn_when_pi_detected_but_no_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pi detected, but adapter neither on PATH nor under ~/.pi/extensions."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_pi_config(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = doctor._check_pi_mcp_adapter()
    assert result is not None
    assert result.name == "pi_mcp_adapter"
    assert result.status == "warn"
    # Detail should mention installation guidance.
    assert "pi-mcp-adapter" in result.detail
    assert "install" in result.detail.lower()


def test_pi_mcp_adapter_ok_when_adapter_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pi detected and adapter executable found on PATH -> ok."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_pi_config(tmp_path)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            "/usr/local/bin/pi-mcp-adapter" if name == "pi-mcp-adapter" else None
        ),
    )
    result = doctor._check_pi_mcp_adapter()
    assert result is not None
    assert result.name == "pi_mcp_adapter"
    assert result.status == "ok"


def test_pi_mcp_adapter_ok_when_extension_dir_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pi detected and adapter present under ~/.pi/extensions -> ok."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_pi_config(tmp_path)
    (tmp_path / ".pi" / "extensions" / "pi-mcp-adapter").mkdir(parents=True)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = doctor._check_pi_mcp_adapter()
    assert result is not None
    assert result.name == "pi_mcp_adapter"
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# run_checks() composition
# ---------------------------------------------------------------------------


def test_run_checks_returns_all_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sandbox the home dir so we don't touch the user's real cache/db dirs.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    results = doctor.run_checks()
    names = [r.name for r in results]

    # Pi connector is not detected in the sandboxed home -> pi_mcp_adapter
    # check is not emitted. Base list stays unchanged.
    assert names == [
        "python_version",
        "module_fastembed",
        "module_sqlite_vec",
        "module_fastmcp",
        "module_httpx",
        "module_bs4",
        "module_yaml",
        "optional_chromadb",
        "optional_qdrant_client",
        "db_dir_writable",
        "embedding_cache_reachable",
        "embedding_model_supported",
        "backend_dim_matches_model",
        "sensitive_file_permissions",
    ]
    # Every result must be a CheckResult with a valid status.
    for r in results:
        assert isinstance(r, CheckResult)
        assert r.status in {"ok", "warn", "fail"}


def test_run_checks_returns_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    results = doctor.run_checks()
    assert isinstance(results, list)
    assert len(results) == 14


def test_run_checks_appends_pi_check_when_pi_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Pi is detected, ``pi_mcp_adapter`` appears in the run_checks output."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_pi_config(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    results = doctor.run_checks()
    names = [r.name for r in results]
    assert "pi_mcp_adapter" in names
    pi_result = next(r for r in results if r.name == "pi_mcp_adapter")
    assert pi_result.status == "warn"
