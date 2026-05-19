"""Preflight checks for the CrossMem environment.

End-users run these via ``crossmem doctor`` to confirm that the Python
runtime, required libraries, optional backends, and the on-disk paths used
by CrossMem are all in order before installing or using the server.

Each check is a small private function returning a :class:`CheckResult` so
tests can target it individually. :func:`run_checks` simply composes them
in a stable order.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossmem.connectors.base import CLIConnector

__all__ = ["CheckResult", "build_support_info", "run_checks"]

# Project URLs surfaced via :func:`build_support_info` so the LLM-driven
# install flow can point users at the right tracker / docs when something
# fails. These are stable identifiers — they live next to ``doctor`` (not in
# ``pyproject.toml``) because the doctor JSON payload is the consumed
# interface, and a single source of truth here avoids parsing wheel metadata
# at runtime.
_PROJECT_URL = "https://github.com/feanor5555/knowledge"
_ISSUES_URL = f"{_PROJECT_URL}/issues"
_DOCS_URL = _PROJECT_URL

Status = Literal["ok", "warn", "fail"]

_MIN_PYTHON = (3, 10)


@dataclass(frozen=True)
class CheckResult:
    """One preflight check outcome.

    ``status`` is a tri-state: ``ok`` means the requirement is satisfied,
    ``warn`` means the check is informational (e.g. an optional dependency
    is missing), and ``fail`` means the user needs to act before CrossMem
    can run.
    """

    name: str
    status: Status
    detail: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_python_version() -> CheckResult:
    info = sys.version_info
    actual = f"{info.major}.{info.minor}.{info.micro}"
    if (info.major, info.minor) >= _MIN_PYTHON:
        return CheckResult(
            name="python_version",
            status="ok",
            detail=f"Python {actual} (>= 3.10 required)",
        )
    return CheckResult(
        name="python_version",
        status="fail",
        detail=(
            f"Python {actual} is too old; CrossMem requires >= 3.10. "
            "Install a newer Python and reinstall crossmem."
        ),
    )


def _try_import(module_name: str) -> tuple[bool, str]:
    """Attempt to import ``module_name``; return (ok, detail-message).

    Uses :func:`__import__` directly (rather than ``importlib.import_module``)
    so tests can intercept by monkeypatching ``builtins.__import__``.
    """
    try:
        __import__(module_name)
    except ImportError as exc:
        return False, f"cannot import {module_name}: {exc}"
    return True, f"{module_name} import succeeded"


def _required_module_check(name: str, module_name: str) -> CheckResult:
    ok, detail = _try_import(module_name)
    return CheckResult(
        name=name,
        status="ok" if ok else "fail",
        detail=detail,
    )


def _optional_module_check(name: str, module_name: str, extra: str) -> CheckResult:
    ok, detail = _try_import(module_name)
    if ok:
        return CheckResult(name=name, status="ok", detail=detail)
    return CheckResult(
        name=name,
        status="warn",
        detail=(
            f"{module_name} not installed (optional). "
            f"Install with: pip install crossmem[{extra}]"
        ),
    )


def _check_module_fastembed() -> CheckResult:
    return _required_module_check("module_fastembed", "fastembed")


def _check_module_sqlite_vec() -> CheckResult:
    return _required_module_check("module_sqlite_vec", "sqlite_vec")


def _check_module_fastmcp() -> CheckResult:
    return _required_module_check("module_fastmcp", "fastmcp")


def _check_module_httpx() -> CheckResult:
    return _required_module_check("module_httpx", "httpx")


def _check_module_bs4() -> CheckResult:
    return _required_module_check("module_bs4", "bs4")


def _check_module_yaml() -> CheckResult:
    return _required_module_check("module_yaml", "yaml")


def _check_optional_chromadb() -> CheckResult:
    return _optional_module_check("optional_chromadb", "chromadb", "chroma")


def _check_optional_qdrant_client() -> CheckResult:
    return _optional_module_check("optional_qdrant_client", "qdrant_client", "qdrant")


def _check_db_dir_writable() -> CheckResult:
    """Verify that ``~/.crossmem`` exists or can be created and is writable."""
    target = Path.home() / ".crossmem"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return CheckResult(
            name="db_dir_writable",
            status="fail",
            detail=f"cannot create {target}: {exc}",
        )
    # Probe by writing a tempfile inside the directory and removing it.
    try:
        with tempfile.NamedTemporaryFile(
            dir=target, prefix=".crossmem-doctor-", delete=False
        ) as fh:
            probe = Path(fh.name)
        probe.write_bytes(b"crossmem-doctor")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return CheckResult(
            name="db_dir_writable",
            status="fail",
            detail=f"cannot write inside {target}: {exc}",
        )
    return CheckResult(
        name="db_dir_writable",
        status="ok",
        detail=f"{target} is writable",
    )


def _check_embedding_cache_reachable() -> CheckResult:
    """Verify that ``~/.cache`` (parent of HF cache) is creatable/writable.

    Returns ``warn`` rather than ``fail`` if it isn't, since fastembed
    transparently falls back to a tempdir when its preferred cache is
    unavailable.
    """
    target = Path.home() / ".cache"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return CheckResult(
            name="embedding_cache_reachable",
            status="warn",
            detail=(
                f"cache dir {target} not creatable ({exc}); "
                "fastembed will fall back to a tempdir"
            ),
        )
    return CheckResult(
        name="embedding_cache_reachable",
        status="ok",
        detail=f"{target} is reachable",
    )


def _check_embedding_model_supported() -> CheckResult:
    """Verify that ``EMBEDDING_MODEL`` is in fastembed's supported list.

    Catches the failure mode discovered in 0.6.12: a default model that
    fastembed cannot load raises ``ValueError`` on the first ``embed*``
    call, after the user has already wired the MCP server into their CLI.
    Detecting it here surfaces the problem before any user-facing call.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        return CheckResult(
            name="embedding_model_supported",
            status="fail",
            detail=f"cannot import fastembed: {exc}",
        )
    try:
        from crossmem.core.embedding import EMBEDDING_MODEL
    except ImportError as exc:  # pragma: no cover - defensive
        return CheckResult(
            name="embedding_model_supported",
            status="fail",
            detail=f"cannot import crossmem.core.embedding: {exc}",
        )
    try:
        models = TextEmbedding.list_supported_models()
    except Exception as exc:  # noqa: BLE001 - fastembed-internal failures
        return CheckResult(
            name="embedding_model_supported",
            status="fail",
            detail=f"cannot list fastembed models: {exc}",
        )
    names = {
        (m.get("model") if isinstance(m, dict) else getattr(m, "model", None))
        for m in models
    }
    if EMBEDDING_MODEL in names:
        return CheckResult(
            name="embedding_model_supported",
            status="ok",
            detail=f"{EMBEDDING_MODEL} is supported by fastembed",
        )
    return CheckResult(
        name="embedding_model_supported",
        status="fail",
        detail=(
            f"default embedding model {EMBEDDING_MODEL!r} is not in "
            "fastembed's supported list; the first embed* call would raise "
            "ValueError. Upgrade fastembed or pick a model from "
            "TextEmbedding.list_supported_models()."
        ),
    )


def _check_pi_mcp_adapter() -> CheckResult | None:
    """Verify the ``pi-mcp-adapter`` is installed when the Pi CLI is detected.

    Pi has no native MCP support; the npm package ``pi-mcp-adapter`` bridges
    Pi to MCP servers. Returns ``None`` when Pi is not detected so the check
    is omitted entirely from non-Pi environments (additive evolution — see
    schemas/doctor.json).

    Why these two markers: ``pi-mcp-adapter`` ships as an npm package whose
    ``bin`` entry lands on ``$PATH`` after ``npm install -g``; alternatively
    Pi installs it as a HOME-relative extension under ``~/.pi/extensions/``
    (consistent with Pi's HOME-only convention noted in connectors/pi.py).
    Either marker is sufficient.
    """
    # Avoid importing PiConnector at module top to keep the doctor module
    # importable without the connectors package (e.g. minimal test setups).
    from crossmem.connectors.pi import PiConnector

    if not PiConnector().detect():
        return None

    on_path = shutil.which("pi-mcp-adapter") is not None
    ext_dir = Path.home() / ".pi" / "extensions" / "pi-mcp-adapter"
    has_extension = ext_dir.is_dir()

    if on_path or has_extension:
        location = shutil.which("pi-mcp-adapter") if on_path else str(ext_dir)
        return CheckResult(
            name="pi_mcp_adapter",
            status="ok",
            detail=f"pi-mcp-adapter found at {location}",
        )
    return CheckResult(
        name="pi_mcp_adapter",
        status="warn",
        detail=(
            "pi-mcp-adapter not found — install via `npm install -g "
            "pi-mcp-adapter` to use Pi with crossmem"
        ),
    )


# ---------------------------------------------------------------------------
# Install-doc presence (task 17.3)
# ---------------------------------------------------------------------------


def _install_docs_dir() -> Path:
    """Return the directory that holds ``install/<cli>.md`` guides.

    We resolve relative to the ``crossmem`` package location:

    * In a **source checkout** the doctor module lives at
      ``<repo>/src/crossmem/doctor.py``, so ``parents[2]`` is the repo
      root and ``install/`` sits next to ``src/``.
    * For a **pip-installed wheel** there is no analogous ``install/``
      directory today (task 15.5 commits per-CLI files but does not
      yet wire them into ``setuptools.package_data``). In that case
      the returned path simply does not exist; the per-connector check
      then reports ``warn`` for every detected CLI, which is the
      correct user-facing signal: regenerate the docs from source via
      ``crossmem docs install --cli <name>`` (task 15.4).

    Returning a ``Path`` (even when it does not exist) keeps the
    function trivially monkeypatchable in tests.
    """
    return Path(__file__).resolve().parents[2] / "install"


def _install_doc_connectors() -> list[CLIConnector]:
    """Return one connector instance per shipped CLI.

    Lives in its own function so tests can replace the connector list
    with stubs (see ``tests/doctor/test_install_doc_check.py``). We
    import lazily because ``crossmem.installer`` pulls in heavier
    dependencies (sqlite-vec, fastembed) at module import time, and
    the doctor module is meant to stay importable on its own.

    Delegates to :func:`crossmem.installer.instantiate_connectors` so
    the connector set never drifts between doctor and install.
    """
    from crossmem import installer

    return installer.instantiate_connectors()


def _per_detected_connector(
    check_fn: Callable[[CLIConnector], CheckResult],
) -> list[CheckResult]:
    """Run ``check_fn`` for each **detected** connector and collect results.

    Captures the iteration over :func:`_install_doc_connectors`, the
    ``try/except`` around :meth:`CLIConnector.detect` (so one broken
    connector never breaks doctor) and the skip-when-not-detected rule
    that backs the additive-evolution contract in
    ``schemas/doctor.json``: per-connector check names only appear when
    the connector is actually present on the user's machine.

    Undetected connectors are omitted entirely (no ``ok``, no ``warn``).
    """
    results: list[CheckResult] = []
    for connector in _install_doc_connectors():
        try:
            detected = connector.detect()
        except Exception:  # noqa: BLE001 - never let one bad connector break doctor
            continue
        if not detected:
            continue
        results.append(check_fn(connector))
    return results


def _check_install_doc_present() -> list[CheckResult]:
    """Emit one ``install_doc_<cli>`` check per **detected** connector.

    Undetected connectors are omitted entirely (no ``ok``, no ``warn``),
    matching the additive-evolution rule in ``schemas/doctor.json`` —
    new check names appear only when relevant.

    Status:

    * ``ok``   — ``install/<cli>.md`` exists.
    * ``warn`` — file missing (or install dir absent). Detail points
      the LLM at ``crossmem docs install --cli <name>`` (task 15.4)
      to regenerate it.
    """
    install_dir = _install_docs_dir()

    def check(connector: CLIConnector) -> CheckResult:
        cli_name = connector.name()
        doc_path = install_dir / f"{cli_name}.md"
        check_name = f"install_doc_{cli_name}"
        if doc_path.is_file():
            return CheckResult(
                name=check_name,
                status="ok",
                detail=f"install doc present at {doc_path}",
            )
        return CheckResult(
            name=check_name,
            status="warn",
            detail=(
                f"missing install doc for {cli_name} "
                f"(expected {doc_path}); regenerate via "
                f"`crossmem docs install --cli {cli_name}`"
            ),
        )

    return _per_detected_connector(check)


# ---------------------------------------------------------------------------
# Backup retention (task 21.7)
# ---------------------------------------------------------------------------


def _count_backup_siblings(path: Path) -> int:
    """Count ``<path>.bak.*`` files in ``path.parent``.

    Mirrors :func:`crossmem.connectors_status._count_backups` but kept
    local to the doctor module to avoid coupling the preflight check to
    the ``crossmem status`` rendering layer.
    """
    from crossmem.connectors.config_io import BACKUP_PREFIX

    parent = path.parent
    if not parent.is_dir():
        return 0
    prefix = f"{path.name}{BACKUP_PREFIX}"
    return sum(1 for entry in parent.iterdir() if entry.name.startswith(prefix))


def _check_backup_retention() -> list[CheckResult]:
    """Emit one ``backup_retention_<cli>`` check per **detected** connector.

    After task 21.7 :mod:`crossmem.connectors.config_io` enforces
    :data:`crossmem.connectors.config_io.BACKUP_RETENTION` newest
    backups on every write. A directory still holding more than that
    cap means the user either has not run an install since the upgrade
    or some other tooling wrote outside the helper. Either way the
    excess is recoverable manually — surface it as ``warn`` with the
    actual count so the LLM (or human) can ``rm`` the stale files.

    Status:

    * ``ok``   — backup count <= cap (including zero).
    * ``warn`` — backup count > cap; detail enumerates both counts and
                 points at the on-disk directory.

    Undetected connectors are omitted entirely (additive evolution —
    same pattern as :func:`_check_install_doc_present`).
    """
    from crossmem.connectors.config_io import BACKUP_PREFIX, BACKUP_RETENTION

    def check(connector: CLIConnector) -> CheckResult:
        cli_name = connector.name()
        cfg_path = connector.config_path()
        count = _count_backup_siblings(cfg_path)
        check_name = f"backup_retention_{cli_name}"
        if count <= BACKUP_RETENTION:
            return CheckResult(
                name=check_name,
                status="ok",
                detail=(
                    f"{count} backup file(s) at {cfg_path.parent} "
                    f"(<= {BACKUP_RETENTION} retained)"
                ),
            )
        return CheckResult(
            name=check_name,
            status="warn",
            detail=(
                f"{count} ``{cfg_path.name}{BACKUP_PREFIX}*`` files "
                f"at {cfg_path.parent} exceed the retention of "
                f"{BACKUP_RETENTION}; the next register/unregister "
                f"call will prune to {BACKUP_RETENTION}, or remove "
                f"the oldest manually"
            ),
        )

    return _per_detected_connector(check)


# ---------------------------------------------------------------------------
# Sensitive-file permissions (task 20.5)
# ---------------------------------------------------------------------------

_SENSITIVE_FILES: tuple[str, ...] = (
    "config.toml",
    ".crossmem-trash.jsonl",
    "knowledge.db",
)


def _check_sensitive_file_permissions() -> CheckResult:
    """Warn when sensitive files have permissions looser than ``0o600``.

    On Linux/Mac the configure/cleanup/sqlite writers clamp these files
    to ``0o600`` themselves, but a user who pre-created the files (or a
    migration from an older crossmem version) may have left them at the
    umask default ``0o644``. This check surfaces that so the user can
    fix it manually with ``chmod 600``.

    On Windows POSIX modes are not expressible and the check is a no-op
    (returns ``ok`` with a "not applicable" detail).
    """
    if sys.platform == "win32":
        return CheckResult(
            name="sensitive_file_permissions",
            status="ok",
            detail="POSIX file modes not applicable on Windows",
        )

    crossmem_dir = Path.home() / ".crossmem"
    offenders: list[str] = []
    for name in _SENSITIVE_FILES:
        path = crossmem_dir / name
        if not path.exists():
            continue
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            offenders.append(f"{name} (mode={oct(mode)})")

    if not offenders:
        return CheckResult(
            name="sensitive_file_permissions",
            status="ok",
            detail=f"all sensitive files under {crossmem_dir} are 0o600 or absent",
        )
    return CheckResult(
        name="sensitive_file_permissions",
        status="warn",
        detail=(
            f"sensitive file(s) under {crossmem_dir} have permissions looser "
            f"than 0o600: {', '.join(offenders)}. Run `chmod 600 "
            f"{crossmem_dir}/<file>` to harden."
        ),
    )


# ---------------------------------------------------------------------------
# Backend dimension vs. model (task 21.3)
# ---------------------------------------------------------------------------


def _resolve_sqlite_db_path() -> Path:
    """Return the SQLite DB path used by the local backend.

    Honours ``CROSSMEM_DB_PATH`` (used by tests and advanced setups) and
    falls back to the documented default at ``~/.crossmem/knowledge.db``.
    Mirrors :func:`crossmem.configure._default_sqlite_path` without
    importing it (the configure module pulls in TOML helpers we don't
    want to drag into the doctor module).
    """
    override = os.environ.get("CROSSMEM_DB_PATH")
    if override:
        return Path(override)
    return Path.home() / ".crossmem" / "knowledge.db"


def _check_backend_dim_matches_model() -> CheckResult:
    """Verify that stored embeddings' dimension equals the model's dimension.

    Reads ``DISTINCT embedding_dim`` from the local SQLite ``documents``
    table and compares against ``EMBEDDING_DIM``. A mismatch means the
    embedding model has been swapped (or the DB was migrated from an
    older crossmem) without a re-embedding pass — queries against the
    legacy rows would silently return garbage.

    Status:

    * ``ok``   — DB absent, empty, or every stored ``embedding_dim``
                 matches ``EMBEDDING_DIM``.
    * ``fail`` — at least one stored row has a different dim; the detail
                 enumerates the offending values.

    The check opens the DB read-only and never creates or migrates a
    file, so it is safe to run before ``crossmem install``.
    """
    try:
        from crossmem.core.embedding import EMBEDDING_DIM
    except ImportError as exc:  # pragma: no cover - defensive
        return CheckResult(
            name="backend_dim_matches_model",
            status="fail",
            detail=f"cannot import crossmem.core.embedding: {exc}",
        )

    db_path = _resolve_sqlite_db_path()
    if not db_path.exists():
        return CheckResult(
            name="backend_dim_matches_model",
            status="ok",
            detail=(
                f"no local SQLite DB at {db_path} yet; nothing to "
                "compare against the embedding model dimension"
            ),
        )

    # Open read-only via the SQLite URI form so the check never creates
    # the file or upgrades the schema as a side effect.
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as exc:
        return CheckResult(
            name="backend_dim_matches_model",
            status="fail",
            detail=f"cannot open {db_path} read-only: {exc}",
        )
    try:
        try:
            rows = conn.execute(
                "SELECT DISTINCT embedding_dim FROM documents "
                "WHERE embedding_dim IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # ``documents`` table absent — DB exists but is empty/fresh.
            return CheckResult(
                name="backend_dim_matches_model",
                status="ok",
                detail=(
                    f"{db_path} has no ``documents`` table yet "
                    f"({exc}); nothing to compare"
                ),
            )
    finally:
        conn.close()

    dims = sorted({int(r[0]) for r in rows})
    if not dims:
        return CheckResult(
            name="backend_dim_matches_model",
            status="ok",
            detail=f"{db_path} has no stored documents yet",
        )
    bad = [d for d in dims if d != EMBEDDING_DIM]
    if not bad:
        return CheckResult(
            name="backend_dim_matches_model",
            status="ok",
            detail=(
                f"all stored embeddings use dim={EMBEDDING_DIM} "
                f"(matches the active model)"
            ),
        )
    return CheckResult(
        name="backend_dim_matches_model",
        status="fail",
        detail=(
            f"stored embedding dim(s) {bad} in {db_path} do not match "
            f"the active model dim {EMBEDDING_DIM}; re-embed via "
            "`crossmem reembed` or delete the DB to start fresh"
        ),
    )


# ---------------------------------------------------------------------------
# Support info (task 19.1)
# ---------------------------------------------------------------------------


def build_support_info() -> dict[str, str]:
    """Return the ``support`` payload surfaced under ``doctor --json``.

    ``skills/crossmem-install/SKILL.md`` instructs the LLM to look here for
    the project issue tracker when something goes wrong, so:

    * ``issues_url`` — the GitHub Issues tracker (SKILL.md's stated contract).
    * ``docs_url``   — the canonical project / docs URL, for the same LLM to
      hand the user a starting page when filing a bug.
    * ``version``    — the installed ``crossmem`` package version from
      :func:`importlib.metadata.version`, falling back to ``"0.0.0"`` when
      running from an uninstalled source tree (mirrors
      :func:`crossmem.docs.install_template._crossmem_version`).

    Additive — see ``schemas/doctor.json`` (``additionalProperties: true``
    at top level keeps the schema version pinned at ``"1"``).
    """
    try:
        version = metadata.version("crossmem")
    except metadata.PackageNotFoundError:
        version = "0.0.0"
    return {
        "issues_url": _ISSUES_URL,
        "docs_url": _DOCS_URL,
        "version": version,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_checks() -> list[CheckResult]:
    """Run every preflight check and return the results in a stable order."""
    results: list[CheckResult] = [
        _check_python_version(),
        _check_module_fastembed(),
        _check_module_sqlite_vec(),
        _check_module_fastmcp(),
        _check_module_httpx(),
        _check_module_bs4(),
        _check_module_yaml(),
        _check_optional_chromadb(),
        _check_optional_qdrant_client(),
        _check_db_dir_writable(),
        _check_embedding_cache_reachable(),
        _check_embedding_model_supported(),
        _check_backend_dim_matches_model(),
        _check_sensitive_file_permissions(),
    ]
    # Connector-specific checks are appended only when the matching connector
    # is detected, so non-users see no noise.
    pi_result = _check_pi_mcp_adapter()
    if pi_result is not None:
        results.append(pi_result)
    results.extend(_check_install_doc_present())
    results.extend(_check_backup_retention())
    return results
