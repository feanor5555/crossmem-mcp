"""``crossmem install`` — auto-detect MCP-capable CLIs and wire them up.

Three steps, each visible to the user via a printed progress line:

1. **Detect** every registered :class:`CLIConnector` and call ``register()``
   on those whose ``detect()`` returns True. Each connector is responsible
   for backing up its own config file before mutating it.
2. **Initialise** the SQLite knowledge base under
   ``~/.crossmem/knowledge.db`` so the schema migration runs ahead of the
   first MCP request (the actual MCP server reuses the same on-disk file).
3. **Materialise** the embedding model. The default
   :class:`EmbeddingService` constructor kicks off the fastembed download
   in a background thread; we print a size hint so users understand why
   the first run takes a while.

Both the connector list and the embedder factory are dependency-injected
so tests can run without touching real CLI configs or downloading the
~120MB ONNX model.

A ``dry_run=True`` invocation short-circuits all three steps: no
``register()`` is called, no DB file is created, and the embedder
factory is never invoked. Instead the installer prints a diff-form
preview (``would add`` / ``would update`` / ``already present``) per
detected connector and returns an :class:`InstallResult` with
``dry_run=True`` so callers can render an explicit preview summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.connectors.amazonq import AmazonQConnector
from crossmem.connectors.base import CLIConnector as _BaseCLIConnector
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
from crossmem.core.embedding import EmbeddingService
from crossmem.doctor import run_checks

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossmem.connectors.base import CLIConnector
    from crossmem.doctor import CheckResult

__all__ = [
    "ALL_CONNECTORS",
    "DEFAULT_SERVER_CMD",
    "InstallAbortedError",
    "InstallResult",
    "install",
    "instantiate_connectors",
]

# We default to ``python -m crossmem.server`` rather than the bare
# ``crossmem`` console-script: pipx-installed wheels expose ``crossmem`` on
# the path, but during development (``PYTHONPATH=src python -m pytest``) only
# the module form is reliably callable. CLIs that prefer a console-script
# entry can override at the connector level later.
DEFAULT_SERVER_CMD = "python -m crossmem.server"

# Registry of every shipped connector. ``installer.install()`` walks this
# list when no explicit ``connectors`` argument is given. Adding a new
# connector means: write the class, append a factory here, ship the test.
# Order is informational only — detected CLIs are reported in the same
# order they appear here.
ALL_CONNECTORS: list[Callable[[], CLIConnector]] = [
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
]


class InstallAbortedError(RuntimeError):
    """Raised by :func:`install` when ``doctor`` reports any ``fail`` check.

    Carries the list of failed :class:`CheckResult`s in ``failed_checks`` so
    callers (the CLI, tests) can render them however they like. The string
    form is a multi-line summary suitable for printing directly.
    """

    def __init__(self, failed_checks: list[CheckResult]) -> None:
        self.failed_checks = failed_checks
        lines = ["crossmem install aborted: doctor preflight reported failures."]
        for check in failed_checks:
            lines.append(f"  [fail] {check.name}: {check.detail}")
        lines.append(
            "Run `crossmem doctor` for full output, fix the issues above, "
            "then retry `crossmem install`."
        )
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class InstallResult:
    """Summary of what ``install()`` did, returned to the caller (and CLI).

    ``detected_clis`` is the list of CLI ``name()`` values that the
    installer wired up. ``db_path`` is the on-disk SQLite file the
    schema was created in (or *would have been* in dry-run mode).
    ``embedding_model`` is the name of the embedder the install warmed
    (so the user can see which model will be used at query time); for
    dry-run it is an empty string because no embedder is built.
    ``dry_run`` indicates whether the run was a preview (no writes).
    """

    detected_clis: list[str]
    db_path: Path
    embedding_model: str
    dry_run: bool = False


def _print(msg: str) -> None:
    """Single funnel for progress output (keeps tests targeting one channel)."""
    print(msg)


def instantiate_connectors() -> list[CLIConnector]:
    """Return one fresh instance per factory in :data:`ALL_CONNECTORS`.

    Canonical connector factory for the whole codebase: any caller that
    needs "one of every shipped CLI connector" funnels through here so
    the connector set never drifts between ``install``, ``uninstall``,
    ``status`` and ``doctor``. Tests can either monkeypatch this function
    directly or swap :data:`ALL_CONNECTORS` for a controlled list.
    """
    return [factory() for factory in ALL_CONNECTORS]


def _materialise_connectors(
    connectors: list[CLIConnector] | None,
) -> list[CLIConnector]:
    """Return ``connectors`` if supplied, else call :func:`instantiate_connectors`."""
    if connectors is not None:
        return connectors
    return instantiate_connectors()


def _restart_hint_line(connector: CLIConnector) -> str | None:
    """Return the indented restart-hint line for ``connector``, or ``None``.

    Non-GUI connectors return ``None`` (no hint printed). For GUI
    connectors we prefer the connector's own :meth:`restart_hint`
    override (e.g. tool-specific phrasing); when the connector does
    *not* override the base implementation we fall back to the short
    ``Restart <name> to pick up changes`` string the install spec asks
    for. The base class default ("Fully quit and relaunch ...") is
    too verbose for the per-connector progress line.
    """
    if not connector.is_gui_app:
        return None
    overrides_hint = type(connector).restart_hint is not _BaseCLIConnector.restart_hint
    text = (
        connector.restart_hint()
        if overrides_hint
        else f"Restart {connector.name()} to pick up changes"
    )
    return f"    {text}"


def _register_detected(connectors: list[CLIConnector]) -> list[str]:
    """Run ``register()`` on every connector whose ``detect()`` is True.

    Returns the list of CLI names that were wired up, in the same order
    as ``connectors``. Connectors are responsible for backing up their
    own config file before writing — see ``connectors/config_io.py``.

    GUI-style connectors (``is_gui_app=True``) additionally get a
    ``Restart <name> to pick up changes`` line printed immediately after
    a successful ``register()`` so the user knows the new MCP server
    will only become live after relaunching the app.
    """
    detected_names: list[str] = []
    for connector in connectors:
        name = connector.name()
        if not connector.detect():
            _print(f"  - {name}: not detected, skipping")
            continue
        _print(f"  - {name}: registering MCP server")
        connector.register(DEFAULT_SERVER_CMD)
        hint = _restart_hint_line(connector)
        if hint is not None:
            _print(hint)
        detected_names.append(name)
    return detected_names


def _ensure_database(home: Path) -> Path:
    """Create ``<home>/.crossmem/knowledge.db`` (running schema migration)."""
    db_path = home / ".crossmem" / "knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Instantiating SQLiteBackend runs ``_init_db`` which is idempotent
    # (every CREATE uses IF NOT EXISTS, schema_version row inserted once).
    SQLiteBackend(db_path)
    return db_path


def _plan_status(current: dict | None, planned: dict) -> str:
    """Compare the current entry with the planned snippet.

    * ``None`` -> ``"would add"`` (no entry yet, or unparseable config).
    * Equal dicts -> ``"already present"`` (re-install is a no-op).
    * Differing dicts -> ``"would update"`` (entry exists but the value
      differs — typically the wired ``server_cmd`` changed between
      crossmem releases).
    """
    if current is None:
        return "would add"
    if current == planned:
        return "already present"
    return "would update"


def _print_dry_run_entry(connector: CLIConnector, server_cmd: str) -> None:
    """Print one diff-form line plus the planned snippet for a connector."""
    name = connector.name()
    planned = connector.mcp_snippet(server_cmd)
    current = connector.current_entry()
    status = _plan_status(current, planned)
    _print(f"  - {name}: {status} (config: {connector.config_path()})")
    # Render the snippet compactly so the user sees what would be written
    # without scrolling through a 5-line pretty-print per connector.
    snippet = json.dumps(planned, sort_keys=True)
    _print(f"      planned entry: {snippet}")


def _run_dry_run(
    connectors: list[CLIConnector] | None,
    home: Path,
) -> InstallResult:
    """Print a diff preview without touching disk or the embedder.

    Walks the connector list (defaulting to :data:`ALL_CONNECTORS`),
    skips ``detect() is False`` entries, and prints one ``would add`` /
    ``would update`` / ``already present`` line per detected connector
    along with the planned MCP snippet. Returns an :class:`InstallResult`
    with ``dry_run=True`` and ``embedding_model=""``.
    """
    _print("Dry-run: previewing crossmem install (no files will be written).")
    _print("Detecting MCP-capable CLIs...")
    cli_objects = _materialise_connectors(connectors)
    detected_names: list[str] = []
    for connector in cli_objects:
        name = connector.name()
        if not connector.detect():
            _print(f"  - {name}: not detected, skipping")
            continue
        _print_dry_run_entry(connector, DEFAULT_SERVER_CMD)
        hint = _restart_hint_line(connector)
        if hint is not None:
            _print(hint)
        detected_names.append(name)
    db_path = home / ".crossmem" / "knowledge.db"
    _print(
        f"Dry-run complete. Would wire up {len(detected_names)} CLI(s); "
        f"would create DB at {db_path}. Re-run without --dry-run to apply."
    )
    return InstallResult(
        detected_clis=detected_names,
        db_path=db_path,
        embedding_model="",
        dry_run=True,
    )


def _run_doctor_preflight(
    doctor_factory: Callable[[], list[CheckResult]],
) -> None:
    """Run ``doctor.run_checks`` (or stub) and abort install on any ``fail``.

    Warnings are printed but do not abort — the user can still install
    even without optional backends. ``ok`` results are silent to keep
    install output focused on actionable content.
    """
    _print("Running doctor preflight checks...")
    results = doctor_factory()
    for result in results:
        if result.status == "warn":
            _print(f"  [warn] {result.name}: {result.detail}")
    failures = [r for r in results if r.status == "fail"]
    if failures:
        raise InstallAbortedError(failures)


def install(
    *,
    connectors: list[CLIConnector] | None = None,
    embedder_factory: Callable[[], object] | None = None,
    doctor_factory: Callable[[], list[CheckResult]] | None = None,
    dry_run: bool = False,
) -> InstallResult:
    """Run the ``crossmem install`` flow and return an :class:`InstallResult`.

    Parameters are dependency-injected for testability:

    * ``connectors`` — explicit list of :class:`CLIConnector` instances.
      ``None`` (the default) uses :data:`ALL_CONNECTORS`.
    * ``embedder_factory`` — zero-arg callable returning an embedder with
      a ``.model_name`` attribute. ``None`` uses :class:`EmbeddingService`.
    * ``doctor_factory`` — zero-arg callable returning a list of
      :class:`CheckResult` (signature of :func:`crossmem.doctor.run_checks`).
      ``None`` uses :func:`crossmem.doctor.run_checks`. Called *before* any
      side-effecting install step; a single ``fail`` raises
      :class:`InstallAbortedError`.
    * ``dry_run`` — when ``True``, print a diff preview per detected
      connector and return early. No ``register()`` is called, no DB is
      created, no embedder is built, and the doctor preflight is
      skipped (a doctor failure should not block a read-only preview).
    """
    home = Path.home()
    if dry_run:
        return _run_dry_run(connectors, home)

    doctor = doctor_factory if doctor_factory is not None else run_checks
    _run_doctor_preflight(doctor)

    _print("Detecting MCP-capable CLIs...")
    cli_objects = _materialise_connectors(connectors)
    detected_clis = _register_detected(cli_objects)

    _print("Initialising knowledge.db...")
    db_path = _ensure_database(home)

    _print(
        "Loading embedding model (~120MB on first run, downloads from "
        "HuggingFace if not cached)..."
    )
    factory = embedder_factory if embedder_factory is not None else EmbeddingService
    embedder = factory()
    model_name = getattr(embedder, "model_name", "unknown")

    _print(
        f"Done. {len(detected_clis)} CLI(s) wired up, "
        f"DB at {db_path}, model: {model_name}."
    )
    return InstallResult(
        detected_clis=detected_clis,
        db_path=db_path,
        embedding_model=str(model_name),
    )
