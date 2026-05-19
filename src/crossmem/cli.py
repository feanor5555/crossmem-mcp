"""Command-line entry point for ``crossmem``.

The CLI exposes the following subcommands plus a default no-arg behaviour:

* ``crossmem`` (no args) — start the MCP server over stdio (the way MCP
  clients spawn the server).
* ``crossmem doctor [--json]`` — preflight checks for the runtime
  environment.
* ``crossmem configure --backend ...`` — switch the active vector backend.
* ``crossmem install`` — auto-detect MCP-capable CLIs, wire them up,
  initialise the knowledge DB and warm the embedding model.
* ``crossmem export --path P [--format zip|jsonl]`` — dump the knowledge
  base to a portable file.
* ``crossmem import --path P`` — restore a previously exported file.
* ``crossmem trash list|restore --id X|empty [--ttl-days N]`` — inspect,
  restore from, or prune the soft-delete trash.

The CLI is intentionally minimal: no third-party dependencies, no color
library — just :mod:`argparse` and a tiny ANSI helper that activates only
when stdout is a TTY.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from crossmem import installer
from crossmem.cleanup import empty_trash, list_trash, restore_from_trash
from crossmem.configure import (
    SUPPORTED_BACKENDS,
    BackendConfigError,
    _default_sqlite_path,
    build_backend,
    load_config,
)
from crossmem.configure import configure as configure_backend
from crossmem.connectors_status import (
    format_status_table,
    gather_connector_status,
)
from crossmem.core.embedding import EmbeddingService
from crossmem.core.store import KnowledgeStore
from crossmem.docs.install_template import (
    UnknownConnectorError,
    render_install_doc,
)
from crossmem.doctor import CheckResult, build_support_info, run_checks
from crossmem.installer import InstallAbortedError, install
from crossmem.mcp_ping import default_server_command, ping
from crossmem.server import main as server_main
from crossmem.uninstaller import (
    UnknownConnectorError as UninstallUnknownConnectorError,
)
from crossmem.uninstaller import uninstall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crossmem.connectors.base import CLIConnector

__all__ = [
    "build_default_store",
    "configure_backend",
    "empty_trash",
    "list_trash",
    "main",
    "ping",
    "restore_from_trash",
]

# ANSI color codes. We deliberately hard-code these rather than depend on a
# library like ``colorama`` (Windows terminals support these natively since
# Windows 10, and CI/log capture is non-TTY anyway).
_ANSI_GREEN = "\x1b[32m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"

_STATUS_COLORS = {
    "ok": _ANSI_GREEN,
    "warn": _ANSI_YELLOW,
    "fail": _ANSI_RED,
}

_EXPORT_FORMATS = ("zip", "jsonl")

# Schema version of the ``doctor --json`` payload. Bumped only on breaking
# changes; additive evolution (new checks, new top-level keys) leaves it as-is.
# The corresponding JSON Schema lives at ``schemas/doctor.json`` in the repo.
DOCTOR_JSON_SCHEMA_VERSION = "1"


def _format_marker(status: str, *, color: bool) -> str:
    """Return the bracketed status marker, optionally ANSI-colored."""
    marker = f"[{status}]"
    if not color:
        return marker
    return f"{_STATUS_COLORS[status]}{marker}{_ANSI_RESET}"


def _print_human(results: Sequence[CheckResult]) -> None:
    """Print results one per line and a final summary."""
    color = sys.stdout.isatty()
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for result in results:
        counts[result.status] += 1
        marker = _format_marker(result.status, color=color)
        # Pad to a stable width so names line up. ``[warn]`` is the longest.
        print(f"{marker:<6} {result.name}: {result.detail}")
    print(f"{counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail")


def _print_json(results: Sequence[CheckResult]) -> None:
    """Print results as a stable JSON document.

    Shape is documented by ``schemas/doctor.json`` (JSON Schema Draft 2020-12):

    .. code-block:: json

        {
          "version": "1",
          "checks": [{"name": "...", "status": "ok|warn|fail", "detail": "..."}],
          "summary": {"ok": N, "warn": M, "fail": K, "total": N+M+K},
          "support": {"issues_url": "...", "docs_url": "...", "version": "..."}
        }

    Existing keys and check ``name`` values are stable across releases; new
    checks and new top-level keys may be added (additive evolution). The
    ``support`` block (task 19.1) is one such additive key — referenced by
    ``skills/crossmem-install/SKILL.md`` so LLM install flows can surface
    the issue tracker on failure.
    """
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for result in results:
        counts[result.status] += 1
    payload = {
        "version": DOCTOR_JSON_SCHEMA_VERSION,
        "checks": [asdict(r) for r in results],
        "summary": {
            "ok": counts["ok"],
            "warn": counts["warn"],
            "fail": counts["fail"],
            "total": len(results),
        },
        "support": build_support_info(),
    }
    print(json.dumps(payload))


def build_default_store() -> KnowledgeStore:
    """Construct a :class:`KnowledgeStore` for the backend named in config.

    Reads ``~/.crossmem/config.toml`` (written by ``crossmem configure``)
    and dispatches via :func:`crossmem.configure.build_backend`. SQLite is
    routed through :func:`crossmem.configure._default_sqlite_path` so the
    CLI and the MCP server agree on the on-disk DB location (including the
    ``CROSSMEM_DB_PATH`` override used by tests).
    """
    backend = build_backend(load_config(), sqlite_path=_default_sqlite_path())
    embedder = EmbeddingService()
    return KnowledgeStore(backend, embedder)


def _cmd_doctor(args: argparse.Namespace) -> int:
    results = run_checks()
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
    return 1 if any(r.status == "fail" for r in results) else 0


def _cmd_configure(args: argparse.Namespace) -> int:
    try:
        result = configure_backend(
            backend=args.backend,
            url=args.url,
            api_key=args.api_key,
            migrate=args.migrate,
        )
    except NotImplementedError as exc:
        # Migration runs before config.toml is written, so a raised
        # NotImplementedError leaves the previous backend active. Surface
        # the manual fallback so the user can finish the data move
        # themselves — the export step still runs against the data-holding
        # backend.
        print(f"Backend switch not applied: {exc}", file=sys.stderr)
        return 1
    print(
        f"Backend set to {result.new_backend} "
        f"(was {result.previous_backend}). Config: {result.path}"
    )
    if result.migrated:
        print(f"Migrated data from {result.previous_backend} to {result.new_backend}.")
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        result = install(dry_run=dry_run)
    except InstallAbortedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result.dry_run:
        print(
            f"Dry-run: would wire up {len(result.detected_clis)} CLI(s) "
            f"({', '.join(result.detected_clis) or 'none detected'}); "
            f"would create DB at {result.db_path}. "
            "Re-run without --dry-run to apply."
        )
        return 0
    print(
        f"Installed: {len(result.detected_clis)} CLI(s) "
        f"({', '.join(result.detected_clis) or 'none detected'}), "
        f"DB at {result.db_path}, model: {result.embedding_model}."
    )
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """Roll back ``crossmem install``.

    Removes the ``crossmem`` MCP entry from every detected CLI config
    (or just one when ``--cli NAME`` is given). ``--purge`` plus
    ``--yes`` additionally deletes ``~/.crossmem``. An unknown
    ``--cli`` value exits 2 (argparse-style usage error) and lists
    every known CLI on stderr so the caller can correct the name.
    """
    try:
        result = uninstall(
            cli=args.cli,
            purge=bool(args.purge),
            confirm=bool(args.yes),
        )
    except UninstallUnknownConnectorError as exc:
        print(
            f"crossmem uninstall: unknown --cli {exc.requested!r}. "
            f"Known CLIs: {', '.join(exc.known)}.",
            file=sys.stderr,
        )
        return 2
    summary = (
        f"Uninstalled crossmem from {len(result.unregistered_clis)} CLI(s) "
        f"({', '.join(result.unregistered_clis) or 'none detected'})."
    )
    if result.purged_home:
        summary += f" Purged {result.home_path}."
    elif args.purge and not args.yes:
        summary += (
            f" Skipped purge of {result.home_path}: "
            "re-run with --purge --yes to remove it."
        )
    print(summary)
    return 0


def _cmd_docs_install(args: argparse.Namespace) -> int:
    """Render ``install/<cli>.md`` for the requested CLI.

    Output destination follows POSIX-style defaults: ``--output PATH``
    writes to the file (no body on stdout), and without ``--output``
    the doc goes to stdout so it can be piped or redirected. Choosing
    stdout over a hard-coded ``install/<cli>.md`` keeps the command
    repo-agnostic — callers explicitly opt into the on-disk layout.

    An unknown ``--cli`` value exits 2 (argparse-style usage error)
    and lists every known CLI on stderr so the LLM caller can pick
    the right one without a round-trip.
    """
    try:
        rendered = render_install_doc(args.cli)
    except UnknownConnectorError as exc:
        known = ", ".join(exc.known)
        print(
            f"crossmem docs install: unknown --cli {exc.requested!r}. "
            f"Known CLIs: {known}.",
            file=sys.stderr,
        )
        return 2
    if args.output is not None:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        return 0
    sys.stdout.write(rendered)
    return 0


def _cmd_mcp_ping(_args: argparse.Namespace) -> int:
    """Probe the local MCP server over stdio (post-install self-test).

    Spawns ``python -m crossmem.server`` as an ephemeral subprocess,
    exchanges ``initialize`` + ``tools/list``, and kills it. Prints
    ``ok`` + tool names on success (exit 0) or ``fail`` + a one-line
    reason on failure (exit 1). The fail path writes to stderr so a
    caller piping stdout (``crossmem mcp ping > tools.txt``) still
    sees the error.
    """
    result = ping(default_server_command())
    if result.ok:
        tool_list = ", ".join(result.tools) if result.tools else "<none>"
        print(f"ok ({len(result.tools)} tools: {tool_list})")
        return 0
    reason = result.error or "unknown error"
    print(f"fail: {reason}", file=sys.stderr)
    return 1


def _status_connectors_factory() -> list[CLIConnector]:
    """Return one instance of every registered :class:`CLIConnector`.

    Delegates to :func:`crossmem.installer.instantiate_connectors` so the
    status report and the installer always agree on the connector set.
    Defined as a module-level function (rather than inline in the
    subcommand) so tests can monkeypatch it to inject controlled
    fixtures without touching real CLI configs.
    """
    return installer.instantiate_connectors()


def _cmd_status(args: argparse.Namespace) -> int:
    """Print the connector status table for ``crossmem status --connectors``.

    ``--connectors`` is currently the only report under ``status``; the
    parser already enforces presence of the flag so we get the table
    unconditionally here. Returns 0 — the report is informational and
    a "registered=no" row is not an error condition.
    """
    if not args.connectors:
        # Belt-and-braces: parser ``--connectors`` is required but we
        # leave this branch in for future flags that change the report.
        return 2
    connectors = _status_connectors_factory()
    rows = gather_connector_status(connectors)
    sys.stdout.write(format_status_table(rows))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    try:
        store = build_default_store()
    except BackendConfigError as exc:
        print(f"crossmem export: {exc}", file=sys.stderr)
        return 1
    target = Path(args.path)
    count = store.export(target, format=args.format)
    print(f"Exported {count} document(s) to {target}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        store = build_default_store()
    except BackendConfigError as exc:
        print(f"crossmem import: {exc}", file=sys.stderr)
        return 1
    source = Path(args.path)
    try:
        count = store.import_data(source)
    except ValueError as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {count} document(s) from {source}")
    return 0


def _cmd_trash_list(_args: argparse.Namespace) -> int:
    """Print the trash inventory: ``doc_id  deleted_at  source_url  title``.

    Empty trash prints a one-line "0 entries" hint instead of nothing so
    LLM callers piping the output can detect the empty case without
    parsing zero bytes. Returns 0 unconditionally — listing is read-only
    and an empty trash is not an error.
    """
    entries = list_trash()
    if not entries:
        print("0 entries (trash is empty)")
        return 0
    for entry in entries:
        print(f"{entry.doc_id}\t{entry.deleted_at}\t{entry.source_url}\t{entry.title}")
    print(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
    return 0


def _cmd_trash_restore(args: argparse.Namespace) -> int:
    """Re-insert ``--id`` from trash. Unknown id -> exit 1 with stderr error."""
    try:
        store = build_default_store()
    except BackendConfigError as exc:
        print(f"crossmem trash restore: {exc}", file=sys.stderr)
        return 1
    try:
        doc = restore_from_trash(store, args.id)
    except ValueError as exc:
        print(f"crossmem trash restore: {exc}", file=sys.stderr)
        return 1
    print(f"Restored {doc.id} from trash.")
    return 0


def _cmd_trash_empty(args: argparse.Namespace) -> int:
    """Drop trash entries older than ``--ttl-days`` (default 30, 0 = wipe)."""
    removed = empty_trash(ttl_days=args.ttl_days)
    plural = "entry" if removed == 1 else "entries"
    print(f"Emptied {removed} {plural} from trash (ttl_days={args.ttl_days}).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossmem",
        description=(
            "crossmem - portable knowledge database for AI coding CLIs. "
            "Run without arguments to start the MCP server."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = subparsers.add_parser(
        "doctor",
        help="run preflight checks and report environment health",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit results as a JSON array (no color, no summary)",
    )
    doctor.set_defaults(func=_cmd_doctor)

    cfg = subparsers.add_parser(
        "configure",
        help="select the active backend (sqlite/chroma/qdrant)",
    )
    cfg.add_argument(
        "--backend",
        required=True,
        choices=SUPPORTED_BACKENDS,
        help="backend to activate",
    )
    cfg.add_argument("--url", default=None, help="backend URL (chroma/qdrant)")
    cfg.add_argument("--api-key", default=None, help="API key for the backend")
    cfg.add_argument(
        "--migrate",
        action="store_true",
        help="migrate existing data without prompting",
    )
    cfg.set_defaults(func=_cmd_configure)

    inst = subparsers.add_parser(
        "install",
        help="detect MCP-capable CLIs, wire them up, initialise the DB",
    )
    inst.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "preview the install: print a diff per detected connector "
            "without writing any config file, backup, or DB."
        ),
    )
    inst.set_defaults(func=_cmd_install)

    uninst = subparsers.add_parser(
        "uninstall",
        help="remove crossmem MCP wiring from detected CLI configs",
    )
    uninst.add_argument(
        "--cli",
        default=None,
        help=(
            "only uninstall from this connector (e.g. claude_code). "
            "Default: roll back every detected CLI."
        ),
    )
    uninst.add_argument(
        "--purge",
        action="store_true",
        help=(
            "additionally remove the on-disk ~/.crossmem directory "
            "(knowledge DB, trash). Requires --yes to take effect."
        ),
    )
    uninst.add_argument(
        "--yes",
        action="store_true",
        help="confirm destructive operations such as --purge",
    )
    uninst.set_defaults(func=_cmd_uninstall)

    status = subparsers.add_parser(
        "status",
        help="report connector wiring status (detected/registered/backups)",
    )
    status.add_argument(
        "--connectors",
        action="store_true",
        required=True,
        help=(
            "print one row per registered connector with detect/register "
            "state, backup count and full config_path"
        ),
    )
    status.set_defaults(func=_cmd_status)

    docs = subparsers.add_parser(
        "docs",
        help="generate install/<cli>.md guides from connector metadata",
    )
    docs_sub = docs.add_subparsers(dest="docs_command", metavar="<docs-command>")
    docs_install = docs_sub.add_parser(
        "install",
        help="render install/<cli>.md for a single CLI from connector code",
    )
    docs_install.add_argument(
        "--cli",
        required=True,
        # ``choices`` would surface as an argparse usage error (exit 2),
        # which is what we want — but argparse prints its own message
        # and skips our own list-the-known-CLIs branch. We validate
        # inside ``_cmd_docs_install`` instead so the error message can
        # name every CLI on its own terms (and stay greppable by tests).
        help="name of the connector to render (e.g. claude_code, goose)",
    )
    docs_install.add_argument(
        "--output",
        default=None,
        help=(
            "destination file (default: write to stdout). The directory "
            "is created if it does not exist."
        ),
    )
    docs_install.set_defaults(func=_cmd_docs_install)

    def _docs_dispatch(args: argparse.Namespace) -> int:
        if getattr(args, "docs_command", None) is None:
            docs.print_help(sys.stderr)
            return 2
        return args.func(args)

    docs.set_defaults(func=_docs_dispatch)

    mcp = subparsers.add_parser(
        "mcp",
        help="MCP server self-tests (ping the local server over stdio)",
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_command", metavar="<mcp-command>")
    mcp_ping = mcp_sub.add_parser(
        "ping",
        help="spawn the local MCP server and verify it answers tools/list",
    )
    mcp_ping.set_defaults(func=_cmd_mcp_ping)

    def _mcp_dispatch(args: argparse.Namespace) -> int:
        if getattr(args, "mcp_command", None) is None:
            mcp.print_help(sys.stderr)
            return 2
        return args.func(args)

    mcp.set_defaults(func=_mcp_dispatch)

    exp = subparsers.add_parser(
        "export",
        help="export the knowledge base to a JSONL or ZIP file",
    )
    exp.add_argument("--path", required=True, help="destination file path")
    exp.add_argument(
        "--format",
        choices=_EXPORT_FORMATS,
        default="zip",
        help="export format (default: zip)",
    )
    exp.set_defaults(func=_cmd_export)

    imp = subparsers.add_parser(
        "import",
        help="import a previously exported JSONL or ZIP file",
    )
    imp.add_argument("--path", required=True, help="source file path")
    imp.set_defaults(func=_cmd_import)

    trash = subparsers.add_parser(
        "trash",
        help="inspect, restore from, or empty the soft-delete trash",
    )
    trash_sub = trash.add_subparsers(dest="trash_command", metavar="<trash-command>")

    trash_list = trash_sub.add_parser(
        "list",
        help="list current trash entries (id, deleted_at, source_url, title)",
    )
    trash_list.set_defaults(func=_cmd_trash_list)

    trash_restore = trash_sub.add_parser(
        "restore",
        help="restore a single trashed document by id",
    )
    trash_restore.add_argument(
        "--id",
        required=True,
        dest="id",
        help="doc_id of the trashed document to restore",
    )
    trash_restore.set_defaults(func=_cmd_trash_restore)

    trash_empty = trash_sub.add_parser(
        "empty",
        help="drop trash entries older than --ttl-days (default 30; 0 wipes all)",
    )
    trash_empty.add_argument(
        "--ttl-days",
        type=int,
        default=30,
        dest="ttl_days",
        help="entries older than this many days are removed (default 30; 0=wipe)",
    )
    trash_empty.set_defaults(func=_cmd_trash_empty)

    def _trash_dispatch(args: argparse.Namespace) -> int:
        if getattr(args, "trash_command", None) is None:
            trash.print_help(sys.stderr)
            return 2
        return args.func(args)

    trash.set_defaults(func=_trash_dispatch)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    With no arguments the MCP server is started over stdio (this is how
    MCP clients spawn the server). With a subcommand, dispatch to its
    handler. ``--help`` and unknown subcommands are handled by argparse
    (which calls ``sys.exit`` itself).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        server_main()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - exercised via console_scripts
    sys.exit(main())
