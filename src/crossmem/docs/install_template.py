"""Render per-CLI install guides from connector metadata.

This module is the single source of truth for ``install/<cli>.md``
files. Real per-CLI guides are produced by :func:`render_install_doc`
(invoked by ``crossmem docs install --cli <name>``) so the headings,
config paths and MCP snippets stay in lockstep with the connector
code — no hand-edited Markdown to drift.

Three inputs feed the renderer:

* :class:`crossmem.connectors.base.CLIConnector` metadata from
  Task 15.1 (``is_gui_app``, :meth:`restart_hint`, :meth:`mcp_snippet`).
* :func:`crossmem.doctor.run_checks` — names of the checks are
  enumerated in the Verify section so the LLM knows what
  ``crossmem doctor`` will assert.
* The current ``crossmem`` package version
  (:func:`importlib.metadata.version`) — recorded as
  ``min_crossmem_version`` in the YAML frontmatter.

The render is deterministic by construction: no timestamps, no
machine-specific paths in the body, stable dict orders. Two
invocations with the same package version produce byte-identical
output (see ``tests/docs/test_render.py::test_render_is_deterministic``).
"""

from __future__ import annotations

import json
from importlib import metadata
from typing import TYPE_CHECKING, Any

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
from crossmem.doctor import run_checks

if TYPE_CHECKING:
    import os

    from crossmem.connectors.base import CLIConnector
    from crossmem.doctor import CheckResult

__all__ = [
    "CONNECTOR_REGISTRY",
    "DEFAULT_SERVER_CMD",
    "MCP_WRAPPER_BY_CLI",
    "UnknownConnectorError",
    "known_cli_names",
    "render_install_doc",
]


# Keep in sync with ``crossmem.installer.DEFAULT_SERVER_CMD``: pipx-installed
# wheels expose a ``crossmem`` console-script that just calls
# ``server.main``, so the published install docs document the user-facing
# form rather than ``python -m crossmem.server`` which is dev-only.
DEFAULT_SERVER_CMD = "crossmem"


# Public registry mapping ``connector.name()`` -> connector class.
# Mirrors :data:`crossmem.installer.ALL_CONNECTORS` but keyed by the same
# string the LLM passes to ``--cli``. Sorted alphabetically so the
# ``crossmem docs install --cli unknown`` error message lists CLIs in
# a stable order without callers having to re-sort.
CONNECTOR_REGISTRY: dict[str, type[CLIConnector]] = {
    "amazonq": AmazonQConnector,
    "claude_code": ClaudeCodeConnector,
    "cline": ClineConnector,
    "continuedev": ContinueDevConnector,
    "cursor": CursorConnector,
    "gemini": GeminiConnector,
    "goose": GooseConnector,
    "kilocode": KiloCodeConnector,
    "opencode": OpenCodeConnector,
    "pi": PiConnector,
    "windsurf": WindsurfConnector,
    "zed": ZedConnector,
}


# Per-format validator command shown in the Troubleshooting section.
# The LLM runs this against the freshly edited config file to confirm
# the merge produced syntactically valid output before debugging
# further. Keyed by ``MCP_WRAPPER_BY_CLI[cli]["format"]``.
#
# JSON uses ``python -m json.tool`` (stdlib, zero deps). YAML uses a
# one-liner against PyYAML — crossmem already requires PyYAML at
# runtime (see ``crossmem.doctor``'s ``module_yaml`` check), so the
# command works on any host where crossmem itself was installed.
# Add new entries here when a new config format appears; never
# hardcode validator strings in the body template.
_VALIDATOR_BY_FORMAT: dict[str, str] = {
    "json": "python -m json.tool < <path>",
    "yaml": 'python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" <path>',  # noqa: E501
}


# Per-CLI MCP wrapper shape. Each entry describes where the snippet
# returned by :meth:`CLIConnector.mcp_snippet` is written inside the
# host config file and which on-disk format that file uses. Used purely
# by the renderer (the connector code itself already knows the key
# layout via ``register()``); we lift the structure out here so the
# generated Markdown shows the LLM exactly what to paste.
#
# Fields:
# * ``format`` — ``"json"`` or ``"yaml"`` (controls the fenced code
#   block language tag and serialisation).
# * ``path`` — tuple of nested keys leading from the file root down to
#   the per-server map. ``("mcpServers",)`` means ``mcpServers``;
#   ``("experimental", "modelContextProtocolServers")`` means a nested
#   two-level key.
MCP_WRAPPER_BY_CLI: dict[str, dict[str, Any]] = {
    "amazonq": {"format": "json", "path": ("mcpServers",)},
    "claude_code": {"format": "json", "path": ("mcpServers",)},
    "cline": {"format": "json", "path": ("mcpServers",)},
    "continuedev": {
        "format": "json",
        "path": ("experimental", "modelContextProtocolServers"),
    },
    "cursor": {"format": "json", "path": ("mcpServers",)},
    "gemini": {"format": "json", "path": ("mcpServers",)},
    "goose": {"format": "yaml", "path": ("extensions",)},
    "kilocode": {"format": "json", "path": ("mcpServers",)},
    "opencode": {"format": "json", "path": ("mcp",)},
    "pi": {"format": "json", "path": ("mcpServers",)},
    "windsurf": {"format": "json", "path": ("mcpServers",)},
    "zed": {"format": "json", "path": ("context_servers",)},
}


class UnknownConnectorError(KeyError):
    """Raised when :func:`render_install_doc` receives an unknown CLI name.

    Carries the requested name and the sorted list of known CLI names
    so the CLI layer can render a user-actionable error.
    """

    def __init__(self, requested: str, known: list[str]) -> None:
        self.requested = requested
        self.known = list(known)
        super().__init__(requested)


def known_cli_names() -> list[str]:
    """Return the sorted list of CLI names accepted by ``--cli``."""
    return sorted(CONNECTOR_REGISTRY)


# ---------------------------------------------------------------------------
# Platform-specific path resolution
# ---------------------------------------------------------------------------


def _windows_friendly(path_str: str) -> str:
    """Normalise a connector-produced path for the Windows row.

    Connectors call ``Path.home()`` which returns the host's home (e.g.
    ``/home/marco`` on Linux). When we simulate ``sys.platform=='win32'``
    the connector still uses the Linux home as the base; for the doc
    we want a stable ``%USERPROFILE%``-style placeholder so the LLM
    can substitute. We swap the host home prefix for ``%USERPROFILE%``
    and convert separators to backslashes.
    """
    home = str(_host_home())
    if path_str.startswith(home):
        path_str = "%USERPROFILE%" + path_str[len(home) :]
    return path_str.replace("/", "\\")


def _unix_friendly(path_str: str, *, kind: str) -> str:
    """Normalise a connector path for the Linux or macOS row.

    Replaces the host's home directory with ``~`` so the rendered doc
    is portable across machines.
    """
    home = str(_host_home())
    if path_str.startswith(home):
        path_str = "~" + path_str[len(home) :]
    # On Linux/macOS hosts the connector already uses forward slashes,
    # but if we render on Windows the connector emits backslashes for
    # the simulated POSIX paths too — normalise them.
    if kind in {"linux", "mac"}:
        path_str = path_str.replace("\\", "/")
    return path_str


def _host_home() -> os.PathLike[str]:
    """Return the host's home directory (lazy import keeps stubs simple)."""
    from pathlib import Path

    return Path.home()


def _config_path_triplet(connector: CLIConnector) -> dict[str, str]:
    """Return ``{linux, mac, win}`` paths for ``connector`` as strings.

    Pure function: never touches ``sys.platform`` or ``os.environ``.
    Calls :meth:`CLIConnector.paths_for_platform` three times with the
    explicit platform / ``appdata`` arguments. Connectors with
    platform-dependent paths override that method; connectors whose
    path is the same on every OS inherit the base default and return
    the runtime path unchanged.
    """
    linux_path = str(connector.paths_for_platform("linux", appdata=None))
    mac_path = str(connector.paths_for_platform("darwin", appdata=None))
    # ``%APPDATA%`` is a stable placeholder, not the host's actual
    # value — the rendered doc must be portable across machines.
    win_path = str(connector.paths_for_platform("win32", appdata=r"%APPDATA%"))
    return {
        "linux": _unix_friendly(linux_path, kind="linux"),
        "mac": _unix_friendly(mac_path, kind="mac"),
        "win": _windows_friendly(win_path),
    }


# ---------------------------------------------------------------------------
# Snippet rendering
# ---------------------------------------------------------------------------


def _build_nested(path: tuple[str, ...], leaf: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``leaf`` under ``{crossmem: leaf}`` then under each path key.

    Example: ``path=("mcpServers",)`` with leaf ``L`` returns
    ``{"mcpServers": {"crossmem": L}}``. With
    ``path=("experimental", "modelContextProtocolServers")`` you get the
    Continue.dev two-level nesting.
    """
    body: dict[str, Any] = {"crossmem": leaf}
    for key in reversed(path):
        body = {key: body}
    return body


def _render_json_snippet(wrapped: dict[str, Any]) -> str:
    return json.dumps(wrapped, indent=2, ensure_ascii=False)


def _render_yaml_snippet(wrapped: dict[str, Any]) -> str:
    """Emit a stable, deterministic YAML representation.

    We avoid the ``yaml`` import path at module load time (one less
    optional dep at import) and instead format the small fixed shape
    Goose uses (``extensions: { crossmem: { type, cmd, args, enabled } }``)
    by hand. The shape is asserted by the connector test suite, so
    drift would surface there.
    """
    return _emit_yaml(wrapped, indent=0)


def _emit_yaml(value: Any, *, indent: int) -> str:
    """Tiny block-style YAML emitter for the snippet subset we use.

    Handles only what Goose's ``mcp_snippet`` produces: nested maps
    whose values are strings, booleans, ints or lists of strings. This
    is intentionally narrow — anything else raises so we notice during
    tests rather than silently emit malformed YAML.
    """
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}\n"
        lines: list[str] = []
        for key, val in value.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                lines.append(_emit_yaml(val, indent=indent + 1).rstrip("\n"))
            elif isinstance(val, list):
                if not val:
                    lines.append(f"{pad}{key}: []")
                else:
                    lines.append(f"{pad}{key}:")
                    for item in val:
                        lines.append(f"{pad}  - {_scalar(item)}")
            else:
                lines.append(f"{pad}{key}: {_scalar(val)}")
        return "\n".join(lines) + "\n"
    raise TypeError(f"Unsupported YAML value type: {type(value).__name__}")


def _scalar(value: Any) -> str:
    """Render a YAML scalar (str/bool/int)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Quote strings that look like booleans, contain colons or
        # start with characters YAML treats specially. The shape we
        # emit is small enough that conservative quoting is fine.
        needs_quote = (
            value in {"true", "false", "yes", "no", "null", ""}
            or ":" in value
            or value.startswith(("-", "?", "*", "&", "!", "|", ">", "%", "@", "`"))
        )
        if needs_quote:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return value
    raise TypeError(f"Unsupported YAML scalar type: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def _crossmem_version() -> str:
    """Return the installed ``crossmem`` package version.

    Falls back to ``"0.0.0"`` only when the package is not installed
    (e.g. when running tests against the source tree without a wheel).
    Using ``importlib.metadata`` keeps the doc tied to the actual
    published version — no second source of truth to drift.
    """
    try:
        return metadata.version("crossmem")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _format_frontmatter(
    *,
    cli_name: str,
    version: str,
    paths: dict[str, str],
    restart_hint: str,
) -> str:
    """Render the YAML frontmatter block with a stable key order.

    Key order is documented by
    :data:`tests.install._helpers.FRONTMATTER_KEYS`. We emit in that
    exact order so :func:`assert_install_doc_schema` sees a clean
    match.
    """
    win_path = paths["win"]
    # If the Windows path contains a backslash, single-quote it so YAML
    # does not interpret the backslash as an escape. The template does
    # the same.
    win_value = f"'{win_path}'" if "\\" in win_path or "%" in win_path else win_path
    lines = [
        "---",
        f"cli: {cli_name}",
        f"min_crossmem_version: {version}",
        f"config_path_linux: {paths['linux']}",
        f"config_path_mac: {paths['mac']}",
        f"config_path_win: {win_value}",
        f"restart_hint: {restart_hint}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _verify_checks_block(checks: list[CheckResult]) -> str:
    """Render the list of doctor check names for the Verify section."""
    lines = [f"- `{check.name}`" for check in checks]
    return "\n".join(lines)


def render_install_doc(
    cli_name: str,
    *,
    version: str | None = None,
    doctor_checks: list[CheckResult] | None = None,
) -> str:
    """Return the complete ``install/<cli>.md`` body for ``cli_name``.

    Parameters
    ----------
    cli_name:
        Key into :data:`CONNECTOR_REGISTRY`. Unknown values raise
        :class:`UnknownConnectorError` carrying the sorted list of
        known names.
    version:
        Override for the ``min_crossmem_version`` frontmatter field.
        Defaults to the installed ``crossmem`` package version via
        :func:`importlib.metadata.version`.
    doctor_checks:
        Override for the list used in the Verify section. Defaults to
        :func:`crossmem.doctor.run_checks` at call time.

    Returns
    -------
    str
        Markdown text ending with a single trailing newline. Identical
        inputs always produce identical output (deterministic by
        construction — no timestamps, stable dict order).
    """
    if cli_name not in CONNECTOR_REGISTRY:
        raise UnknownConnectorError(cli_name, known_cli_names())

    connector_cls = CONNECTOR_REGISTRY[cli_name]
    connector = connector_cls()
    paths = _config_path_triplet(connector)
    snippet_leaf = connector.mcp_snippet(DEFAULT_SERVER_CMD)
    wrapper = MCP_WRAPPER_BY_CLI[cli_name]
    wrapped = _build_nested(wrapper["path"], snippet_leaf)
    fmt = wrapper["format"]
    if fmt == "json":
        snippet_text = _render_json_snippet(wrapped)
        fence_lang = "json"
    else:
        snippet_text = _render_yaml_snippet(wrapped).rstrip("\n")
        fence_lang = "yaml"
    # Format-aware troubleshooting check. ``json.tool`` cannot parse
    # YAML — hardcoding it in the body template would give the LLM a
    # broken command for the YAML-config CLIs (e.g. goose). Look up the
    # validator by format so adding a new format (TOML, ...) is a one-
    # line entry in ``_VALIDATOR_BY_FORMAT`` rather than a template
    # edit.
    validator_cmd = _VALIDATOR_BY_FORMAT[fmt]
    format_label = fmt.upper()

    pkg_version = version if version is not None else _crossmem_version()
    checks = doctor_checks if doctor_checks is not None else run_checks()
    restart_hint = connector.restart_hint()

    frontmatter = _format_frontmatter(
        cli_name=cli_name,
        version=pkg_version,
        paths=paths,
        restart_hint=restart_hint,
    )

    body = _BODY_TEMPLATE.format(
        cli=cli_name,
        config_path_linux=paths["linux"],
        config_path_mac=paths["mac"],
        config_path_win=paths["win"],
        restart_hint=restart_hint,
        snippet_lang=fence_lang,
        snippet=snippet_text,
        doctor_checks=_verify_checks_block(checks),
        config_format_label=format_label,
        validator_cmd=validator_cmd,
    )
    return frontmatter + body


# The body template is kept verbatim so the diff against
# ``install/_template.md`` is small and human-reviewable. Format
# placeholders all have stable values for a given input.
_BODY_TEMPLATE = """\
# Install crossmem for {cli}

This file was generated by `crossmem docs install --cli {cli}` from the
connector code. Do not edit by hand — re-run the command to regenerate.

LLM, read top to bottom and execute the commands literally. Each section
is required. Do not invent steps and do not skip the Verify section.

## Prerequisites

- Python 3.10 or newer on `PATH` (`python --version`).
- `pipx` installed (`pipx --version`). If missing, install per the
  official pipx docs for the host OS.
- The target CLI ({cli}) is already installed and starts cleanly.
- Network access for the first run only — the embedding model
  (`paraphrase-multilingual-MiniLM-L12-v2`, ~300 MB) is downloaded once
  and cached under `~/.crossmem/`.

## Install

```bash
pipx install crossmem
crossmem doctor
```

`crossmem doctor` must exit 0. If it reports a missing dependency,
install that dependency first and re-run the doctor — do not proceed
until it is green.

## Configure MCP

Locate the {cli} MCP config file for the host OS:

- Linux: `{config_path_linux}`
- macOS: `{config_path_mac}`
- Windows: `{config_path_win}`

Back the file up before editing (`cp <path> <path>.bak`). Merge the
following snippet into the existing config — do not replace the whole
file:

```{snippet_lang}
{snippet}
```

## Verify

1. {restart_hint}
2. From inside {cli}, ask the model to call the `query` MCP tool with a
   throwaway string (e.g. `crossmem ping`). The call must return a JSON
   response, even if empty.
3. Re-run `crossmem doctor` — it must still exit 0. The doctor checks
   covered by this verify step are:

{doctor_checks}

If all three checks pass, the install is complete.

## Troubleshooting

- **`crossmem` not found after `pipx install`** — ensure
  `$(pipx environment --value PIPX_BIN_DIR)` is on `PATH` and start a
  new shell.
- **MCP server does not appear in {cli}** — verify the config file path
  for the host OS, confirm the {config_format_label} is valid
  (`{validator_cmd}`), and check that {cli} was fully
  restarted.
- **Embedding model download fails** — re-run `crossmem doctor`; it
  retries the download with a progress bar. On corporate networks set
  `HF_HUB_OFFLINE=0` and ensure `huggingface.co` is reachable.
- **`crossmem doctor` reports schema-version drift** — run
  `pipx upgrade crossmem` followed by `crossmem doctor` again.
"""
