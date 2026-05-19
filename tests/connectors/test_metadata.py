"""Cross-cutting metadata tests for every shipped CLIConnector.

Task 15.1 introduces three pieces of LLM-install metadata on
:class:`crossmem.connectors.base.CLIConnector`:

* ``is_gui_app`` — class attribute, must match the GUI/CLI grouping
  declared in TODO.md.
* ``restart_hint()`` — non-empty human-readable string.
* ``mcp_snippet(server_cmd)`` — dict mirroring the real per-CLI
  schema the connector's ``register()`` writes. The minimum contract
  is exactly one of ``command``/``cmd`` (str) plus ``args``
  (``list[str]``); per-connector accuracy tests below pin the exact
  shape for connectors that deviate from the common default.

These tests assert the contract across every concrete connector and
guard against drift in the GUI/CLI grouping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossmem.connectors.base import CLIConnector

GUI_CONNECTORS: list[Callable[[], CLIConnector]] = [
    CursorConnector,
    ClineConnector,
    KiloCodeConnector,
    WindsurfConnector,
    ZedConnector,
    ContinueDevConnector,
]

CLI_CONNECTORS: list[Callable[[], CLIConnector]] = [
    ClaudeCodeConnector,
    OpenCodeConnector,
    GooseConnector,
    GeminiConnector,
    AmazonQConnector,
    PiConnector,
]

ALL_FACTORIES: list[Callable[[], CLIConnector]] = GUI_CONNECTORS + CLI_CONNECTORS


def test_we_cover_all_twelve_connectors() -> None:
    """Sanity: the GUI + CLI lists together cover all 12 shipped connectors."""
    assert len(ALL_FACTORIES) == 12
    names = {factory().name() for factory in ALL_FACTORIES}
    assert len(names) == 12, "connector names must be unique"


@pytest.mark.parametrize("factory", GUI_CONNECTORS, ids=lambda f: f.__name__)
def test_gui_connectors_marked_is_gui_app_true(
    factory: Callable[[], CLIConnector],
) -> None:
    connector = factory()
    assert connector.is_gui_app is True, (
        f"{factory.__name__} should be marked as GUI app"
    )


@pytest.mark.parametrize("factory", CLI_CONNECTORS, ids=lambda f: f.__name__)
def test_cli_connectors_marked_is_gui_app_false(
    factory: Callable[[], CLIConnector],
) -> None:
    connector = factory()
    assert connector.is_gui_app is False, (
        f"{factory.__name__} should be marked as CLI-driven (is_gui_app=False)"
    )


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_restart_hint_is_non_empty_string(
    factory: Callable[[], CLIConnector],
) -> None:
    hint = factory().restart_hint()
    assert isinstance(hint, str)
    assert hint.strip(), f"{factory.__name__}.restart_hint() must not be empty"


@pytest.mark.parametrize("factory", GUI_CONNECTORS, ids=lambda f: f.__name__)
def test_gui_restart_hint_mentions_restart(
    factory: Callable[[], CLIConnector],
) -> None:
    """GUI hints should clearly tell the user to relaunch the app."""
    hint = factory().restart_hint().lower()
    assert any(kw in hint for kw in ("restart", "relaunch", "quit", "reload"))


# --- mcp_snippet contract (applies to all connectors) -----------------


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_mcp_snippet_is_non_empty_dict(
    factory: Callable[[], CLIConnector],
) -> None:
    """Every snippet is a non-empty dict (some payload is present)."""
    snippet = factory().mcp_snippet("python -m crossmem.server")
    assert isinstance(snippet, dict), f"{factory.__name__}: snippet must be a dict"
    assert snippet, f"{factory.__name__}: snippet must not be empty"


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_mcp_snippet_has_command_key(
    factory: Callable[[], CLIConnector],
) -> None:
    """Each snippet exposes a non-empty command under ``command`` or ``cmd``.

    The CLI parses the entry to spawn ``crossmem``, so the executable
    name must be present in exactly one of the two conventional keys.
    """
    snippet = factory().mcp_snippet("python -m crossmem.server")
    cmd_keys = [k for k in ("command", "cmd") if k in snippet]
    assert cmd_keys, f"{factory.__name__}: snippet missing both 'command' and 'cmd'"
    assert len(cmd_keys) == 1, (
        f"{factory.__name__}: snippet has both 'command' and 'cmd'; "
        "pick one to match the CLI's real schema"
    )
    value = snippet[cmd_keys[0]]
    assert isinstance(value, str) and value.strip(), (
        f"{factory.__name__}: snippet[{cmd_keys[0]!r}] must be a non-empty str"
    )


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_mcp_snippet_args_is_list_of_str(
    factory: Callable[[], CLIConnector],
) -> None:
    """``args`` must be a ``list[str]`` (possibly empty for single-token cmds)."""
    snippet = factory().mcp_snippet("python -m crossmem.server")
    assert "args" in snippet, f"{factory.__name__}: snippet missing 'args'"
    args = snippet["args"]
    assert isinstance(args, list), f"{factory.__name__}: snippet['args'] not a list"
    assert all(isinstance(a, str) for a in args), (
        f"{factory.__name__}: snippet['args'] must contain only strings"
    )


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_mcp_snippet_args_non_empty_for_multi_token_cmd(
    factory: Callable[[], CLIConnector],
) -> None:
    """When ``server_cmd`` carries arguments, the snippet must surface them.

    Either the connector splits via ``shlex`` (default + Goose) and
    ``args`` carries the trailing tokens, or it stores the full
    command string elsewhere — but in no case may a multi-token
    ``server_cmd`` collapse silently into ``command``/``cmd`` alone
    with empty ``args``: install docs would then render an unusable
    entry.
    """
    snippet = factory().mcp_snippet("python -m crossmem.server")
    assert snippet["args"], (
        f"{factory.__name__}: multi-token server_cmd must yield non-empty args; "
        f"got {snippet!r}"
    )


@pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
def test_mcp_snippet_handles_single_word_command(
    factory: Callable[[], CLIConnector],
) -> None:
    """A bare ``crossmem`` command still yields a usable snippet.

    ``args`` may legitimately be empty when no arguments are passed;
    we only require the executable to land in the connector's
    command key.
    """
    snippet = factory().mcp_snippet("crossmem")
    cmd_keys = [k for k in ("command", "cmd") if k in snippet]
    assert cmd_keys, f"{factory.__name__}: snippet missing command key"
    assert snippet[cmd_keys[0]] == "crossmem"
    assert "args" in snippet
    assert isinstance(snippet["args"], list)


# --- Per-connector accuracy: snippet matches real on-disk schema ------


def test_goose_mcp_snippet_matches_extension_entry() -> None:
    """Goose's snippet equals the YAML extension entry ``register`` writes."""
    snippet = GooseConnector().mcp_snippet("python -m crossmem.server")
    assert snippet == {
        "type": "stdio",
        "cmd": "python",
        "args": ["-m", "crossmem.server"],
        "enabled": True,
    }


def test_goose_mcp_snippet_single_word_keeps_schema() -> None:
    """Bare command still produces the four Goose-specific keys."""
    snippet = GooseConnector().mcp_snippet("crossmem")
    assert snippet == {
        "type": "stdio",
        "cmd": "crossmem",
        "args": [],
        "enabled": True,
    }


@pytest.mark.parametrize(
    "factory",
    [f for f in ALL_FACTORIES if f is not GooseConnector],
    ids=lambda f: f.__name__,
)
def test_default_mcp_snippet_shape(
    factory: Callable[[], CLIConnector],
) -> None:
    """Non-Goose connectors use the common ``{command, args, env}`` shape."""
    snippet = factory().mcp_snippet("python -m crossmem.server")
    assert snippet == {
        "command": "python",
        "args": ["-m", "crossmem.server"],
        "env": {},
    }
