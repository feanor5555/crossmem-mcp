"""Tests for ``crossmem docs install`` rendering.

The :mod:`crossmem.docs.install_template` module renders a per-CLI
``install/<cli>.md`` from three sources: the connector's class-level
metadata (Task 15.1), :func:`crossmem.doctor.run_checks` (Verify
section) and the current ``crossmem`` package version.

These tests pin the renderer to the schema enforced by
``tests/install/_helpers.assert_install_doc_schema`` and snapshot
the output per connector. Snapshots live next to this file in
``snapshots/<cli>.md`` and are byte-identical to what the CLI
writes — regenerate them via the helper at the bottom of this
module if a renderer change is intentional.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crossmem.docs import install_template
from crossmem.docs.install_template import (
    CONNECTOR_REGISTRY,
    render_install_doc,
)
from tests.install._helpers import assert_install_doc_schema

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

ALL_CLI_NAMES: tuple[str, ...] = tuple(sorted(CONNECTOR_REGISTRY))


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_has_twelve_connectors() -> None:
    """Renderer must know about all 12 shipped connectors."""
    assert len(CONNECTOR_REGISTRY) == 12


def test_registry_keys_match_connector_names() -> None:
    """Each registry key must equal the connector's own ``name()`` value."""
    for key, cls in CONNECTOR_REGISTRY.items():
        assert cls().name() == key


# ---------------------------------------------------------------------------
# Snapshot tests (one per connector)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cli_name", ALL_CLI_NAMES)
def test_snapshot_matches(cli_name: str) -> None:
    """Rendered output is byte-identical to the committed snapshot."""
    rendered = render_install_doc(cli_name)
    snapshot_path = SNAPSHOT_DIR / f"{cli_name}.md"
    assert snapshot_path.is_file(), (
        f"Missing snapshot for {cli_name}: {snapshot_path}. "
        f"Regenerate via tests/docs/_regen_snapshots.py."
    )
    expected = snapshot_path.read_text(encoding="utf-8")
    assert rendered == expected, (
        f"Snapshot drift for {cli_name}. "
        f"Re-run tests/docs/_regen_snapshots.py if the change is intentional."
    )


@pytest.mark.parametrize("cli_name", ALL_CLI_NAMES)
def test_snapshot_matches_install_doc_schema(cli_name: str, tmp_path: Path) -> None:
    """Each rendered file must satisfy the shared install-doc schema."""
    rendered = render_install_doc(cli_name)
    target = tmp_path / f"{cli_name}.md"
    target.write_text(rendered, encoding="utf-8")
    assert_install_doc_schema(target)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cli_name", ALL_CLI_NAMES)
def test_render_is_deterministic(cli_name: str) -> None:
    """Two renders with identical inputs must yield byte-identical output."""
    first = render_install_doc(cli_name)
    second = render_install_doc(cli_name)
    assert first == second


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_docs_install_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``crossmem docs install --cli <name>`` writes the doc to stdout."""
    from crossmem import cli as cli_module

    exit_code = cli_module.main(["docs", "install", "--cli", "claude_code"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == render_install_doc("claude_code")


def test_cli_docs_install_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output PATH`` writes the doc to PATH and emits no body on stdout."""
    from crossmem import cli as cli_module

    target = tmp_path / "claude_code.md"
    exit_code = cli_module.main(
        ["docs", "install", "--cli", "claude_code", "--output", str(target)]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    # Body must not leak to stdout when --output is set.
    assert "## Prerequisites" not in captured.out
    assert target.read_text(encoding="utf-8") == render_install_doc("claude_code")


def test_cli_docs_install_unknown_cli_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown CLI: exit 2, list known CLIs on stderr."""
    from crossmem import cli as cli_module

    exit_code = cli_module.main(["docs", "install", "--cli", "not-a-cli"])
    captured = capsys.readouterr()
    assert exit_code == 2
    # stderr must mention every known CLI so the LLM can self-correct.
    for known in ALL_CLI_NAMES:
        assert known in captured.err


def test_cli_docs_install_unknown_cli_writes_nothing(
    tmp_path: Path,
) -> None:
    """Unknown CLI must not touch ``--output`` even if it was provided."""
    from crossmem import cli as cli_module

    target = tmp_path / "should-not-exist.md"
    exit_code = cli_module.main(
        ["docs", "install", "--cli", "bogus", "--output", str(target)]
    )
    assert exit_code == 2
    assert not target.exists()


# ---------------------------------------------------------------------------
# Internal helpers (exposed for snapshot regeneration)
# ---------------------------------------------------------------------------


def test_module_exposes_render_function() -> None:
    """Public surface: ``render_install_doc`` is callable and returns str."""
    assert callable(install_template.render_install_doc)
    result = install_template.render_install_doc("claude_code")
    assert isinstance(result, str)
    assert result.endswith("\n")
