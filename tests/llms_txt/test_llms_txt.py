"""Validate the repo-root ``llms.txt`` against the llmstxt.org spec.

The llmstxt.org specification requires a Markdown file at the project
root that an LLM (or its harness) can fetch to discover the project's
authoritative entry points. The minimal shape is:

1. **One** H1 (``# <project>``) on the first non-blank line.
2. A blockquote (``> ...``) directly after the H1 with a short
   summary of the project.
3. Zero or more ``## <Section>`` blocks containing Markdown links.

For crossmem the ``## Installation`` section is mandatory and lists a
Markdown link per supported CLI pointing at the matching
``install/<cli>.md`` guide (Task 15.5). This test enforces both the
structural rules above and the link-target rule (every linked file
must actually exist in the repo).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LLMS_TXT = REPO_ROOT / "llms.txt"
INSTALL_DIR = REPO_ROOT / "install"

# Files in ``install/`` that are not per-CLI guides and therefore not
# required to appear as links in the Installation section.
_NON_CLI_FILES: frozenset[str] = frozenset({"_template.md", "README.md"})

_MARKDOWN_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")


def _read_llms_txt() -> str:
    assert LLMS_TXT.is_file(), f"missing repo-root file: {LLMS_TXT}"
    return LLMS_TXT.read_text(encoding="utf-8")


def _non_blank_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def test_llms_txt_exists_at_repo_root() -> None:
    assert LLMS_TXT.is_file(), (
        f"llms.txt must live at the repo root for LLM discovery: {LLMS_TXT}"
    )


def test_llms_txt_has_exactly_one_h1() -> None:
    """Spec: exactly one H1 at the top of the document."""
    text = _read_llms_txt()
    h1_lines = [
        line
        for line in text.splitlines()
        if line.startswith("# ") and not line.startswith("## ")
    ]
    assert len(h1_lines) == 1, (
        f"llms.txt must contain exactly one H1, found {len(h1_lines)}: {h1_lines}"
    )
    # And it must be the very first non-blank line.
    first = _non_blank_lines(text)[0]
    assert first.startswith("# "), (
        f"llms.txt: first non-blank line must be the H1, got {first!r}"
    )


def test_llms_txt_h1_is_crossmem() -> None:
    """The H1 names the project."""
    text = _read_llms_txt()
    first = _non_blank_lines(text)[0]
    assert first.strip() == "# crossmem", (
        f"llms.txt H1 must be '# crossmem', got {first!r}"
    )


def test_llms_txt_blockquote_follows_h1() -> None:
    """Spec: a blockquote summary appears directly after the H1."""
    text = _read_llms_txt()
    non_blank = _non_blank_lines(text)
    assert len(non_blank) >= 2, "llms.txt must contain H1 and a blockquote summary"
    assert non_blank[1].lstrip().startswith("> "), (
        "llms.txt: the line after the H1 must be a Markdown blockquote "
        f"('> ...'), got {non_blank[1]!r}"
    )


def _section_body(text: str, heading: str) -> str:
    """Return the body of ``## <heading>`` up to the next ``## `` or EOF."""
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$(?P<body>.*?)(?=^## |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    assert match is not None, f"llms.txt: missing '## {heading}' section"
    return match.group("body")


def test_installation_section_exists() -> None:
    """The mandatory ``## Installation`` section is present."""
    text = _read_llms_txt()
    _section_body(text, "Installation")


def _installation_links(text: str) -> list[tuple[str, str]]:
    body = _section_body(text, "Installation")
    return [
        (m.group("text"), m.group("target")) for m in _MARKDOWN_LINK_RE.finditer(body)
    ]


def test_installation_section_has_markdown_links() -> None:
    """At least one Markdown link sits inside ``## Installation``."""
    text = _read_llms_txt()
    links = _installation_links(text)
    assert links, (
        "llms.txt: '## Installation' must contain Markdown links "
        "pointing at install/<cli>.md guides"
    )


def _expected_install_targets() -> set[str]:
    return {
        f"install/{p.name}"
        for p in INSTALL_DIR.glob("*.md")
        if p.name not in _NON_CLI_FILES
    }


def test_installation_links_cover_all_per_cli_guides() -> None:
    """Every ``install/<cli>.md`` guide is linked from llms.txt."""
    text = _read_llms_txt()
    targets = {target for _, target in _installation_links(text)}
    # Restrict to install/ links so future sections (e.g. ``Concepts``)
    # that link elsewhere don't pollute the comparison.
    install_targets = {t for t in targets if t.startswith("install/")}
    expected = _expected_install_targets()
    missing = expected - install_targets
    extra = install_targets - expected
    assert not missing, (
        f"llms.txt: Installation section is missing links for: {sorted(missing)}"
    )
    assert not extra, (
        f"llms.txt: Installation section links to non-existent guides: {sorted(extra)}"
    )


@pytest.mark.parametrize(
    "target",
    sorted(_expected_install_targets()),
)
def test_installation_link_targets_exist(target: str) -> None:
    """Every link target in ``## Installation`` resolves to a real file."""
    text = _read_llms_txt()
    listed = {t for _, t in _installation_links(text)}
    assert target in listed, f"llms.txt: Installation section must link to {target!r}"
    resolved = (REPO_ROOT / target).resolve()
    assert resolved.is_file(), (
        f"llms.txt: linked file {target!r} does not exist at {resolved}"
    )
