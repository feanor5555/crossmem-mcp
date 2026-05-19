"""Validation for ``skills/crossmem-install/SKILL.md``.

The ``skills-ref`` CLI is not an installed dev dependency in this repo, so we
inline its core checks (frontmatter schema, naming convention, body line
budget) here. The checks mirror what ``skills-ref validate`` would enforce
per the agentskills.io spec.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "crossmem-install"
SKILL_MD = SKILL_DIR / "SKILL.md"

MAX_BODY_LINES = 500
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _read_skill() -> str:
    assert SKILL_MD.exists(), f"missing SKILL.md at {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the YAML frontmatter block at the top of the file.

    The skill spec uses a minimal subset: top-level ``key: value`` pairs
    only. We intentionally avoid pulling in ``pyyaml`` here so the test is
    self-contained and fast.
    """
    if not text.startswith("---\n"):
        raise AssertionError("SKILL.md must start with a '---' frontmatter marker")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise AssertionError("SKILL.md frontmatter is not terminated by '---'")
    block = text[4:end]
    body = text[end + len("\n---\n") :]
    meta: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise AssertionError(f"frontmatter line is not 'key: value': {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def test_skill_file_exists() -> None:
    assert SKILL_MD.is_file(), f"expected file at {SKILL_MD}"


def test_directory_name_matches_skill_name() -> None:
    meta, _ = _split_frontmatter(_read_skill())
    assert meta.get("name") == SKILL_DIR.name, (
        "the 'name' frontmatter field must match the skill directory name"
    )


def test_skill_name_follows_naming_convention() -> None:
    meta, _ = _split_frontmatter(_read_skill())
    name = meta.get("name", "")
    assert NAME_PATTERN.match(name), (
        f"skill name {name!r} must be lowercase kebab-case (a-z, 0-9, '-')"
    )


@pytest.mark.parametrize("field", ["name", "description", "compatibility"])
def test_frontmatter_has_required_fields(field: str) -> None:
    meta, _ = _split_frontmatter(_read_skill())
    assert field in meta, f"frontmatter must define '{field}'"
    assert meta[field], f"frontmatter '{field}' must not be empty"


def test_compatibility_documents_python_and_pipx() -> None:
    meta, _ = _split_frontmatter(_read_skill())
    compat = meta["compatibility"]
    assert "Python 3.10" in compat, "compatibility must mention Python 3.10+"
    assert "pipx" in compat, "compatibility must mention pipx"


def test_description_carries_when_to_use_trigger() -> None:
    meta, _ = _split_frontmatter(_read_skill())
    description = meta["description"].lower()
    # The description doubles as the LLM trigger; it must surface both the
    # crossmem identifier and an install-related intent verb so retrievers
    # match user requests like "install crossmem" or "register crossmem MCP".
    assert "crossmem" in description, "description must mention crossmem by name"
    intent_words = ("install", "register", "configure", "set up", "setup")
    assert any(word in description for word in intent_words), (
        "description must surface an install/register/configure intent"
    )


def test_body_within_line_budget() -> None:
    _, body = _split_frontmatter(_read_skill())
    body_lines = body.splitlines()
    assert len(body_lines) <= MAX_BODY_LINES, (
        f"SKILL.md body is {len(body_lines)} lines; budget is {MAX_BODY_LINES}"
    )


def test_body_starts_with_h1_heading() -> None:
    _, body = _split_frontmatter(_read_skill())
    first_meaningful = next(
        (line for line in body.splitlines() if line.strip()),
        "",
    )
    assert first_meaningful.startswith("# "), (
        "body must open with an H1 heading per the agent-skills spec"
    )


def test_body_references_per_cli_install_files() -> None:
    _, body = _split_frontmatter(_read_skill())
    # Progressive Disclosure: the master skill must defer CLI-specific steps
    # to ``install/<cli>.md`` files rather than inlining them.
    assert "install/" in body, (
        "body must reference per-CLI files under install/<cli>.md so the LLM "
        "follows progressive disclosure instead of inlining 12 CLIs"
    )
