"""Snapshot test for `install/_template.md` structure.

Verifies that the install-doc template carries the five mandatory
H2 headings in the right order plus a YAML frontmatter with the
documented keys. The helper used here will be reused once the
per-CLI `install/<cli>.md` files exist (Task 15.5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.install._helpers import (
    FRONTMATTER_KEYS,
    REQUIRED_HEADINGS,
    assert_install_doc_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_DIR = REPO_ROOT / "install"
TEMPLATE_PATH = INSTALL_DIR / "_template.md"
README_PATH = INSTALL_DIR / "README.md"


def test_install_dir_exists() -> None:
    assert INSTALL_DIR.is_dir(), f"{INSTALL_DIR} must exist"


def test_readme_exists_and_mentions_template() -> None:
    assert README_PATH.is_file(), f"{README_PATH} must exist"
    text = README_PATH.read_text(encoding="utf-8")
    assert "_template.md" in text, "README must reference _template.md"


def test_template_exists() -> None:
    assert TEMPLATE_PATH.is_file(), f"{TEMPLATE_PATH} must exist"


def test_template_has_required_headings_and_frontmatter() -> None:
    assert_install_doc_schema(TEMPLATE_PATH)


@pytest.mark.parametrize("heading", REQUIRED_HEADINGS)
def test_template_contains_each_required_heading(heading: str) -> None:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert f"## {heading}" in text, f"missing heading '## {heading}'"


@pytest.mark.parametrize("key", FRONTMATTER_KEYS)
def test_template_frontmatter_contains_key(key: str) -> None:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Cheap textual check; full structural validation lives in
    # assert_install_doc_schema.
    assert f"{key}:" in text, f"frontmatter must contain key '{key}'"
