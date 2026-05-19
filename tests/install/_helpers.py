"""Shared helper for validating install-doc schema.

Used by `test_template.py` today and intended to be reused by per-CLI
install-doc tests once `install/<cli>.md` files exist (Task 15.5).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Prerequisites",
    "Install",
    "Configure MCP",
    "Verify",
    "Troubleshooting",
)

FRONTMATTER_KEYS: tuple[str, ...] = (
    "cli",
    "min_crossmem_version",
    "config_path_linux",
    "config_path_mac",
    "config_path_win",
    "restart_hint",
)

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_frontmatter(text: str) -> tuple[dict[str, object] | None, str]:
    """Return parsed YAML frontmatter (or None) and the remaining body."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None, text
    raw = match.group(1)
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        msg = "YAML frontmatter must be a mapping"
        raise AssertionError(msg)
    return data, text[match.end() :]


def assert_install_doc_schema(path: Path, *, require_frontmatter: bool = True) -> None:
    """Assert that `path` follows the install-doc schema.

    - Must contain the five required `## ` headings in order:
      Prerequisites, Install, Configure MCP, Verify, Troubleshooting.
    - If `require_frontmatter` is True, must have a YAML frontmatter
      whose keys are exactly the documented FRONTMATTER_KEYS set.
    """
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)

    if require_frontmatter:
        if frontmatter is None:
            msg = f"{path}: missing YAML frontmatter"
            raise AssertionError(msg)
        actual = set(frontmatter.keys())
        expected = set(FRONTMATTER_KEYS)
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            msg = (
                f"{path}: frontmatter keys mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
            raise AssertionError(msg)

    headings = [m.group(1).strip() for m in _H2_RE.finditer(body)]
    required = list(REQUIRED_HEADINGS)
    # Filter to the required ones only, preserving order, to allow
    # additional informational headings in real CLI files later.
    seen = [h for h in headings if h in required]
    if seen != required:
        msg = (
            f"{path}: required headings missing or out of order. "
            f"Expected order {required}, got {seen} "
            f"(all H2 headings: {headings})"
        )
        raise AssertionError(msg)
