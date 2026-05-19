"""Pre-flight lint for ``install/<cli>.md`` files (Task 17.4).

This script is the worker behind the ``install-docs-lint`` GitHub Actions
workflow. It enforces three contracts so the per-CLI install docs stay
LLM-executable:

1. **Connector parity** — every connector in
   :data:`crossmem.installer.ALL_CONNECTORS` has an ``install/<name>.md``.
   When a new connector lands without its install doc, the script exits 1
   with a clear "missing install doc for <name>" message.
2. **Header schema** — each per-CLI file satisfies
   :func:`tests.install._helpers.assert_install_doc_schema` (YAML
   frontmatter with the documented keys, five required H2 headings in
   order).
3. **Configure-MCP snippet parses** — the first fenced code block under
   ``## Configure MCP`` parses as JSON or YAML, based on the fence tag,
   and contains a ``crossmem`` key somewhere in the nested mapping.

The script also re-validates ``skills/crossmem-install/SKILL.md`` against
the agentskills.io schema (frontmatter has ``name``/``description``/
``compatibility``, body fits the 500-line budget). This mirrors what
``skills-ref validate`` would do without requiring the tool to be
installed in CI.

Usage:

    python tools/check_install_docs.py [--repo-root PATH]

Exit codes:

* 0 — all checks pass.
* 1 — at least one check failed; details on stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Prerequisites",
    "Install",
    "Configure MCP",
    "Verify",
    "Troubleshooting",
)

FRONTMATTER_KEYS: frozenset[str] = frozenset(
    {
        "cli",
        "min_crossmem_version",
        "config_path_linux",
        "config_path_mac",
        "config_path_win",
        "restart_hint",
    }
)

# Skills schema (agentskills.io subset that ``skills-ref validate`` checks).
SKILL_REQUIRED_FIELDS: tuple[str, ...] = ("name", "description", "compatibility")
SKILL_MAX_BODY_LINES = 500
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CONFIGURE_MCP_FENCE_RE = re.compile(
    r"^## Configure MCP\b.*?^```(?P<lang>\w+)\r?\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)
_NON_CLI_FILES: frozenset[str] = frozenset({"_template.md", "README.md"})


def _split_frontmatter(text: str) -> tuple[dict[str, object] | None, str]:
    """Return parsed YAML frontmatter (or None) and the remaining body."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None, text
    raw = match.group(1)
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        msg = "YAML frontmatter must be a mapping"
        raise ValueError(msg)
    return data, text[match.end() :]


def _check_install_doc(path: Path) -> list[str]:
    """Return a list of human-readable errors for ``path``, empty on success."""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")

    try:
        frontmatter, body = _split_frontmatter(text)
    except ValueError as exc:
        return [f"{path}: frontmatter parse error: {exc}"]

    if frontmatter is None:
        return [f"{path}: missing YAML frontmatter"]

    actual_keys = set(frontmatter.keys())
    if actual_keys != FRONTMATTER_KEYS:
        missing = sorted(FRONTMATTER_KEYS - actual_keys)
        extra = sorted(actual_keys - FRONTMATTER_KEYS)
        errors.append(
            f"{path}: frontmatter keys mismatch (missing={missing}, extra={extra})"
        )

    headings = [m.group(1).strip() for m in _H2_RE.finditer(body)]
    required = list(REQUIRED_HEADINGS)
    seen = [h for h in headings if h in required]
    if seen != required:
        errors.append(
            f"{path}: required headings missing or out of order. "
            f"Expected {required}, got {seen} (all H2: {headings})"
        )

    # Configure-MCP snippet must parse.
    match = _CONFIGURE_MCP_FENCE_RE.search(body)
    if match is None:
        errors.append(f"{path}: no fenced code block in '## Configure MCP' section")
    else:
        lang = match.group("lang")
        snippet = match.group("body")
        try:
            if lang == "json":
                parsed = json.loads(snippet)
            elif lang == "yaml":
                parsed = yaml.safe_load(snippet)
            else:
                errors.append(
                    f"{path}: unsupported Configure-MCP fence language {lang!r}"
                )
                parsed = None
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            errors.append(f"{path}: Configure-MCP snippet does not parse: {exc}")
            parsed = None

        if parsed is not None and not _contains_crossmem_key(parsed):
            errors.append(
                f"{path}: Configure-MCP snippet must define a 'crossmem' entry"
            )

    return errors


def _contains_crossmem_key(value: object) -> bool:
    if isinstance(value, dict):
        return "crossmem" in value or any(
            _contains_crossmem_key(v) for v in value.values()
        )
    return False


def _check_connector_parity(install_dir: Path) -> list[str]:
    """Return errors for connectors registered but missing an install doc."""
    try:
        from crossmem.installer import ALL_CONNECTORS  # noqa: PLC0415
    except ImportError as exc:
        return [
            "could not import crossmem.installer.ALL_CONNECTORS "
            f"(PYTHONPATH issue?): {exc}"
        ]

    errors: list[str] = []
    for factory in ALL_CONNECTORS:
        name = factory().name()
        expected = install_dir / f"{name}.md"
        if not expected.is_file():
            errors.append(
                f"missing install doc for connector {name!r}: expected {expected}"
            )
    return errors


def _check_skill(skill_md: Path) -> list[str]:
    """Inlined ``skills-ref validate`` checks for ``crossmem-install``."""
    if not skill_md.is_file():
        return [f"missing skill file: {skill_md}"]

    errors: list[str] = []
    text = skill_md.read_text(encoding="utf-8")

    try:
        frontmatter, body = _split_frontmatter(text)
    except ValueError as exc:
        return [f"{skill_md}: frontmatter parse error: {exc}"]

    if frontmatter is None:
        return [f"{skill_md}: missing YAML frontmatter"]

    for field in SKILL_REQUIRED_FIELDS:
        if not frontmatter.get(field):
            errors.append(f"{skill_md}: frontmatter must define non-empty {field!r}")

    name = str(frontmatter.get("name", ""))
    if name and not SKILL_NAME_RE.match(name):
        errors.append(f"{skill_md}: skill name {name!r} must be lowercase kebab-case")
    if name and name != skill_md.parent.name:
        errors.append(
            f"{skill_md}: 'name' ({name!r}) must match parent directory "
            f"({skill_md.parent.name!r})"
        )

    body_lines = body.splitlines()
    if len(body_lines) > SKILL_MAX_BODY_LINES:
        errors.append(
            f"{skill_md}: body is {len(body_lines)} lines; "
            f"budget is {SKILL_MAX_BODY_LINES}"
        )

    first_meaningful = next(
        (line for line in body_lines if line.strip()),
        "",
    )
    if not first_meaningful.startswith("# "):
        errors.append(f"{skill_md}: body must open with an H1 heading")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root containing install/ and skills/.",
    )
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root.resolve()
    install_dir = repo_root / "install"
    skill_md = repo_root / "skills" / "crossmem-install" / "SKILL.md"

    errors: list[str] = []

    if not install_dir.is_dir():
        print(f"error: install directory not found at {install_dir}", file=sys.stderr)
        return 1

    errors.extend(_check_connector_parity(install_dir))

    for path in sorted(install_dir.glob("*.md")):
        if path.name in _NON_CLI_FILES:
            continue
        errors.extend(_check_install_doc(path))

    errors.extend(_check_skill(skill_md))

    if errors:
        print("install-docs lint failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("install-docs lint passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
