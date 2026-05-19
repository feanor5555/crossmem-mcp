"""Task 25.2 — LICENSE file + pyproject.toml distribution metadata.

The repo is about to be flipped to public under GPLv3 (see CLAUDE.md
"Lizenz: GPLv3"). Two things must be in place before that switch:

1. A top-level ``LICENSE`` file with the official GPL-3.0 text. Without
   it, redistributors have no licence grant in the source tree and SPDX
   scanners (PyPI, GitHub) cannot detect the licence.
2. ``pyproject.toml`` must surface the licence, authors, description,
   project URLs, README and the matching OSI classifier so ``pip show``
   and the PyPI listing carry the right metadata.

The optional-dependency section is **not** asserted here — task 25.3
edits ``[project.optional-dependencies]`` and we keep those two patches
on independent contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
LICENSE_PATH = REPO_ROOT / "LICENSE"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
README_PATH = REPO_ROOT / "README.md"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# LICENSE file
# ---------------------------------------------------------------------------


def test_license_file_exists_at_repo_root() -> None:
    """``LICENSE`` must live at the repo root so GitHub/PyPI detect it."""
    assert LICENSE_PATH.is_file(), (
        f"LICENSE file missing at repo root (expected {LICENSE_PATH}). "
        "GPLv3 distribution requires the full licence text in the source tree."
    )


def test_license_contains_official_gpl3_header() -> None:
    """LICENSE must carry the official GPL-3.0 header (not a summary)."""
    text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in text, (
        "LICENSE must contain the official 'GNU GENERAL PUBLIC LICENSE' "
        "header — a paraphrase or summary is not a valid grant."
    )
    assert "Version 3, 29 June 2007" in text, (
        "LICENSE must be GPL **version 3** (29 June 2007). The project "
        "declares GPL-3.0-or-later in pyproject.toml."
    )


def test_license_contains_full_terms() -> None:
    """LICENSE must be the full text, not a stub. Sanity-check key sections."""
    text = LICENSE_PATH.read_text(encoding="utf-8")
    # A truncated licence is the most common mistake — assert the document
    # is long enough to include the full terms (official text ~674 lines).
    assert text.count("\n") > 600, (
        f"LICENSE looks truncated ({text.count(chr(10))} lines). "
        "The official GPL-3.0 text is ~674 lines."
    )
    # Spot-check signature clauses from across the document so a future
    # truncation does not silently pass.
    for marker in (
        "TERMS AND CONDITIONS",
        "0. Definitions.",
        "15. Disclaimer of Warranty.",
        "END OF TERMS AND CONDITIONS",
    ):
        assert marker in text, f"LICENSE is missing required GPL-3.0 marker: {marker!r}"


# ---------------------------------------------------------------------------
# pyproject.toml [project] metadata
# ---------------------------------------------------------------------------


def test_pyproject_declares_gpl3_license() -> None:
    """``[project].license`` must declare GPL-3.0-or-later (SPDX)."""
    project = _load_pyproject()["project"]
    license_field = project.get("license")
    assert license_field is not None, (
        "pyproject.toml [project].license must be set (SPDX expression). "
        "Without it, ``pip show`` and PyPI show 'UNKNOWN' for licence."
    )
    # PEP 639 allows either a string SPDX expression or a {text=...} table;
    # accept the SPDX string form for forward compatibility.
    if isinstance(license_field, str):
        spdx = license_field
    else:
        spdx = license_field.get("text", "")
    assert spdx == "GPL-3.0-or-later", (
        "pyproject.toml [project].license must be the SPDX expression "
        f"'GPL-3.0-or-later' (CLAUDE.md mandate). Got: {license_field!r}"
    )


def test_pyproject_lists_authors() -> None:
    """``[project].authors`` must be populated for wheel metadata."""
    project = _load_pyproject()["project"]
    authors = project.get("authors")
    assert authors, (
        "pyproject.toml [project].authors must be set so wheel metadata "
        "carries an Author-Email field."
    )
    assert isinstance(authors, list) and authors, (
        f"[project].authors must be a non-empty list, got {authors!r}"
    )
    first = authors[0]
    assert isinstance(first, dict) and "name" in first, (
        f"[project].authors entries need at least a 'name' key, got {first!r}"
    )


def test_pyproject_has_description() -> None:
    """``[project].description`` must be a short summary for PyPI."""
    project = _load_pyproject()["project"]
    description = project.get("description", "")
    assert description, (
        "pyproject.toml [project].description must be set (PyPI summary)."
    )
    # PyPI summary is single-line. Keep it tight; trip well before the
    # PyPI 512-char limit so we notice if someone pastes a paragraph here.
    assert "\n" not in description, (
        f"[project].description must be single-line, got: {description!r}"
    )
    assert len(description) <= 200, (
        f"[project].description is too long ({len(description)} chars), "
        "keep it under 200 for the PyPI listing."
    )


def test_pyproject_lists_project_urls() -> None:
    """``[project].urls`` must surface Homepage / Source / Issues for PyPI."""
    project = _load_pyproject()["project"]
    urls = project.get("urls", {})
    assert urls, (
        "pyproject.toml [project].urls must be set (PyPI 'Project links' panel)."
    )
    # Required keys — PyPI surfaces these as clickable links.
    for key in ("Homepage", "Source", "Issues"):
        assert key in urls, (
            f"[project].urls must include {key!r}; got keys {sorted(urls)!r}"
        )
        assert urls[key].startswith("https://"), (
            f"[project].urls.{key} must be an https URL, got {urls[key]!r}"
        )


def test_pyproject_points_at_readme() -> None:
    """``[project].readme`` must point at a real README file."""
    project = _load_pyproject()["project"]
    readme = project.get("readme")
    assert readme, (
        "pyproject.toml [project].readme must be set so PyPI renders a "
        "long description."
    )
    # Accept either a bare path or the table form.
    if isinstance(readme, str):
        readme_path = REPO_ROOT / readme
    else:
        readme_path = REPO_ROOT / readme["file"]
    assert readme_path.is_file(), (
        f"[project].readme points at {readme_path}, but that file does not "
        "exist — wheel build will fail with FileNotFoundError."
    )


def test_pyproject_classifiers_include_gplv3() -> None:
    """Classifiers must include the GPL-3.0-or-later OSI classifier."""
    project = _load_pyproject()["project"]
    classifiers = project.get("classifiers", [])
    assert classifiers, (
        "pyproject.toml [project].classifiers must be set (PyPI categorisation)."
    )
    gpl_classifier = (
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
    )
    assert gpl_classifier in classifiers, (
        f"[project].classifiers must include {gpl_classifier!r}; got {classifiers!r}"
    )
