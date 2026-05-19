"""Tests for ``tools/check_install_docs.py`` (Task 17.4).

The script is the heart of the ``install-docs-lint`` CI job. It enforces three
contracts on every PR / push:

1. **Connector parity** — every connector registered in
   :data:`crossmem.installer.ALL_CONNECTORS` has a matching
   ``install/<name>.md`` file. A new connector without its install doc must
   fail the job loudly.
2. **Header schema** — every ``install/<cli>.md`` satisfies the schema
   from :func:`tests.install._helpers.assert_install_doc_schema`.
3. **Configure-MCP snippet parses** — the first fenced code block under
   ``## Configure MCP`` is valid JSON or YAML (whatever the fence tag says).

The script also re-uses the same SKILL.md checks that the dedicated
``tests/skills`` test module enforces, so ``skills-ref validate`` does not
need to be installed in CI.

We run the script via ``subprocess`` so the tests exercise the exact entry
point the workflow calls. A success run on the real repo is the happy
path; a fabricated tmp-repo with a missing install doc proves the failure
mode.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "check_install_docs.py"


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the check script with ``repo`` as the repository root."""
    env = dict(os.environ)
    # The script imports ``crossmem.installer`` from the source tree.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_script_passes_on_real_repo() -> None:
    """All 12 connectors ship an ``install/<cli>.md`` — script must exit 0."""
    result = _run(REPO_ROOT)
    assert result.returncode == 0, (
        f"check_install_docs.py failed on the real repo:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_script_fails_when_connector_doc_is_missing(tmp_path: Path) -> None:
    """Deleting an ``install/<cli>.md`` from a fake repo must trip exit 1.

    We mirror the layout the script reads from (``install/`` directory and
    ``skills/crossmem-install/SKILL.md``) into ``tmp_path`` and then drop one
    connector's install doc to simulate the "new connector landed, install
    doc was forgotten" regression.
    """
    fake = tmp_path / "repo"
    fake.mkdir()
    shutil.copytree(REPO_ROOT / "install", fake / "install")
    shutil.copytree(REPO_ROOT / "skills", fake / "skills")

    # Pick any connector file and delete it; the registry still references
    # the connector, so the script must complain.
    victim = fake / "install" / "claude_code.md"
    assert victim.exists(), "fixture setup: claude_code.md was expected"
    victim.unlink()

    result = _run(fake)
    assert result.returncode != 0, (
        "deleting an install doc must fail the script, "
        f"but got exit {result.returncode}.\nstdout:\n{result.stdout}"
    )
    combined = result.stdout + result.stderr
    assert "claude_code" in combined, (
        f"failure output must name the missing connector, got:\n{combined}"
    )


def test_script_fails_on_broken_configure_snippet(tmp_path: Path) -> None:
    """Corrupting the Configure-MCP JSON must fail the script.

    The whole point of the lint is to catch broken instructions before they
    reach an LLM. We mutate one install doc's JSON snippet into invalid
    JSON and expect a non-zero exit.
    """
    fake = tmp_path / "repo"
    fake.mkdir()
    shutil.copytree(REPO_ROOT / "install", fake / "install")
    shutil.copytree(REPO_ROOT / "skills", fake / "skills")

    target = fake / "install" / "claude_code.md"
    text = target.read_text(encoding="utf-8")
    # Insert an unbalanced brace just before the closing fence of the first
    # JSON block. The text-level corruption is enough to break ``json.loads``
    # without touching headings or frontmatter.
    broken = text.replace('"crossmem": {', '"crossmem": {{', 1)
    assert broken != text, "fixture setup: failed to corrupt JSON snippet"
    target.write_text(broken, encoding="utf-8")

    result = _run(fake)
    assert result.returncode != 0, (
        "broken JSON snippet must fail the script, "
        f"got exit {result.returncode}.\nstdout:\n{result.stdout}"
    )
