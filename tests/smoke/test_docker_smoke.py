"""End-to-end smoke test on a fresh ``ubuntu:24.04`` container.

The default CI matrix in ``ci.yml`` exercises crossmem inside the same
checkout that built it (``PYTHONPATH=src python -m pytest``). That catches
regressions in the source tree but tells us nothing about whether the
*published* wheel actually installs and runs on a clean system: missing
``package_data`` entries, stray editable-install assumptions, or a typo
in ``pyproject.toml`` would all sneak past.

This smoke test closes that gap. It builds a wheel via ``python -m build``,
mounts it into a freshly pulled ``ubuntu:24.04`` container and runs the
real ``pip install crossmem`` flow against it. Inside the container we
then:

1. Call ``crossmem doctor`` to confirm the installed package can run its
   preflight checks (exits 0 with the default extras absent only when
   every required module is importable from the wheel).
2. Drive :func:`crossmem.installer.install` against a tmp HOME with a
   pre-seeded Goose config file. Goose is the only one of the twelve
   supported CLIs whose config we can synthesize headless-ly — Cursor /
   Claude Code / Cline are GUI apps and their auth flows are not CI-
   testable. We assert the installer wrote the ``crossmem`` entry into
   the Goose YAML.
3. Run a ``store -> query`` round-trip via :class:`KnowledgeStore` with a
   stub embedder. Pulling the 120MB fastembed model on every nightly run
   is wasteful and orthogonal to "does the wheel install"; the model
   download itself is covered by ``ci.yml`` via the fastembed cache.

Skips by default — running the test mutates the local Docker daemon and
pulls a ~80MB ubuntu image. Opt in by either:

* setting ``CROSSMEM_RUN_SMOKE=1`` (CI does this in ``nightly.yml``), or
* passing ``-m smoke`` to pytest (the marker is registered in
  ``pyproject.toml``).

If ``docker`` is not on ``PATH`` the test skips with a clear message —
contributors without Docker installed will never see a spurious failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image is intentionally pinned: ``ubuntu:24.04`` is the current LTS at the
# time of writing and is the closest match to ``ubuntu-latest`` runners.
# Bumping it deliberately (e.g. to ``ubuntu:26.04``) should be a separate
# task so the change is reviewable.
IMAGE = "ubuntu:24.04"

pytestmark = pytest.mark.smoke


def _docker_available() -> bool:
    """Return True iff a usable ``docker`` binary is on ``PATH``."""
    return shutil.which("docker") is not None


def _smoke_enabled() -> bool:
    """Return True iff the user (or CI) explicitly enabled smoke runs."""
    return os.environ.get("CROSSMEM_RUN_SMOKE") == "1"


def _build_wheel(dest: Path) -> Path:
    """Build a wheel into ``dest`` via ``python -m build --wheel`` and return its path.

    Uses the current Python interpreter so the wheel matches the same
    Python ABI as the test runner. ``python -m build`` creates a fresh
    isolated build env, so the resulting wheel is reproducible regardless
    of which dev-deps happen to be installed in the runner's venv.
    """
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dest)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheels = sorted(dest.glob("crossmem-*.whl"))
    if not wheels:
        raise AssertionError(f"no wheel produced under {dest}")
    return wheels[-1]


def _container_script(wheel_name: str) -> str:
    """Return the bash script the container runs after the wheel mount.

    The script intentionally avoids ``pip install`` of any optional extra
    so the run mirrors a vanilla ``pip install crossmem`` from PyPI. Every
    failure mode (missing import, broken entry-point, wrong default model
    in fastembed's supported list) trips an explicit ``exit 1`` so the
    test sees a non-zero container exit code with a usable error.
    """
    # ``set -euo pipefail`` makes every step fail loudly. The ``DEBIAN_FRONTEND``
    # var keeps ``apt-get`` from blocking on tzdata configure prompts.
    return textwrap.dedent(
        f"""
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive

        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \\
            python3 python3-pip python3-venv ca-certificates >/dev/null

        python3 -m venv /opt/venv
        # shellcheck disable=SC1091
        . /opt/venv/bin/activate

        pip install --quiet --upgrade pip
        pip install --quiet "/wheels/{wheel_name}"

        echo '--- crossmem doctor ---'
        crossmem doctor

        echo '--- crossmem install (Goose-only, fake HOME) ---'
        export HOME=/tmp/fakehome
        mkdir -p "$HOME/.config/goose"
        echo '{{}}' > "$HOME/.config/goose/config.yaml"

        python3 - <<'PYEOF'
        import sys
        from pathlib import Path

        from crossmem.connectors.goose import GooseConnector
        from crossmem.installer import install

        class _StubEmbedder:
            model_name = 'stub'
            def embed_query(self, text):
                return [0.0] * 384
            def embed_passage(self, text):
                return [0.0] * 384
            def embed_passage_batch(self, texts, batch_size=32):
                return [[0.0] * 384 for _ in texts]

        result = install(
            connectors=[GooseConnector()],
            embedder_factory=_StubEmbedder,
        )
        assert 'goose' in result.detected_clis, result.detected_clis
        cfg = Path.home() / '.config' / 'goose' / 'config.yaml'
        text = cfg.read_text(encoding='utf-8')
        assert 'crossmem' in text, text
        assert 'crossmem.server' in text, text
        print('install: ok, detected =', result.detected_clis)

        # store -> query round-trip on the freshly-created knowledge.db
        from crossmem.backends.sqlite_backend import SQLiteBackend
        from crossmem.core.store import KnowledgeStore

        backend = SQLiteBackend(result.db_path)
        store = KnowledgeStore(backend=backend, embedder=_StubEmbedder())
        ids = store.store(
            content='hello smoke world',
            source_url='https://example.test/smoke',
            title='smoke',
            source_type='web',
        )
        assert ids, 'store returned no ids'
        hits = store.query('smoke', top_k=5)
        assert any(h.metadata.source_url == 'https://example.test/smoke' for h in hits), hits
        print('roundtrip: ok, ids =', ids)
        PYEOF

        echo '--- smoke ok ---'
        """  # noqa: E501
    ).strip()


def _run_in_container(wheel_dir: Path, wheel_name: str) -> subprocess.CompletedProcess:
    """Run the smoke script inside a fresh ``ubuntu:24.04`` container.

    ``--rm`` cleans the container up afterwards. ``--network=none`` is
    *not* set: ``apt-get`` and ``pip install`` both need outbound HTTP.
    The wheel directory is mounted read-only so the container cannot
    accidentally clobber the host's ``dist/`` tree.
    """
    script = _container_script(wheel_name)
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{wheel_dir}:/wheels:ro",
            IMAGE,
            "bash",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(not _docker_available(), reason="docker not available on PATH")
@pytest.mark.skipif(
    not _smoke_enabled(),
    reason="set CROSSMEM_RUN_SMOKE=1 to run the docker end-to-end smoke",
)
def test_docker_end_to_end_smoke(tmp_path: Path) -> None:
    """Build a wheel, install it inside ``ubuntu:24.04``, run a store->query.

    Validates that the *published* artefact (not the source tree) is
    self-contained: console-script, MCP server registration via the Goose
    connector, and the SQLite-backed ``store -> query`` round-trip all
    succeed on a system that has never seen crossmem before.
    """
    wheel_dir = tmp_path / "dist"
    wheel = _build_wheel(wheel_dir)
    result = _run_in_container(wheel_dir, wheel.name)
    combined = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "smoke ok" in result.stdout, combined
    assert "install: ok" in result.stdout, combined
    assert "roundtrip: ok" in result.stdout, combined


def test_smoke_module_collects_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: the helpers stay importable even when docker is absent.

    Prevents a regression where someone wraps the helpers in a top-level
    ``shutil.which`` guard that turns them into ``None`` at import time
    (breaking pytest collection on dev machines that lack docker).
    """
    monkeypatch.setenv("PATH", "")
    # Importing the module again would no-op (already loaded); we just
    # exercise the predicate to make sure it returns False rather than
    # raising when ``docker`` is missing.
    assert _docker_available() is False
