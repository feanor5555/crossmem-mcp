from __future__ import annotations

from pathlib import Path

from tests._fixtures.embedder import FixedEmbedder


def test_no_legacy_mock_embedder_fixture() -> None:
    """Guard against re-introducing the redundant mock_embedder fixture.

    21.5 introduced :class:`FixedEmbedder` as the shared test embedder; the
    old ``mock_embedder`` autouse fixture in ``tests/conftest.py`` became
    redundant. 22.9 removed it. Use ``FixedEmbedder`` directly instead.
    """
    repo_root = Path(__file__).resolve().parents[2]
    conftest = repo_root / "tests" / "conftest.py"
    if conftest.exists():
        assert "mock_embedder" not in conftest.read_text(encoding="utf-8")


def test_fixed_embedder_dimension():
    vec = FixedEmbedder().embed_passage("hello")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_fixed_embedder_deterministic():
    embedder = FixedEmbedder()
    vec1 = embedder.embed_passage("hello world")
    vec2 = embedder.embed_passage("hello world")
    assert vec1 == vec2


def test_fixed_embedder_different_inputs():
    embedder = FixedEmbedder()
    vec1 = embedder.embed_passage("hello")
    vec2 = embedder.embed_passage("world")
    assert vec1 != vec2
