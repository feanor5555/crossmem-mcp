from __future__ import annotations

import threading
import time

import numpy as np
import pytest


class _SlowStub:
    """Stub for TextEmbedding that sleeps before becoming usable."""

    construction_count = 0

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        type(self).construction_count += 1
        self.model_name = model_name
        time.sleep(0.2)

    def embed(self, texts):
        for _ in texts:
            yield np.zeros(384, dtype=np.float32)


class _RaisingStub:
    """Stub for TextEmbedding that fails during construction."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("model failed")


class _CountingStub:
    """Stub that counts constructions and sleeps briefly."""

    construction_count = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).construction_count += 1
        time.sleep(0.05)

    def embed(self, texts):
        for _ in texts:
            yield np.zeros(384, dtype=np.float32)


def _reset_stub(cls: type) -> None:
    cls.construction_count = 0


def test_warmup_starts_in_background(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_stub(_SlowStub)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _SlowStub)

    from crossmem.core.embedding import EmbeddingService

    service = EmbeddingService(model_name="dummy")
    # Warmup runs in background -> not ready immediately
    assert not service._ready.is_set()

    # After enough time the warmup must have completed
    time.sleep(0.5)
    assert service._ready.is_set()


def test_embed_waits_for_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_stub(_SlowStub)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _SlowStub)

    from crossmem.core.embedding import EmbeddingService

    service = EmbeddingService(model_name="dummy")
    start = time.monotonic()
    service.embed("x")
    elapsed = time.monotonic() - start
    # The slow stub takes 0.2s; allow tolerance for scheduler jitter
    assert elapsed >= 0.15


def test_embed_propagates_warmup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _RaisingStub)

    from crossmem.core.embedding import EmbeddingService

    service = EmbeddingService(model_name="dummy")
    with pytest.raises(RuntimeError, match="model failed"):
        service.embed("x")


def test_warmup_runs_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_stub(_CountingStub)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _CountingStub)

    from crossmem.core.embedding import EmbeddingService

    service = EmbeddingService(model_name="dummy")

    results: list[list[float]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(service.embed("x"))
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 5
    assert _CountingStub.construction_count == 1
