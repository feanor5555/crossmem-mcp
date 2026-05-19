from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crossmem.core.embedding import EMBEDDING_DIM, EMBEDDING_MODEL, EmbeddingService

if TYPE_CHECKING:
    from pathlib import Path

# Default + test model — fastembed-supported, 384-dim, multilingual.
# Matches ``EMBEDDING_MODEL`` in the embedding module.
_TEST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_embedding_service_embed():
    svc = EmbeddingService(model_name=_TEST_MODEL)
    vec = svc.embed("hello")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in vec)


def test_default_model_loads_and_embed_query_returns_384_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke: ``EmbeddingService()`` with no args + active SHA pin works.

    Regression guard for the 0.6.12 fix: the previous default
    (``intfloat/multilingual-e5-small``) is not in fastembed's supported
    list and raised ``ValueError`` on first ``embed*`` call. The new
    default must instantiate AND produce a 384-dim query vector with the
    SHA256 pin enforced.
    """
    monkeypatch.delenv("CROSSMEM_SKIP_MODEL_SHA_PIN", raising=False)
    svc = EmbeddingService()
    assert svc.model_name == EMBEDDING_MODEL
    vec = svc.embed_query("x")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in vec)


def test_embedding_service_cache_hit():
    svc = EmbeddingService(model_name=_TEST_MODEL)
    svc.embed("cache test")
    assert "cache test" in svc._cache

    # Second call should use cache (no error, same result)
    vec1 = svc.embed("cache test")
    vec2 = svc.embed("cache test")
    assert vec1 == vec2


def test_embed_passage_batch_returns_one_vector_per_input():
    svc = EmbeddingService(model_name=_TEST_MODEL)
    texts = ["hello", "world", "test"]
    results = svc.embed_passage_batch(texts)
    assert len(results) == 3
    for vec in results:
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM


def test_embed_passage_batch_populates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed_passage_batch`` must write every computed vector to the LRU.

    The previous ``embed_batch`` API bypassed the cache entirely. The new
    contract is "shared helper with ``embed``": a follow-up
    ``embed_passage`` for the same text must be served from cache without
    invoking the model.
    """
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    svc = EmbeddingService(model_name="stub-minilm")
    svc.embed_passage_batch(["alpha", "beta", "gamma"])
    # Cache keys use the prefixed text for e5 models and the raw text
    # otherwise. ``stub-non-e5`` is non-e5 -> raw keys.
    for key in ("alpha", "beta", "gamma"):
        assert key in svc._cache


def test_embed_passage_batch_uses_cache_on_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-passing a previously embedded text must not invoke the model."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")

    calls: list[list[str]] = []

    class _CountingEmbedding(_StubTextEmbedding):
        def embed(self, texts):
            batch = list(texts)
            calls.append(batch)
            return super().embed(batch)

    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _CountingEmbedding)
    svc = EmbeddingService(model_name="stub-minilm")
    svc.embed_passage_batch(["alpha", "beta"])
    # First call: both texts are misses -> exactly one model batch.
    assert calls == [["alpha", "beta"]]
    calls.clear()

    svc.embed_passage_batch(["alpha", "beta"])
    # Second call: all cached -> no further model invocation.
    assert calls == []


def test_embed_passage_batch_deduplicates_repeated_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate texts within one batch hit the model exactly once."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")

    calls: list[list[str]] = []

    class _CountingEmbedding(_StubTextEmbedding):
        def embed(self, texts):
            batch = list(texts)
            calls.append(batch)
            return super().embed(batch)

    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _CountingEmbedding)
    svc = EmbeddingService(model_name="stub-minilm")
    vecs = svc.embed_passage_batch(["dup", "dup", "dup"])
    assert calls == [["dup"]]
    # All three positions must still be filled with the same vector.
    assert vecs[0] == vecs[1] == vecs[2]
    assert len(vecs[0]) == EMBEDDING_DIM


def test_embed_passage_batch_is_idempotent_for_stored_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling the batch twice on the same input returns identical vectors."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    svc = EmbeddingService(model_name="stub-minilm")
    first = svc.embed_passage_batch(["a", "b"])
    second = svc.embed_passage_batch(["a", "b"])
    assert first == second


def test_embed_passage_batch_applies_passage_prefix_for_e5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """e5 models cache batched passages under the ``"passage: "`` prefix."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    svc = EmbeddingService(model_name="intfloat/multilingual-e5-small")
    svc.embed_passage_batch(["alpha"])
    assert "passage: alpha" in svc._cache
    assert "alpha" not in svc._cache


def test_embedding_service_case_insensitive_cache():
    svc = EmbeddingService(model_name=_TEST_MODEL)
    svc.embed("Hello")
    cache_size_after_first = len(svc._cache)

    svc.embed("hello")
    cache_size_after_second = len(svc._cache)

    # Both "Hello" and "hello" normalize to "hello" -> same cache entry
    assert cache_size_after_second == cache_size_after_first
    assert "hello" in svc._cache


# -------- query/passage prefix (e5-conditional) --------


def test_embed_query_and_passage_match_for_non_e5_model():
    """Non-e5 models embed query and passage identically (no prefix)."""
    svc = EmbeddingService(model_name=_TEST_MODEL)
    q = svc.embed_query("hello")
    p = svc.embed_passage("hello")
    assert q == p
    assert len(q) == EMBEDDING_DIM


def test_embed_query_caches_under_raw_key_for_non_e5_model():
    """For non-e5 models, embed_query must cache under the raw text."""
    svc = EmbeddingService(model_name=_TEST_MODEL)
    svc.embed_query("hi")
    assert "hi" in svc._cache
    assert "query: hi" not in svc._cache


def test_embed_passage_caches_under_raw_key_for_non_e5_model():
    """For non-e5 models, embed_passage must cache under the raw text."""
    svc = EmbeddingService(model_name=_TEST_MODEL)
    svc.embed_passage("hi")
    assert "hi" in svc._cache
    assert "passage: hi" not in svc._cache


def test_embed_query_uses_prefix_for_e5_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """e5 models receive the ``"query: "`` prefix and cache under it."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    svc = EmbeddingService(model_name="intfloat/multilingual-e5-small")
    svc.embed_query("hi")
    assert "query: hi" in svc._cache
    assert "hi" not in svc._cache


def test_embed_passage_uses_prefix_for_e5_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """e5 models receive the ``"passage: "`` prefix and cache under it."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    svc = EmbeddingService(model_name="intfloat/multilingual-e5-small")
    svc.embed_passage("hi")
    assert "passage: hi" in svc._cache
    assert "hi" not in svc._cache


# -------- SHA256 model pin --------


class _StubTextEmbedding:
    """Stub TextEmbedding with deterministic 384-dim zero embeddings."""

    def __init__(self, model_name: str = "stub-model", *_a, **_k) -> None:
        self.model_name = model_name

    def embed(self, texts):
        for _ in texts:
            yield np.zeros(EMBEDDING_DIM, dtype=np.float32)


def test_sha256_mismatch_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the loaded model's SHA differs from the pin, embed_* must raise."""
    monkeypatch.delenv("CROSSMEM_SKIP_MODEL_SHA_PIN", raising=False)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    fake_path = tmp_path / "model.onnx"
    fake_path.write_bytes(b"actual-model-bytes")
    actual_sha = hashlib.sha256(b"actual-model-bytes").hexdigest()
    monkeypatch.setattr(
        "crossmem.core.embedding._locate_model_file",
        lambda _m: fake_path,
    )
    monkeypatch.setattr(
        "crossmem.core.embedding.EMBEDDING_MODEL_SHA256",
        {"pinned-model": "0" * 64},
    )
    svc = EmbeddingService(model_name="pinned-model")
    with pytest.raises(RuntimeError) as exc_info:
        svc.embed_query("hi")
    msg = str(exc_info.value)
    assert "SHA256" in msg
    assert "pinned-model" in msg
    assert actual_sha in msg


def test_sha256_skipped_when_env_var_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CROSSMEM_SKIP_MODEL_SHA_PIN=1 bypasses the pin even on mismatch."""
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    fake_path = tmp_path / "model.onnx"
    fake_path.write_bytes(b"actual-model-bytes")
    monkeypatch.setattr(
        "crossmem.core.embedding._locate_model_file",
        lambda _m: fake_path,
    )
    monkeypatch.setattr(
        "crossmem.core.embedding.EMBEDDING_MODEL_SHA256",
        {"pinned-model": "0" * 64},
    )
    svc = EmbeddingService(model_name="pinned-model")
    # Must not raise; returns a 384-dim vector.
    vec = svc.embed_query("hi")
    assert len(vec) == EMBEDDING_DIM


def test_sha256_pin_skipped_when_model_unregistered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model name not in the registry is loaded without verification."""
    monkeypatch.delenv("CROSSMEM_SKIP_MODEL_SHA_PIN", raising=False)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("crossmem.core.embedding.EMBEDDING_MODEL_SHA256", {})
    svc = EmbeddingService(model_name="unregistered-model")
    vec = svc.embed_query("hi")
    assert len(vec) == EMBEDDING_DIM


def test_sha256_pin_passes_on_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A correct SHA in the registry loads without error."""
    monkeypatch.delenv("CROSSMEM_SKIP_MODEL_SHA_PIN", raising=False)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    fake_path = tmp_path / "model.onnx"
    fake_path.write_bytes(b"correct-model-bytes")
    correct_sha = hashlib.sha256(b"correct-model-bytes").hexdigest()
    monkeypatch.setattr(
        "crossmem.core.embedding._locate_model_file",
        lambda _m: fake_path,
    )
    monkeypatch.setattr(
        "crossmem.core.embedding.EMBEDDING_MODEL_SHA256",
        {"pinned-model": correct_sha},
    )
    svc = EmbeddingService(model_name="pinned-model")
    vec = svc.embed_query("hi")
    assert len(vec) == EMBEDDING_DIM


def test_sha256_pin_skipped_when_locate_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model file cannot be located, the pin check is skipped."""
    monkeypatch.delenv("CROSSMEM_SKIP_MODEL_SHA_PIN", raising=False)
    monkeypatch.setattr("crossmem.core.embedding.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("crossmem.core.embedding._locate_model_file", lambda _m: None)
    monkeypatch.setattr(
        "crossmem.core.embedding.EMBEDDING_MODEL_SHA256",
        {"pinned-model": "0" * 64},
    )
    svc = EmbeddingService(model_name="pinned-model")
    vec = svc.embed_query("hi")
    assert len(vec) == EMBEDDING_DIM


def test_locate_model_file_returns_none_for_missing_attrs() -> None:
    """The locator returns None on private-API drift instead of crashing."""
    from crossmem.core.embedding import _locate_model_file

    class _Broken:
        pass

    assert _locate_model_file(_Broken()) is None
    assert _locate_model_file(None) is None


def test_fp16_storage_documented_in_module_source() -> None:
    """Regression guard: the fp16-rounding-stays-in-fp32 caveat is documented.

    The fastembed cache stores half-precision tuples, but sqlite-vec only
    accepts fp32. Casting back to fp32 does NOT recover the lost precision —
    the rounding artefacts persist in the on-disk vector. Future readers
    must find this explained in the module source with a pointer to the
    "Embedding" section in ``CLAUDE.md``.
    """
    import inspect

    from crossmem.core import embedding

    source = inspect.getsource(embedding)
    lowered = source.lower()
    assert "fp16" in lowered, "fp16 rationale must mention 'fp16' literally"
    assert "fp32" in lowered, "fp16 rationale must mention 'fp32' literally"
    assert "claude.md" in lowered, (
        "fp16 rationale must reference the Embedding section in CLAUDE.md"
    )


# -------- fp16 precision contract (task 24.3) --------


class _NonZeroStubTextEmbedding:
    """Stub emitting fp32 values with non-trivial mantissa bits.

    The values are deliberately chosen so a naive fp32 -> list[float] -> tuple
    path WOULD differ from the fp16-rounded path: 0.1 has no exact binary
    representation in either fp32 or fp16, so an fp16 round-trip changes the
    bits. That makes the test below load-bearing — if the production code
    ever stopped rounding to fp16, this test would catch it.
    """

    def __init__(self, model_name: str = "stub-non-e5", *_a, **_k) -> None:
        self.model_name = model_name

    def embed(self, texts):
        # 0.1 cannot be represented exactly in fp16 OR fp32, so the
        # fp16 -> fp32 widening leaves an artefact that a missing fp16
        # round-trip would not produce.
        for _ in texts:
            yield np.full(EMBEDDING_DIM, 0.1, dtype=np.float32)


def test_embed_query_returns_fp16_roundtrip_stable_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings returned from the public API are bit-stable under fp16 round-trip.

    The :data:`crossmem.core.models.Vec` alias and the ``_Embedder`` Protocol
    docstring declare that values transported as ``list[float]`` carry fp16
    precision. The behavioural guarantee that backs that documentation is:
    re-rounding a returned vector to ``np.float16`` and back to ``list[float]``
    must yield the identical Python floats. If anyone ever removed the fp16
    rounding in ``_compute_embedding`` / ``_embed_many``, this test would
    fail because the underlying fp32 mantissa would no longer match the
    fp16 grid.
    """
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr(
        "crossmem.core.embedding.TextEmbedding", _NonZeroStubTextEmbedding
    )
    svc = EmbeddingService(model_name="stub-non-e5")

    vec = svc.embed_query("hello")
    roundtripped = np.array(vec, dtype=np.float16).astype(np.float32).tolist()
    assert vec == roundtripped, (
        "embed_query must return fp16-rounded values — see the precision "
        "policy in crossmem.core.embedding and the Vec alias in "
        "crossmem.core.models."
    )


def test_embed_passage_batch_returns_fp16_roundtrip_stable_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fp16 contract as the ``embed_query`` test above, but exercised
    via the batched entry point used by ``KnowledgeStore.store``.
    """
    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr(
        "crossmem.core.embedding.TextEmbedding", _NonZeroStubTextEmbedding
    )
    svc = EmbeddingService(model_name="stub-non-e5")

    vecs = svc.embed_passage_batch(["alpha", "beta"])
    for vec in vecs:
        roundtripped = np.array(vec, dtype=np.float16).astype(np.float32).tolist()
        assert vec == roundtripped, (
            "embed_passage_batch must return fp16-rounded values — see the "
            "precision policy in crossmem.core.embedding and the Vec alias "
            "in crossmem.core.models."
        )


# -------- concurrent cache safety (task 24.12) --------


class _YieldingStubTextEmbedding:
    """Stub that yields control between batch items to widen race windows.

    The default ``_StubTextEmbedding`` produces values in tight C-level
    iteration that the GIL effectively serialises against the cache code,
    masking real races. This variant releases the GIL between yields so
    multiple worker threads can interleave inside ``_cache_put`` /
    ``_cache_get`` and the LRU eviction path.
    """

    def __init__(self, model_name: str = "stub-yielding", *_a, **_k) -> None:
        self.model_name = model_name

    def embed(self, texts):
        import time

        for _ in texts:
            time.sleep(0)  # release the GIL
            yield np.zeros(EMBEDDING_DIM, dtype=np.float32)


def test_embed_query_concurrent_stress_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel ``embed_query`` calls must not raise ``KeyError`` from the LRU.

    Without serialisation, ``OrderedDict.popitem(last=False)`` in
    ``_cache_put`` races against ``move_to_end`` in ``_cache_get`` when many
    threads hammer unique keys that exceed ``_CACHE_MAX_SIZE``: two threads
    can observe ``len(_cache) > _CACHE_MAX_SIZE``, the second ``popitem``
    can collide with a concurrent ``move_to_end`` or another eviction,
    surfacing ``KeyError`` or leaving the LRU in a bad state. The stub
    deliberately releases the GIL inside ``embed`` to widen the race
    window; the regression guard then fans out unique-key queries from
    multiple threads and asserts that no thread surfaces an exception.
    """
    import threading

    from crossmem.core.embedding import _CACHE_MAX_SIZE

    monkeypatch.setenv("CROSSMEM_SKIP_MODEL_SHA_PIN", "1")
    monkeypatch.setattr(
        "crossmem.core.embedding.TextEmbedding", _YieldingStubTextEmbedding
    )
    svc = EmbeddingService(model_name="stub-yielding")
    # Wait for the warmup thread to publish the model before fanning out —
    # the race we want to expose is in the cache, not in warmup serialisation.
    svc._ensure_ready()

    n_threads = 16
    keys_per_thread = _CACHE_MAX_SIZE  # guaranteed eviction pressure
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(offset: int) -> None:
        try:
            barrier.wait()
            for i in range(keys_per_thread):
                svc.embed_query(f"t{offset}-k{i}")
        except BaseException as exc:  # pragma: no cover - we assert errors == []
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent embed_query raised: {errors!r}"
    # Cache must respect its capacity bound regardless of the race winner.
    assert len(svc._cache) <= _CACHE_MAX_SIZE


def test_embedding_service_serialises_cache_access() -> None:
    """The cache must be guarded by a reentrant lock.

    Behavioural regression guard for task 24.12: under the GIL,
    ``OrderedDict`` operations are individually atomic, so a missing lock
    rarely surfaces ``KeyError`` in practice — but the LRU ordering and
    eviction bounds are still racy, and a free-threaded build (PEP 703)
    would expose those races directly. We require an explicit lock so the
    guarantee holds on every CPython build, not only the GIL-enabled one.
    The lock must be reentrant to allow ``_embed_one`` to call back into
    ``_cache_get`` / ``_cache_put`` without self-deadlock.
    """
    svc = EmbeddingService.__new__(EmbeddingService)
    # ``_cache_lock`` is part of the contract — the public API is "no
    # KeyError under stress", but the implementation lever is the lock.
    # Inspecting the attribute directly keeps the regression guard cheap
    # and unambiguous compared to a probabilistic stress harness.
    EmbeddingService.__init__(svc, model_name="stub-non-e5")
    lock = svc._cache_lock
    # RLock factory returns a private ``_thread.RLock`` instance; check the
    # reentrant behaviour by acquiring twice from the same thread.
    assert lock.acquire(blocking=False)
    try:
        assert lock.acquire(blocking=False), "_cache_lock must be reentrant"
        lock.release()
    finally:
        lock.release()


def test_vec_type_alias_is_importable_from_models() -> None:
    """The ``Vec`` type alias is part of the public surface.

    External callers that want to type-annotate code against CrossMem's
    embedding contract must be able to import ``Vec`` from
    :mod:`crossmem.core.models` without reaching into private names.
    """
    from crossmem.core.models import Vec

    # ``Vec`` is a parameterised type alias — at runtime that's a
    # ``types.GenericAlias`` instance; the important contract is that the
    # symbol resolves and behaves as ``tuple[float, ...]`` (immutable
    # container, see ``Document.embedding``).
    assert Vec is not None
    assert Vec == tuple[float, ...]
