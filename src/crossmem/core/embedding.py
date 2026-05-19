"""Embedding service — fastembed wrapper with LRU cache and SHA-pinned model.

fp16 precision policy
---------------------
Vectors leave fastembed as fp32 numpy arrays. We immediately round them to
fp16 (``np.array(vec, dtype=np.float16)``) before caching and before
returning them to callers. The rounding happens at two sites — the
single-text path (``_compute_embedding``) and the batched path
(``_embed_many``) — and the resulting half-precision values are what every
downstream consumer sees, including the SQLite backend.

sqlite-vec's ``vec0`` virtual table only accepts ``float[N]`` (fp32) storage,
so the backend casts the fp16-rounded tuple back to fp32 on insert. That
cast widens the type but does NOT recover the bits that were truncated by
the fp16 round-trip: the stored fp32 vector carries fp16-precision
artefacts forever. This is intentional — fp16 cuts the in-memory LRU
footprint roughly in half and the recall impact is negligible at 384
dimensions — but it must be visible to anyone reading the code, because
"why is the stored fp32 not bit-exact with what fastembed produces?" is a
predictable question.

See the "Embedding" section in ``CLAUDE.md`` for the project-level
rationale (cache footprint, sqlite-vec vec0 limitation, model choice).
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

# Default model — must be in fastembed's supported list. The spec originally
# named ``intfloat/multilingual-e5-small`` but only the ``-large`` e5 variant
# is registered in current fastembed, so we ship the 384-dim multilingual
# MiniLM as the default. Same dimension, same multilingual coverage, ~120MB
# smaller, no PyTorch dependency.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
_CACHE_MAX_SIZE = 512

# e5 task prefixes — the multilingual-e5 family was trained with explicit
# "query: " / "passage: " prefixes; mixing or omitting them measurably
# degrades retrieval quality. MiniLM and other non-e5 models were trained
# without these prefixes, so we apply them only when the model name
# identifies an e5 model.
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


def _uses_e5_prefixes(model_name: str) -> bool:
    """Return True if the model name identifies an e5 family model."""
    return "e5" in model_name.lower()


# SHA256 pin registry — drift detection. A swapped or corrupted model file
# changes embedding semantics silently, so we hash the loaded ONNX file and
# refuse to serve queries on mismatch. Models without a registered hash are
# loaded without verification. Set ``CROSSMEM_SKIP_MODEL_SHA_PIN=1`` to
# bypass the check entirely.
EMBEDDING_MODEL_SHA256: dict[str, str] = {
    # qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q :: model_optimized.onnx
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
        "634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99"
    ),
}

_SKIP_PIN_ENV = "CROSSMEM_SKIP_MODEL_SHA_PIN"


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        # Reentrant lock around every cache mutation. The GIL serialises
        # individual ``OrderedDict`` ops, but concurrent ``embed_query``
        # threads can still interleave ``move_to_end`` (in ``_cache_get``)
        # and ``popitem(last=False)`` (in ``_cache_put``) across multiple
        # bytecodes, breaking LRU ordering and — on free-threaded
        # (PEP 703) builds — exposing ``KeyError``. Reentrancy keeps the
        # batch path (``_embed_many``) free to call ``_cache_get`` /
        # ``_cache_put`` repeatedly inside the same critical section.
        self._cache_lock = threading.RLock()
        self._ready = threading.Event()
        self._warmup_error: BaseException | None = None
        threading.Thread(target=self._warmup, daemon=True).start()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _warmup(self) -> None:
        try:
            self._model = TextEmbedding(model_name=self._model_name)
            self._verify_model_sha()
        except BaseException as exc:  # noqa: BLE001 - propagated via _ensure_ready
            self._warmup_error = exc
        finally:
            self._ready.set()

    def _verify_model_sha(self) -> None:
        """Fail-fast if the loaded model file's SHA256 differs from the pin.

        Skipped when ``CROSSMEM_SKIP_MODEL_SHA_PIN=1`` or when the current
        model name has no entry in :data:`EMBEDDING_MODEL_SHA256`.
        """
        if os.environ.get(_SKIP_PIN_ENV) == "1":
            return
        expected = EMBEDDING_MODEL_SHA256.get(self._model_name)
        if expected is None:
            return
        path = _locate_model_file(self._model)
        if path is None:
            return
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Embedding model SHA256 mismatch: expected {expected}, "
                f"got {actual}. Model: {self._model_name} at {path}. "
                "Refusing to load — embedding semantics may have drifted."
            )

    def _ensure_ready(self) -> None:
        if not self._ready.is_set():
            self._ready.wait()
        if self._warmup_error is not None:
            raise self._warmup_error

    def embed(self, text: str) -> list[float]:
        return list(self._embed_one(text))

    def embed_query(self, text: str) -> list[float]:
        """Embed a short user query.

        For e5 models the ``"query: "`` prefix is prepended (e5 was trained
        with this convention; mixing or omitting it degrades retrieval).
        Non-e5 models receive the raw text.

        Precision: the returned ``list[float]`` carries fp16-rounded values
        (see the module docstring). The transport type is ``list[float]``
        but the bits originate from ``np.float16``; callers MUST NOT assume
        bit-exact fp32 fidelity.
        """
        return self.embed(self._with_query_prefix(text))

    def embed_passage(self, text: str) -> list[float]:
        """Embed an indexed passage.

        For e5 models the ``"passage: "`` prefix is prepended. Non-e5 models
        receive the raw text — they were trained without the convention.

        Precision: the returned ``list[float]`` carries fp16-rounded values
        (see the module docstring). The transport type is ``list[float]``
        but the bits originate from ``np.float16``; callers MUST NOT assume
        bit-exact fp32 fidelity.
        """
        return self.embed(self._with_passage_prefix(text))

    def embed_passage_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed many passages in a single fastembed call when cache misses.

        The LRU cache is consulted per input (so already-seen passages do
        not hit the model) and freshly computed vectors are written back to
        the cache — same key normalisation and same eviction policy as
        :meth:`embed`. The prefix logic mirrors :meth:`embed_passage`.

        The original ``embed_batch`` API bypassed the LRU cache entirely
        and was never wired up from production code; this method replaces
        it and is invoked by :meth:`KnowledgeStore.store` to amortise the
        per-call model overhead across all chunks of one document
        (~1.5ms/chunk vs ~50ms when called sequentially).

        Precision: each returned ``list[float]`` carries fp16-rounded
        values (see the module docstring). The transport type is
        ``list[float]`` but the bits originate from ``np.float16``;
        callers MUST NOT assume bit-exact fp32 fidelity, even on a
        roundtrip through the fp32-typed sqlite-vec ``vec0`` store.
        """
        prefixed = [self._with_passage_prefix(t) for t in texts]
        return self._embed_many(prefixed, batch_size=batch_size)

    # ------------------------------------------------------------------
    # internal helpers — single source of truth for cache + prefix logic
    # ------------------------------------------------------------------

    def _with_query_prefix(self, text: str) -> str:
        if _uses_e5_prefixes(self._model_name):
            return _QUERY_PREFIX + text
        return text

    def _with_passage_prefix(self, text: str) -> str:
        if _uses_e5_prefixes(self._model_name):
            return _PASSAGE_PREFIX + text
        return text

    def _cache_key(self, text: str) -> str:
        return text.lower().strip()

    def _cache_get(self, key: str) -> tuple[float, ...] | None:
        # Atomic get + LRU bump under ``_cache_lock``. Without the lock the
        # ``move_to_end`` between two threads could race with eviction in
        # ``_cache_put`` (see ``__init__`` for the full rationale).
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return cached

    def _cache_put(self, key: str, vec: tuple[float, ...]) -> None:
        # Atomic insert + bounded eviction. The check-then-pop pattern must
        # not interleave with another thread's ``popitem`` — otherwise both
        # threads can observe ``len > _CACHE_MAX_SIZE`` and over-evict, or
        # (on free-threaded builds) raise ``KeyError`` on an emptied dict.
        with self._cache_lock:
            self._cache[key] = vec
            if len(self._cache) > _CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        """Cache-aware single embedding — shared by ``embed`` and the batch path.

        The model call itself is intentionally NOT held under the lock —
        fastembed's ``embed`` releases the GIL and can take milliseconds,
        and serialising it would defeat the warmup parallelism. The cache
        accesses ARE serialised by the lock that ``_cache_get`` and
        ``_cache_put`` acquire individually.
        """
        self._ensure_ready()
        key = self._cache_key(text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        vec = self._compute_embedding(key)
        self._cache_put(key, vec)
        return vec

    def _embed_many(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Cache-aware batched embedding — shared by ``embed_passage_batch``.

        Texts already present in the LRU are returned from cache without
        invoking the model. The remaining unique texts are computed in
        fastembed batches of ``batch_size`` (amortising the per-call
        overhead) and written back to the cache before the call returns.
        """
        self._ensure_ready()
        assert self._model is not None  # _ensure_ready guarantees this

        keys = [self._cache_key(t) for t in texts]
        results: list[tuple[float, ...] | None] = [None] * len(texts)

        # First pass: serve cache hits and collect unique misses preserving
        # insertion order. Duplicates within ``texts`` share a single model
        # call and a single cache entry.
        pending: dict[str, list[int]] = {}
        for idx, key in enumerate(keys):
            cached = self._cache_get(key)
            if cached is not None:
                results[idx] = cached
            else:
                pending.setdefault(key, []).append(idx)

        # Second pass: batch the unique missing keys through the model.
        miss_keys = list(pending)
        for start in range(0, len(miss_keys), batch_size):
            batch = miss_keys[start : start + batch_size]
            for key, vec in zip(batch, self._model.embed(batch), strict=True):
                # Round fp32 -> fp16 once here. The half-precision tuple is
                # what we cache and what every caller (including the SQLite
                # backend, which widens back to fp32 for vec0 storage) sees.
                # The fp16 rounding artefacts persist in the stored fp32
                # vector — see module docstring + "Embedding" in CLAUDE.md.
                vec_f16 = tuple(np.array(vec, dtype=np.float16).tolist())
                self._cache_put(key, vec_f16)
                for idx in pending[key]:
                    results[idx] = vec_f16

        return [list(vec) for vec in results]  # type: ignore[arg-type]

    def _compute_embedding(self, text: str) -> tuple[float, ...]:
        assert self._model is not None  # _ensure_ready guarantees this
        embeddings = list(self._model.embed([text]))
        vec = embeddings[0]
        # fp32 -> fp16 rounding mirrors ``_embed_many`` (see module docstring +
        # "Embedding" in CLAUDE.md). The downstream fp32 store in sqlite-vec
        # widens the type but keeps the rounding artefacts.
        vec_f16 = np.array(vec, dtype=np.float16).tolist()
        return tuple(vec_f16)


def _locate_model_file(text_embedding: TextEmbedding | None) -> Path | None:
    """Return the on-disk ONNX model file for a loaded ``TextEmbedding``.

    Returns ``None`` if the path cannot be derived (e.g. fastembed internals
    changed). Callers treat ``None`` as "skip the pin check" so a private-API
    drift never bricks the runtime — only an actual SHA mismatch does.
    """
    if text_embedding is None:
        return None
    try:
        inner = text_embedding.model
        cache_dir = Path(inner.cache_dir)
        desc = inner.model_description
        hf_source = desc.sources.hf
        if not hf_source:
            return None
        model_file_name = desc.model_file
        hf_dir = cache_dir / f"models--{hf_source.replace('/', '--')}"
        snapshots_dir = hf_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        for snap in snapshots_dir.iterdir():
            candidate = snap / model_file_name
            if candidate.is_file():
                return candidate
    except (AttributeError, OSError):
        return None
    return None
