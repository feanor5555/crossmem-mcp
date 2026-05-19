"""Shared deterministic embedder for the test suite.

The production ``_Embedder`` Protocol (see ``crossmem.core.store``) requires
three things from any embedder:

  * a public ``model_name`` attribute (read by ``KnowledgeStore.store`` for
    every chunk's metadata — no silent fallback),
  * a prefix-aware ``embed_query(text)`` method,
  * a prefix-aware ``embed_passage(text)`` method.

The real :class:`crossmem.core.embedding.EmbeddingService` additionally
exposes ``embed_passage_batch(texts, batch_size=...)`` so callers that
import in bulk (notably :meth:`KnowledgeStore.store`) can amortise model
overhead. ``FixedEmbedder`` mirrors the same shape so any test can drop
it in wherever the real service is wired — without depending on
fastembed, without downloading a 300 MB model, and without losing
determinism between runs.

Determinism note: ``embed_query`` and ``embed_passage`` prefix the input
text with ``"query: "`` / ``"passage: "`` before hashing, so a query and
a passage with the same surface text still produce different vectors.
This mirrors the e5 training convention used by the real embedder for e5
checkpoints; for the MiniLM default the prefixes are inert at the model
level but harmless for hash-based test vectors.
"""

from __future__ import annotations

import hashlib

from crossmem.core.embedding import EMBEDDING_DIM

__all__ = ["FixedEmbedder"]


class FixedEmbedder:
    """Deterministic in-process embedder for tests.

    Same text -> same 384-dim vector across runs and processes. No I/O,
    no model download, no network. Implements the full ``_Embedder``
    Protocol (``model_name``, ``embed_query``, ``embed_passage``,
    ``embed_passage_batch``) so :class:`KnowledgeStore` can call the batch
    entry-point without branching for tests.

    Parameters
    ----------
    model_name:
        Returned verbatim from the ``model_name`` attribute. Tests that
        assert on stored metadata pass a stable string (e.g. ``"mock"``)
        rather than the default.
    dim:
        Embedding dimension. Defaults to ``EMBEDDING_DIM`` (384) to match
        the pinned production model. Tests that exercise dimension
        validation pass a deliberately wrong value.
    """

    def __init__(self, model_name: str = "mock-test", dim: int = EMBEDDING_DIM) -> None:
        self.model_name = model_name
        self._dim = dim

    def _hash_vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(b) / 255.0 for b in h] * ((self._dim // len(h)) + 1)
        return raw[: self._dim]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vec("query: " + text)

    def embed_passage(self, text: str) -> list[float]:
        return self._hash_vec("passage: " + text)

    def embed_passage_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        # ``batch_size`` is accepted for Protocol-shape parity with the real
        # ``EmbeddingService.embed_passage_batch``; for the hash-based mock
        # the batching has no effect on the output.
        del batch_size
        return [self.embed_passage(t) for t in texts]
