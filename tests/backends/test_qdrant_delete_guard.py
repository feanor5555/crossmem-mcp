"""Tests for QdrantBackend.delete payload guard (task 26.8).

These tests stub ``qdrant_client`` in :mod:`sys.modules` and pass a
``MagicMock`` as the client, so they run without the optional
``qdrant-client`` dependency installed.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _install_qdrant_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the qdrant_client modules so QdrantBackend.__init__ imports succeed."""
    qclient = types.ModuleType("qdrant_client")
    qclient.QdrantClient = MagicMock()  # type: ignore[attr-defined]
    qhttp = types.ModuleType("qdrant_client.http")
    qmodels = types.ModuleType("qdrant_client.http.models")
    # Provide attribute access for the symbols the backend uses on qmodels.
    # The backend stores ``qmodels`` and reaches into it via attribute
    # access, so a SimpleNamespace-like ModuleType suffices.
    qmodels.PointIdsList = MagicMock(name="PointIdsList")  # type: ignore[attr-defined]
    qmodels.PointStruct = MagicMock(name="PointStruct")  # type: ignore[attr-defined]
    qmodels.VectorParams = MagicMock(name="VectorParams")  # type: ignore[attr-defined]
    qmodels.Distance = types.SimpleNamespace(COSINE="Cosine")  # type: ignore[attr-defined]
    qmodels.PayloadSchemaType = types.SimpleNamespace(KEYWORD="keyword")  # type: ignore[attr-defined]
    qmodels.Filter = MagicMock(name="Filter")  # type: ignore[attr-defined]
    qmodels.FieldCondition = MagicMock(name="FieldCondition")  # type: ignore[attr-defined]
    qmodels.MatchValue = MagicMock(name="MatchValue")  # type: ignore[attr-defined]
    qhttp.models = qmodels  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", qclient)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qhttp)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", qmodels)


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    # _ensure_collection: collection already exists -> skip create.
    client.get_collections.return_value = types.SimpleNamespace(
        collections=[types.SimpleNamespace(name="test_crossmem")]
    )
    return client


@pytest.fixture
def backend(
    monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
) -> Iterator[object]:
    _install_qdrant_stub(monkeypatch)
    # Reload the backend module so the stubbed imports take effect on
    # the module-level ``from qdrant_client...`` lines.
    if "crossmem.backends.qdrant_backend" in sys.modules:
        del sys.modules["crossmem.backends.qdrant_backend"]
    from crossmem.backends.qdrant_backend import QdrantBackend

    yield QdrantBackend(client=mock_client, collection_name="test_crossmem")


class TestQdrantDeletePayloadGuard:
    def test_delete_raises_on_payload_mismatch_and_skips_delete_call(
        self, backend, mock_client: MagicMock
    ) -> None:
        # retrieve returns one point whose payload identifies a DIFFERENT
        # CrossMem document — simulating a 64-bit hash collision.
        mock_client.retrieve.return_value = [
            types.SimpleNamespace(payload={"doc_id": "other-doc"})
        ]

        with pytest.raises(ValueError, match="point id collision"):
            backend.delete("d1")

        # delete must NOT have been called.
        mock_client.delete.assert_not_called()

    def test_delete_calls_retrieve_with_derived_point_id(
        self, backend, mock_client: MagicMock
    ) -> None:
        # Happy-path: retrieve returns a point whose payload matches.
        mock_client.retrieve.return_value = [
            types.SimpleNamespace(payload={"doc_id": "d1"})
        ]
        backend.delete("d1")

        # retrieve was called once with the derived point id.
        assert mock_client.retrieve.call_count == 1
        call = mock_client.retrieve.call_args
        assert call.kwargs["ids"] == [backend._point_id("d1")]
        assert call.kwargs["with_payload"] is True
        # And delete was called exactly once after the payload check passed.
        assert mock_client.delete.call_count == 1

    def test_delete_missing_point_is_noop(
        self, backend, mock_client: MagicMock
    ) -> None:
        mock_client.retrieve.return_value = []

        # Use a valid 32-hex doc_id so ``_point_id`` succeeds.
        backend.delete("a" * 32)

        mock_client.delete.assert_not_called()

    def test_delete_raises_when_payload_doc_id_is_none(
        self, backend, mock_client: MagicMock
    ) -> None:
        # Payload exists but is missing the doc_id field — also a mismatch.
        mock_client.retrieve.return_value = [types.SimpleNamespace(payload={})]

        with pytest.raises(ValueError, match="point id collision"):
            backend.delete("d1")
        mock_client.delete.assert_not_called()


class TestQdrantPointId:
    def test_point_id_is_full_uint64_range(self, backend) -> None:
        # First 16 hex chars all-F -> 0xFFFFFFFFFFFFFFFF = 2**64 - 1.
        # The previous 63-bit mask would have produced 0x7FFFFFFFFFFFFFFF.
        pid = backend._point_id("f" * 32)
        assert pid == (1 << 64) - 1

    def test_point_id_is_deterministic(self, backend) -> None:
        assert backend._point_id("abcdef0123456789") == backend._point_id(
            "abcdef0123456789ffffffffffffffff"
        )
