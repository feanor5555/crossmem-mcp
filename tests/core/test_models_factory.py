"""Tests for Document.from_payload and Document.from_dict factories."""

from __future__ import annotations

import pytest

from crossmem.core.models import (
    Document,
    Metadata,
    generate_content_hash,
    generate_id,
)


def test_from_payload_minimal_defaults() -> None:
    """from_payload supplies sensible defaults for optional fields."""
    doc = Document.from_payload(
        content="hello",
        source_url="https://example.com/a",
        title="A",
        source_type="web",
    )
    assert doc.content == "hello"
    assert doc.embedding == ()
    assert doc.metadata.source_url == "https://example.com/a"
    assert doc.metadata.title == "A"
    assert doc.metadata.source_type == "web"
    assert doc.metadata.namespace == "default"
    assert doc.metadata.tags == ()
    assert doc.metadata.embedding_model == ""
    assert doc.metadata.embedding_dim == 0
    # content_hash and id derived deterministically
    assert doc.metadata.content_hash == generate_content_hash("hello")
    assert doc.id == generate_id(
        "default", "https://example.com/a", doc.metadata.content_hash
    )
    # stored_at populated as a non-empty ISO string
    assert doc.metadata.stored_at != ""
    assert "T" in doc.metadata.stored_at


def test_from_payload_explicit_fields() -> None:
    """Explicit fields override defaults."""
    doc = Document.from_payload(
        content="content",
        source_url="https://docs.example.com",
        title="Docs",
        source_type="github",
        namespace="alice",
        tags=["python", "asyncio"],
        embedding=[0.1, 0.2, 0.3],
        embedding_model="model-x",
        embedding_dim=3,
        stored_at="2025-01-15T10:30:00+00:00",
    )
    assert doc.metadata.namespace == "alice"
    assert doc.metadata.tags == ("python", "asyncio")
    assert doc.embedding == (0.1, 0.2, 0.3)
    assert doc.metadata.embedding_model == "model-x"
    assert doc.metadata.embedding_dim == 3
    assert doc.metadata.stored_at == "2025-01-15T10:30:00+00:00"
    assert doc.id == generate_id(
        "alice", "https://docs.example.com", doc.metadata.content_hash
    )


def test_from_payload_deterministic_id() -> None:
    """Identical inputs always yield the same id and content_hash."""
    a = Document.from_payload(
        content="same content",
        source_url="https://example.com/x",
        title="Title",
        source_type="web",
        stored_at="2025-01-15T10:30:00+00:00",
    )
    b = Document.from_payload(
        content="same content",
        source_url="https://example.com/x",
        title="Different Title",
        source_type="web",
        stored_at="2099-12-31T23:59:59+00:00",
    )
    assert a.id == b.id
    assert a.metadata.content_hash == b.metadata.content_hash


def test_from_payload_chunk_index_disambiguates_duplicate_content() -> None:
    """Same-content payloads with different chunk_index get distinct IDs."""
    args = {
        "content": "shared content",
        "source_url": "https://example.com/dup",
        "title": "Dup",
        "source_type": "web",
        "stored_at": "2025-01-15T10:30:00+00:00",
    }
    a = Document.from_payload(**args, chunk_index=0)
    b = Document.from_payload(**args, chunk_index=1)
    assert a.id != b.id
    # content_hash still reflects only the content body.
    assert a.metadata.content_hash == b.metadata.content_hash
    # IDs are deterministic.
    assert a.id == Document.from_payload(**args, chunk_index=0).id
    # Default (no chunk_index) yields a third, distinct id namespace.
    c = Document.from_payload(**args)
    assert c.id not in (a.id, b.id)


def test_from_payload_embedding_dim_inferred() -> None:
    """When embedding_dim left at default and embedding given, dim is len()."""
    doc = Document.from_payload(
        content="x",
        source_url="https://example.com",
        title="t",
        source_type="web",
        embedding=[1.0, 2.0, 3.0, 4.0],
    )
    assert doc.metadata.embedding_dim == 4


def test_from_payload_tags_isolated() -> None:
    """Tags list passed in is copied — caller mutation doesn't leak.

    With ``tags`` stored as ``tuple[str, ...]`` the isolation falls out
    of the tuple-coercion automatically; this test guards the contract
    explicitly.
    """
    tags = ["a", "b"]
    doc = Document.from_payload(
        content="x",
        source_url="https://example.com",
        title="t",
        source_type="web",
        tags=tags,
    )
    tags.append("c")
    assert doc.metadata.tags == ("a", "b")


def test_from_dict_roundtrip() -> None:
    """from_dict(to_dict(doc)) reproduces an equal Document."""
    original = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="some content",
        embedding=[0.1, 0.2, 0.3],
        metadata=Metadata(
            source_url="https://example.com",
            title="Example",
            source_type="web",
            stored_at="2025-01-15T10:30:00+00:00",
            embedding_model="model-x",
            embedding_dim=3,
            namespace="alice",
            tags=["python", "asyncio"],
            content_hash="hash123",
        ),
    )
    restored = Document.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_roundtrip_via_from_payload() -> None:
    """A Document built via from_payload also survives a to_dict/from_dict roundtrip."""
    original = Document.from_payload(
        content="hello world",
        source_url="https://example.com/a",
        title="A",
        source_type="web",
        tags=["x"],
        embedding=[0.5] * 4,
        embedding_model="m",
    )
    restored = Document.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_metadata_optional_fields_have_defaults() -> None:
    """Missing optional metadata fields fall back to documented defaults."""
    payload = {
        "id": "x" * 32,
        "content": "c",
        "embedding": [],
        "metadata": {"source_url": "https://example.com"},
    }
    doc = Document.from_dict(payload)
    assert doc.metadata.title == ""
    assert doc.metadata.source_type == ""
    assert doc.metadata.namespace == "default"
    assert doc.metadata.tags == ()
    assert doc.metadata.content_hash == ""
    assert doc.metadata.embedding_dim == 0


def test_from_dict_missing_required_field_raises() -> None:
    """from_dict raises ValueError when a required field is missing."""
    payload = {
        "id": "x",
        "content": "c",
        "embedding": [],
        # metadata missing -> KeyError -> ValueError
    }
    with pytest.raises(ValueError, match="missing required field"):
        Document.from_dict(payload)


def test_from_dict_missing_embedding_message_lists_keys() -> None:
    """Missing ``embedding`` yields an explicit error naming the field and keys."""
    payload = {
        "id": "x" * 32,
        "content": "c",
        "metadata": {"source_url": "https://example.com"},
    }
    with pytest.raises(ValueError) as exc_info:
        Document.from_dict(payload)
    msg = str(exc_info.value)
    assert "Document.from_dict" in msg
    assert "'embedding'" in msg
    assert "got keys=" in msg
    # Each top-level key in the payload should be referenced
    for key in payload:
        assert key in msg
