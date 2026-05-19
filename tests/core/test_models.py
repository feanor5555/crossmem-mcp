"""Tests for Document and Metadata dataclasses."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from crossmem.core.models import (
    Document,
    Metadata,
    generate_content_hash,
    generate_id,
)


def test_generate_id_deterministic() -> None:
    """Same inputs must produce the same 32-char hex ID."""
    id1 = generate_id("default", "https://example.com", "abc123")
    id2 = generate_id("default", "https://example.com", "abc123")
    assert id1 == id2
    assert len(id1) == 32
    assert all(c in "0123456789abcdef" for c in id1)


def test_generate_id_different_inputs() -> None:
    """Different inputs must produce different IDs."""
    id1 = generate_id("default", "https://example.com", "abc123")
    id2 = generate_id("other", "https://example.com", "abc123")
    id3 = generate_id("default", "https://other.com", "abc123")
    id4 = generate_id("default", "https://example.com", "xyz789")
    assert len({id1, id2, id3, id4}) == 4


def test_generate_id_chunk_index_disambiguates() -> None:
    """chunk_index produces distinct IDs for otherwise identical inputs.

    Same (namespace, source_url, content_hash) but different chunk indices
    must yield different IDs so duplicate-content chunks in the same
    document can never collide and silently overwrite each other.
    """
    base = generate_id("default", "https://example.com", "abc123")
    zero = generate_id("default", "https://example.com", "abc123", chunk_index=0)
    one = generate_id("default", "https://example.com", "abc123", chunk_index=1)
    two = generate_id("default", "https://example.com", "abc123", chunk_index=2)
    # Indexed and unindexed forms are intentionally distinct so callers can't
    # accidentally mix the two namespaces.
    assert len({base, zero, one, two}) == 4
    # Deterministic across calls.
    assert zero == generate_id(
        "default", "https://example.com", "abc123", chunk_index=0
    )


def test_generate_content_hash() -> None:
    """Content hash must match known SHA-256 digest."""
    content = "hello world"
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert generate_content_hash(content) == expected


def test_document_creation() -> None:
    """Document with all fields can be created and fields are accessible."""
    meta = Metadata(
        source_url="https://docs.python.org/3/",
        title="Python Docs",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
        namespace="default",
        tags=["python", "python:3.12"],
        content_hash="abc123",
    )
    doc = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="Python documentation summary",
        embedding=[0.1] * 384,
        metadata=meta,
    )
    assert doc.id == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
    assert doc.content == "Python documentation summary"
    assert len(doc.embedding) == 384
    assert doc.metadata.source_url == "https://docs.python.org/3/"
    assert doc.metadata.title == "Python Docs"
    assert doc.metadata.source_type == "web"
    assert doc.metadata.namespace == "default"
    assert doc.metadata.tags == ("python", "python:3.12")
    assert doc.metadata.embedding_dim == 384


def test_metadata_defaults() -> None:
    """Metadata tags default to empty tuple, namespace to 'default'."""
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
    )
    assert meta.tags == ()
    assert meta.namespace == "default"
    assert meta.content_hash == ""

    # Tags are now an immutable tuple — no shared-default hazard possible.
    meta2 = Metadata(
        source_url="https://other.com",
        title="Other",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
    )
    assert meta2.tags == ()


def test_metadata_is_frozen() -> None:
    """Metadata is immutable: attribute assignment raises FrozenInstanceError."""
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.title = "Mutated"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.tags = ["python"]  # type: ignore[misc]


def test_document_is_frozen() -> None:
    """Document is immutable: attribute assignment raises FrozenInstanceError."""
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
    )
    doc = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="hello",
        embedding=[0.0] * 384,
        metadata=meta,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.content = "mutated"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.embedding = [1.0] * 384  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.metadata = meta  # type: ignore[misc]


def test_metadata_tags_truly_immutable() -> None:
    """``Metadata.tags`` is a tuple — mutation methods do not exist.

    The ``@dataclass(frozen=True)`` decorator only blocks attribute
    reassignment. Without converting the underlying container, callers
    could still call ``meta.tags.append(...)`` or ``meta.tags.clear()``
    and silently mutate "frozen" state.
    """
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
        tags=["python", "asyncio"],
    )
    assert isinstance(meta.tags, tuple)
    with pytest.raises(AttributeError):
        meta.tags.append("new")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        meta.tags.clear()  # type: ignore[attr-defined]


def test_document_embedding_truly_immutable() -> None:
    """``Document.embedding`` is a tuple — mutation methods do not exist."""
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
    )
    doc = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="hello",
        embedding=[0.1, 0.2, 0.3],
        metadata=meta,
    )
    assert isinstance(doc.embedding, tuple)
    with pytest.raises(AttributeError):
        doc.embedding.append(0.0)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        doc.embedding.clear()  # type: ignore[attr-defined]


def test_document_is_hashable_and_set_compatible() -> None:
    """Tuple fields restore the ``__hash__`` that mutable fields broke.

    ``@dataclass(frozen=True)`` auto-generates ``__hash__`` only when
    every field is itself hashable. With ``list``-typed fields the
    Document silently fell back to the mutable-instance behaviour; with
    tuples ``hash(doc)`` works and the doc is usable as a dict key /
    set member.
    """
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
        tags=["python"],
    )
    doc1 = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="hello",
        embedding=[0.1, 0.2, 0.3],
        metadata=meta,
    )
    doc2 = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="hello",
        embedding=[0.1, 0.2, 0.3],
        metadata=meta,
    )
    # hash() must not raise
    assert hash(doc1) == hash(doc2)
    # set membership works on the equal instance
    assert doc1 in {doc2}
    assert len({doc1, doc2}) == 1


def test_factory_coerces_list_inputs_to_tuple() -> None:
    """``Document.from_payload`` accepts list inputs but stores tuples."""
    doc = Document.from_payload(
        content="hello",
        source_url="https://example.com",
        title="Example",
        source_type="web",
        tags=["python", "asyncio"],
        embedding=[0.1, 0.2, 0.3],
        embedding_model="m",
    )
    assert isinstance(doc.embedding, tuple)
    assert isinstance(doc.metadata.tags, tuple)
    assert doc.embedding == (0.1, 0.2, 0.3)
    assert doc.metadata.tags == ("python", "asyncio")


def test_from_dict_coerces_to_tuple() -> None:
    """``Document.from_dict`` rehydrates list-serialised fields as tuples."""
    meta = Metadata(
        source_url="https://example.com",
        title="Example",
        source_type="web",
        stored_at="2025-01-15T10:30:00Z",
        embedding_model="m",
        embedding_dim=3,
        tags=["python"],
    )
    doc = Document(
        id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        content="hello",
        embedding=[0.1, 0.2, 0.3],
        metadata=meta,
    )
    restored = Document.from_dict(doc.to_dict())
    assert isinstance(restored.embedding, tuple)
    assert isinstance(restored.metadata.tags, tuple)
    assert restored == doc
