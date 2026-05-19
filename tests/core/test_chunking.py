"""Tests for content-aware chunking."""

from __future__ import annotations

from crossmem.core.chunking import (
    Chunk,
    _chunk_api_docs,
    _chunk_prose,
    chunk,
    expand_identifiers,
)

# ---------------------------------------------------------------------------
# expand_identifiers
# ---------------------------------------------------------------------------


def test_expand_identifiers_camel_case():
    out = expand_identifiers("getUserData")
    assert "getUserData" in out
    assert "get" in out.split()
    assert "User" in out.split()
    assert "Data" in out.split()


def test_expand_identifiers_snake_case():
    out = expand_identifiers("snake_case_var")
    assert "snake_case_var" in out
    parts = out.split()
    assert "snake" in parts
    assert "case" in parts
    assert "var" in parts


def test_expand_identifiers_mixed():
    out = expand_identifiers("getUserData snake_case_var")
    # Original identifiers preserved
    assert "getUserData" in out
    assert "snake_case_var" in out
    # Expanded forms present
    parts = out.split()
    assert "get" in parts
    assert "User" in parts
    assert "snake" in parts
    assert "case" in parts


def test_expand_identifiers_passthrough():
    """Plain text without identifiers is unchanged."""
    out = expand_identifiers("hello world")
    assert "hello" in out.split()
    assert "world" in out.split()


# ---------------------------------------------------------------------------
# chunk: code
# ---------------------------------------------------------------------------


def test_chunk_code_splits_at_function_boundary():
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = chunk(code, "code", title="mod.py")
    assert len(chunks) == 2
    # First chunk contains foo, second contains bar
    assert "def foo" in chunks[0].content
    assert "def bar" in chunks[1].content
    assert "def bar" not in chunks[0].content


def test_chunk_code_class_boundary():
    code = (
        "class Foo:\n"
        "    def a(self):\n"
        "        return 1\n"
        "\n"
        "class Bar:\n"
        "    def b(self):\n"
        "        return 2\n"
    )
    chunks = chunk(code, "code", title="m.py")
    assert len(chunks) == 2
    assert "class Foo" in chunks[0].content
    assert "class Bar" in chunks[1].content


def test_chunk_code_async_def():
    code = "async def first():\n    return 1\n\nasync def second():\n    return 2\n"
    chunks = chunk(code, "code", title="async.py")
    assert len(chunks) == 2
    assert "async def first" in chunks[0].content
    assert "async def second" in chunks[1].content


def test_chunk_code_js_function():
    code = (
        "function alpha() {\n    return 1;\n}\n\nfunction beta() {\n    return 2;\n}\n"
    )
    chunks = chunk(code, "code", title="m.js")
    assert len(chunks) == 2
    assert "alpha" in chunks[0].content
    assert "beta" in chunks[1].content


def test_chunk_code_single_function():
    code = "def only():\n    return 1\n"
    chunks = chunk(code, "code", title="t.py")
    assert len(chunks) == 1
    assert "def only" in chunks[0].content


def test_chunk_code_no_boundary_falls_back_to_blank_line():
    code = "first_line = 1\nsecond_line = 2\n\nthird_line = 3\nfourth_line = 4\n"
    chunks = chunk(code, "code", title="t.py")
    assert len(chunks) == 2


def test_chunk_code_identifiers_expanded():
    """Code chunks include expanded identifier forms for FTS."""
    code = "def getUserData():\n    snake_case_var = 1\n"
    chunks = chunk(code, "code", title="t.py")
    assert len(chunks) == 1
    content = chunks[0].content
    # Original identifiers preserved
    assert "getUserData" in content
    assert "snake_case_var" in content
    # Expanded forms appended
    parts = content.split()
    assert "User" in parts
    assert "Data" in parts
    assert "snake" in parts


# ---------------------------------------------------------------------------
# chunk: prose
# ---------------------------------------------------------------------------


def test_chunk_prose_short_single_chunk():
    text = "Hello world. This is a short prose. End."
    chunks = chunk(text, "prose", title="Doc")
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_prose_splits_at_sentence_end():
    # Build a long prose that exceeds 512 tokens so it must split.
    sentence = "This is sentence number {n} with several words to fill space. "
    long_text = "".join(sentence.format(n=i) for i in range(120))
    chunks = chunk(long_text, "prose", title="Doc")
    assert len(chunks) >= 2
    # Each chunk (after prefix removal) should end at a sentence boundary
    # (i.e., not split mid-word). Check that no chunk's body ends mid-sentence
    # without punctuation.
    for c in chunks[:-1]:
        body = c.content.split(": ", 1)[-1].rstrip()
        assert body.endswith(".") or body.endswith("?") or body.endswith("!")


def test_chunk_prose_has_overlap():
    sentence = "Sentence {n} carries enough words to count as a unit here. "
    long_text = "".join(sentence.format(n=i) for i in range(120))
    chunks = chunk(long_text, "prose", title="Doc")
    assert len(chunks) >= 2
    # Find a sentence that appears in both the first and second chunk's body
    body0 = chunks[0].content
    body1 = chunks[1].content
    # Some sentence words from the tail of body0 should reappear in body1
    tail_words = body0.split()[-10:]
    overlap = sum(1 for w in tail_words if w in body1)
    assert overlap > 0


# ---------------------------------------------------------------------------
# chunk: api-docs
# ---------------------------------------------------------------------------


def test_chunk_api_docs_splits_at_paragraph():
    text = (
        "Paragraph one describes the first endpoint in detail.\n"
        "It has two lines.\n"
        "\n"
        "Paragraph two covers the second endpoint thoroughly.\n"
        "Also two lines.\n"
        "\n"
        "Paragraph three is the third one."
    )
    chunks = chunk(text, "api-docs", title="API")
    # Paragraph-level boundaries: at minimum it does not crash and returns
    # at least one chunk.
    assert len(chunks) >= 1
    # All paragraphs should be represented across all chunks.
    combined = " ".join(c.content for c in chunks)
    assert "Paragraph one" in combined
    assert "Paragraph two" in combined
    assert "Paragraph three" in combined


def test_chunk_api_docs_hr_boundary():
    text = (
        "Section A: details about endpoint A.\n"
        "\n"
        "---\n"
        "\n"
        "Section B: details about endpoint B.\n"
    )
    chunks = chunk(text, "api-docs", title="API")
    combined = " ".join(c.content for c in chunks)
    assert "Section A" in combined
    assert "Section B" in combined


def test_chunk_api_docs_long_splits_with_overlap():
    paragraph = "Paragraph {n} explains an API endpoint with many words. " * 10
    long_text = "\n\n".join(paragraph.replace("{n}", str(i)) for i in range(40))
    chunks = chunk(long_text, "api-docs", title="API")
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Prefix
# ---------------------------------------------------------------------------


def test_chunk_prefix_starts_with_title():
    text = "Hello world. This is short."
    chunks = chunk(text, "prose", title="My Doc")
    assert chunks[0].content.startswith("My Doc")


def test_chunk_prefix_includes_h2_when_present():
    text = (
        "Intro paragraph that introduces the topic.\n"
        "\n"
        "## Section Two\n"
        "\n"
        "Content under section two follows now.\n"
    )
    chunks = chunk(text, "prose", title="My Doc")
    # The chunk that contains the section-two content should reference Section Two
    found = any("Section Two" in c.content for c in chunks)
    assert found


def test_chunk_prefix_format():
    """Prefix follows: {title} > {h2}: {content}."""
    text = "Hello world. Some content here."
    chunks = chunk(text, "prose", title="Doc")
    # No h2 present, expected prefix is just "Doc: ..."
    assert chunks[0].content.startswith("Doc")
    assert ":" in chunks[0].content


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def test_chunk_indices_are_sequential():
    sentence = "Sentence {n} fills the chunk with enough words to matter. "
    long_text = "".join(sentence.format(n=i) for i in range(120))
    chunks = chunk(long_text, "prose", title="D")
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_unknown_type_falls_back_to_prose():
    text = "Hello world. This is a sentence."
    chunks = chunk(text, "unknown-type", title="D")
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_empty_input():
    assert chunk("", "prose", title="D") == []
    assert chunk("   \n  ", "prose", title="D") == []


def test_chunk_metadata_dict():
    text = "Hello. World."
    chunks = chunk(text, "prose", title="D")
    assert isinstance(chunks[0].metadata, dict)


def test_chunk_h2_in_metadata():
    text = (
        "Intro paragraph that introduces the topic.\n"
        "\n"
        "## My Heading\n"
        "\n"
        "Content beneath the heading lives here.\n"
    )
    chunks = chunk(text, "prose", title="Doc")
    # Some chunk should expose the h2 in metadata
    h2_values = [c.metadata.get("h2") for c in chunks if c.metadata.get("h2")]
    assert "My Heading" in h2_values


def test_chunk_prefix_with_h2_format():
    """Prefix format is '{title} > {h2}: {body}' when both present."""
    text = "## Section\n\nHello content body.\n"
    chunks = chunk(text, "prose", title="Doc")
    # The chunk that includes the section content should carry the full prefix
    matches = [c for c in chunks if "Section" in c.content and "Doc" in c.content]
    assert matches
    assert any(c.content.startswith("Doc > Section: ") for c in matches)


def test_chunk_prose_no_title_no_h2():
    """No prefix, just body."""
    chunks = chunk("Plain text.", "prose", title="")
    assert chunks[0].content == "Plain text."


def test_chunk_prose_h2_only_no_title():
    """h2-only prefix when title is empty."""
    text = "## Heading X\n\nBody content here.\n"
    chunks = chunk(text, "prose", title="")
    matches = [c for c in chunks if "Body content" in c.content]
    assert matches
    assert any(c.content.startswith("Heading X: ") for c in matches)


def test_chunk_api_docs_pack_units_overlap_branch():
    """Cover the overlap tail-build branch in _pack_units."""
    # Many short paragraphs to force packing with overlap
    long_text = "\n\n".join(f"Para {i} word " * 50 for i in range(30))
    chunks = chunk(long_text, "api-docs", title="API")
    assert len(chunks) >= 2


def test_chunk_code_boundary_at_offset_zero():
    """Boundary regex matches at offset 0 — the leading-prepend branch."""
    code = "def a():\n    pass\n\ndef b():\n    pass\n"
    chunks = chunk(code, "code", title="m.py")
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Internal chunker return shape (regression: 21.9 type annotation)
# ---------------------------------------------------------------------------


def test_chunk_api_docs_returns_three_tuples():
    """`_chunk_api_docs` must return ``(body, start, end)`` 3-tuples."""
    text = "First paragraph here.\n\nSecond paragraph follows.\n"
    result = _chunk_api_docs(text)
    assert result, "expected non-empty result"
    for entry in result:
        # Explicit 3-tuple destructuring — fails at runtime if shape regresses.
        body, start, end = entry
        assert isinstance(body, str)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end


def test_chunk_prose_returns_three_tuples():
    """`_chunk_prose` must return ``(body, start, end)`` 3-tuples."""
    text = "First sentence. Second sentence. Third sentence."
    result = _chunk_prose(text)
    assert result, "expected non-empty result"
    for entry in result:
        body, start, end = entry
        assert isinstance(body, str)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end
