"""Content-aware chunking for CrossMem.

Splits text into overlapping chunks using strategies tailored to the
content type:

| Content type | Chunk size  | Overlap | Split boundary       |
|--------------|-------------|---------|----------------------|
| code         | 512 tokens  | 0%      | function/class regex |
| api-docs     | 256 tokens  | 10%     | paragraph / `<hr>`   |
| prose / web  | 512 tokens  | 15-20%  | sentence end         |

Each chunk receives a context prefix of ``"{title} > {h2}: {content}"``
(or ``"{title}: {content}"`` when no preceding h2 is found).

Token counts are approximated via ``len(text.split())``. This avoids a
hard dependency on tiktoken; the resulting bound is conservative for
typical English/multilingual text where one whitespace-separated word
maps to roughly one token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration constants (per spec)
# ---------------------------------------------------------------------------

CODE_CHUNK_TOKENS = 512
CODE_OVERLAP_RATIO = 0.0

API_DOCS_CHUNK_TOKENS = 256
API_DOCS_OVERLAP_RATIO = 0.10

PROSE_CHUNK_TOKENS = 512
PROSE_OVERLAP_RATIO = 0.18  # mid-point of the 15-20% range


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text with its context prefix and position info."""

    content: str  # already prefixed with "{title} > {h2}: ..."
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identifier expansion (used to enrich code chunks for FTS5 retrieval)
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_camel(token: str) -> list[str]:
    parts = _CAMEL_BOUNDARY.split(token)
    return [p for p in parts if p]


def _split_snake(token: str) -> list[str]:
    return [p for p in token.split("_") if p]


def expand_identifiers(code: str) -> str:
    """Return ``code`` plus space-separated camelCase / snake_case parts.

    For every identifier in ``code`` that is either camelCase or snake_case,
    the original is preserved and the split parts are appended. Plain
    lowercase / single-token identifiers are passed through unchanged.

    Example: ``"getUserData snake_case_var"`` becomes
    ``"getUserData snake_case_var get User Data snake case var"``.
    """
    extras: list[str] = []
    seen: set[str] = set()
    for match in _IDENT_RE.finditer(code):
        token = match.group(0)
        parts: list[str] = []
        if "_" in token:
            parts = _split_snake(token)
        elif _CAMEL_BOUNDARY.search(token):
            parts = _split_camel(token)
        for p in parts:
            if p == token or p in seen:
                continue
            seen.add(p)
            extras.append(p)
    if not extras:
        return code
    return code + " " + " ".join(extras)


# ---------------------------------------------------------------------------
# Token counting & windowing helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Approximate token count via whitespace split."""
    return len(text.split())


def _pack_units(
    units: list[tuple[str, int]],
    chunk_tokens: int,
    overlap_ratio: float,
    joiner: str,
) -> list[tuple[str, int, int]]:
    """Greedily pack atomic ``units`` into chunks of ~``chunk_tokens`` tokens.

    Each unit is a ``(text, source_offset)`` tuple. Once a chunk is emitted,
    the tail of it (``overlap_ratio * chunk_tokens`` tokens worth of
    trailing units) is reused as the start of the next chunk. ``joiner``
    is the string used to glue unit texts together. Returns a list of
    ``(joined_text, first_offset, last_offset)`` tuples.
    """
    if not units:
        return []
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for unit in units:
        unit_text, _ = unit
        unit_tokens = _count_tokens(unit_text)
        if current and current_tokens + unit_tokens > chunk_tokens:
            chunks.append(
                (
                    joiner.join(u for u, _ in current),
                    current[0][1],
                    current[-1][1],
                )
            )
            if overlap_ratio > 0:
                overlap_target = int(chunk_tokens * overlap_ratio)
                tail: list[tuple[str, int]] = []
                tail_tokens = 0
                for prev in reversed(current):
                    prev_tokens = _count_tokens(prev[0])
                    if tail_tokens + prev_tokens > overlap_target and tail:
                        break
                    tail.insert(0, prev)
                    tail_tokens += prev_tokens
                current = list(tail)
                current_tokens = tail_tokens
            else:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(
            (
                joiner.join(u for u, _ in current),
                current[0][1],
                current[-1][1],
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

# Match the start of a top-level Python function/class or JS/TS
# function/class on its own line. Top-level only (no leading whitespace) so
# that methods inside a class do not split the class definition. This is
# pragmatic: unusual layouts may fall through to the blank-line fallback.
_CODE_BOUNDARY_RE = re.compile(
    r"^(?:async\s+def\s+|def\s+|class\s+|function\s+|export\s+(?:default\s+)?(?:async\s+)?function\s+)",
    re.MULTILINE,
)


def _split_code_blocks(code: str) -> list[tuple[str, int]]:
    """Split ``code`` at function/class boundaries; fall back to blank lines.

    Returns ``(block_text, source_offset)`` tuples.
    """
    boundaries = [m.start() for m in _CODE_BOUNDARY_RE.finditer(code)]
    if len(boundaries) >= 2:
        # Prepend a leading boundary at offset 0 if not already there — it
        # captures any prelude before the first def/class.
        if boundaries[0] != 0:
            boundaries = [0, *boundaries]
        blocks: list[tuple[str, int]] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(code)
            block = code[start:end].strip("\n")
            if block.strip():
                blocks.append((block, start))
        if len(blocks) >= 2:
            return blocks
    # Fallback: split on blank-line gaps.
    fallback: list[tuple[str, int]] = []
    pos = 0
    for piece in re.split(r"(\n\s*\n)", code):
        if not piece:
            continue
        if not re.fullmatch(r"\n\s*\n", piece) and piece.strip():
            fallback.append((piece, pos))
        pos += len(piece)
    return fallback


def _chunk_code(code: str) -> list[tuple[str, int, int]]:
    """Chunk code at function/class boundaries (one block per chunk).

    Blocks larger than ``CODE_CHUNK_TOKENS`` are emitted as-is rather than
    sub-split: keeping a function intact preserves semantic context, which
    the spec values over a strict token cap. Identifier expansion is
    appended to each chunk to enrich FTS retrieval.
    """
    blocks = _split_code_blocks(code)
    if not blocks:
        return []
    return [(expand_identifiers(b), off, off) for b, off in blocks]


# ---------------------------------------------------------------------------
# API-docs chunking
# ---------------------------------------------------------------------------

# Paragraph or horizontal-rule boundary. Markdown HRs (---, ***, ___) on
# their own line are treated as paragraph breaks.
_API_DOCS_BOUNDARY_RE = re.compile(
    r"\n\s*\n|^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE
)


def _split_api_docs(text: str) -> list[tuple[str, int]]:
    """Split ``text`` at paragraph / HR boundaries; track source offsets."""
    paragraphs: list[tuple[str, int]] = []
    last_end = 0
    for m in _API_DOCS_BOUNDARY_RE.finditer(text):
        piece = text[last_end : m.start()]
        if piece.strip():
            offset = last_end + (len(piece) - len(piece.lstrip()))
            paragraphs.append((piece.strip(), offset))
        last_end = m.end()
    tail = text[last_end:]
    if tail.strip():
        offset = last_end + (len(tail) - len(tail.lstrip()))
        paragraphs.append((tail.strip(), offset))
    return paragraphs


def _chunk_api_docs(text: str) -> list[tuple[str, int, int]]:
    paragraphs = _split_api_docs(text)
    if not paragraphs:
        return []
    return _pack_units(
        paragraphs,
        chunk_tokens=API_DOCS_CHUNK_TOKENS,
        overlap_ratio=API_DOCS_OVERLAP_RATIO,
        joiner="\n\n",
    )


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------

# Split on sentence terminators followed by whitespace. Keeps the
# terminator attached to the preceding sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split ``text`` into sentences with source offsets."""
    sentences: list[tuple[str, int]] = []
    last_end = 0
    for m in _SENTENCE_RE.finditer(text):
        piece = text[last_end : m.start()]
        if piece.strip():
            offset = last_end + (len(piece) - len(piece.lstrip()))
            sentences.append((piece.strip(), offset))
        last_end = m.end()
    tail = text[last_end:]
    if tail.strip():
        offset = last_end + (len(tail) - len(tail.lstrip()))
        sentences.append((tail.strip(), offset))
    return sentences


def _chunk_prose(text: str) -> list[tuple[str, int, int]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    return _pack_units(
        sentences,
        chunk_tokens=PROSE_CHUNK_TOKENS,
        overlap_ratio=PROSE_OVERLAP_RATIO,
        joiner=" ",
    )


# ---------------------------------------------------------------------------
# H2 anchor tracking (for context prefix)
# ---------------------------------------------------------------------------

_H2_LINE_RE = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)


def _build_h2_index(text: str) -> list[tuple[int, str]]:
    """Return a list of ``(offset, h2_title)`` tuples in document order."""
    return [(m.start(), m.group(1).strip()) for m in _H2_LINE_RE.finditer(text)]


def _nearest_h2(h2_index: list[tuple[int, str]], end: int) -> str:
    """Return the most recent h2 heading at or before ``end``.

    A chunk's span ends at ``end``; any h2 inside or before that point
    contributes to the chunk's section context.
    """
    nearest = ""
    for pos, title in h2_index:
        if pos <= end:
            nearest = title
        else:
            break
    return nearest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_prefix(title: str, h2: str, body: str) -> str:
    title = title.strip()
    h2 = h2.strip()
    if title and h2:
        return f"{title} > {h2}: {body}"
    if title:
        return f"{title}: {body}"
    if h2:
        return f"{h2}: {body}"
    return body


def chunk(text: str, content_type: str, title: str = "") -> list[Chunk]:
    """Split ``text`` into context-prefixed chunks.

    ``content_type`` selects the splitting strategy. Unknown values fall
    back to the prose strategy.
    """
    if not text or not text.strip():
        return []

    if content_type == "code":
        bodies = _chunk_code(text)
    elif content_type == "api-docs":
        bodies = _chunk_api_docs(text)
    else:
        # "prose", "web", or any unknown type
        bodies = _chunk_prose(text)

    if not bodies:
        return []

    h2_index = _build_h2_index(text)
    chunks: list[Chunk] = []
    for i, (body, start, end) in enumerate(bodies):
        h2 = _nearest_h2(h2_index, end)
        prefixed = _format_prefix(title, h2, body)
        meta: dict = {"offset": start}
        if h2:
            meta["h2"] = h2
        chunks.append(Chunk(content=prefixed, chunk_index=i, metadata=meta))
    return chunks
