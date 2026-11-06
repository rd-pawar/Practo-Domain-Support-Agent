"""
chunkers.py - Two project-owned chunking strategies.

Strategy 1: Fixed-size overlap chunking
Strategy 2: Sentence-boundary chunking

Both preserve parent_doc_id and chunking_strategy in chunk metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Chunk:
    """A single text chunk with metadata linking it to its parent document."""
    chunk_id: str
    text: str
    parent_doc_id: str
    chunking_strategy: Literal["fixed", "sentence"]
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size overlap chunking
# ---------------------------------------------------------------------------
FIXED_CHUNK_SIZE = 200       # characters
FIXED_CHUNK_OVERLAP = 40     # characters


def fixed_size_chunks(
    text: str,
    doc_id: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split *text* into fixed-size chunks of *chunk_size* characters with
    *overlap* characters of context carried over from adjacent chunks.

    Parameters
    ----------
    text        : Full document content string.
    doc_id      : Stable parent document ID (e.g. "KB-001").
    chunk_size  : Maximum characters per chunk (default 200).
    overlap     : Characters of overlap between consecutive chunks (default 40).

    Returns
    -------
    List of Chunk objects, each tagged with strategy="fixed".
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    text = text.strip()
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_fixed_{idx:04d}",
                    text=chunk_text,
                    parent_doc_id=doc_id,
                    chunking_strategy="fixed",
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1

        # Advance by (chunk_size - overlap) so next chunk shares overlap chars
        next_start = start + chunk_size - overlap
        if next_start <= start:           # safety: avoid infinite loop
            next_start = start + 1
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-boundary chunking
# ---------------------------------------------------------------------------
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
MAX_SENTENCE_LEN = 400       # chars; long sentences get split at this boundary


def _split_long_sentence(sentence: str, max_len: int = MAX_SENTENCE_LEN) -> list[str]:
    """Fallback: split a sentence that exceeds max_len at word boundaries."""
    if len(sentence) <= max_len:
        return [sentence]
    parts: list[str] = []
    words = sentence.split()
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > max_len and current:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def sentence_chunks(
    text: str,
    doc_id: str,
    max_sentence_len: int = MAX_SENTENCE_LEN,
) -> list[Chunk]:
    """
    Split *text* at sentence boundaries (`.`, `!`, `?` followed by whitespace).
    Sentences longer than *max_sentence_len* are further split at word
    boundaries so no chunk exceeds a reasonable embedding length.

    Parameters
    ----------
    text             : Full document content string.
    doc_id           : Stable parent document ID (e.g. "KB-001").
    max_sentence_len : Characters before forcing a word-boundary split (default 400).

    Returns
    -------
    List of Chunk objects, each tagged with strategy="sentence".
    """
    if not text.strip():
        return []

    raw_sentences = SENTENCE_PATTERN.split(text.strip())
    # Flatten, handling long-sentence fallback
    all_sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        all_sentences.extend(_split_long_sentence(s, max_sentence_len))

    chunks: list[Chunk] = []
    char_pos = 0
    for idx, sentence in enumerate(all_sentences):
        start = text.find(sentence, char_pos)
        end = start + len(sentence) if start != -1 else char_pos + len(sentence)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_sentence_{idx:04d}",
                text=sentence,
                parent_doc_id=doc_id,
                chunking_strategy="sentence",
                char_start=max(start, 0),
                char_end=end,
            )
        )
        if start != -1:
            char_pos = start + len(sentence)

    return chunks


# ---------------------------------------------------------------------------
# Convenience: chunk a full KB document dict
# ---------------------------------------------------------------------------
def chunk_document(doc: dict, strategy: Literal["fixed", "sentence"] = "fixed") -> list[Chunk]:
    """Chunk a KB document dict using the specified strategy."""
    doc_id = doc["id"]
    content = doc["content"]
    if strategy == "fixed":
        return fixed_size_chunks(content, doc_id)
    return sentence_chunks(content, doc_id)
