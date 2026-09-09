# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Chunking strategies for splitting documents into embeddable segments.

Provides three strategies:
- FixedSizeChunker: splits text into chunks of a fixed token/character count
- SentenceChunker: splits on sentence boundaries
- RecursiveChunker: hierarchical splitting using multiple separators
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the character length of this chunk."""
        return len(self.text)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, str | int | float] | None = None) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: The input text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            A list of Chunk objects.
        """
        ...

    def chunk_many(
        self,
        texts: Sequence[str],
        metadata_list: Sequence[dict[str, str | int | float]] | None = None,
    ) -> list[list[Chunk]]:
        """Chunk multiple texts.

        Args:
            texts: Sequence of texts to chunk.
            metadata_list: Optional per-text metadata.

        Returns:
            A list of chunk lists, one per input text.
        """
        results: list[list[Chunk]] = []
        for i, text in enumerate(texts):
            meta = metadata_list[i] if metadata_list else None
            results.append(self.chunk(text, metadata=meta))
        return results


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-size character chunks with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, str | int | float] | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks.

        Args:
            text: The input text.
            metadata: Optional metadata for each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        idx = 0
        pos = 0

        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=pos,
                        end_char=end,
                        metadata=metadata or {},
                    )
                )
                idx += 1

            if end >= len(text):
                break
            pos += step

        return chunks


class SentenceChunker(BaseChunker):
    """Splits text on sentence boundaries, grouping sentences up to a max size.

    Args:
        max_chunk_size: Maximum character count per chunk.
        min_chunk_size: Minimum character count; shorter chunks merge with neighbors.
    """

    # Regex to split on sentence-ending punctuation followed by whitespace
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 1024, min_chunk_size: int = 100) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: dict[str, str | int | float] | None = None) -> list[Chunk]:
        """Split text into sentence-boundary-aware chunks.

        Args:
            text: The input text.
            metadata: Optional metadata for each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        sentences = self._SENTENCE_RE.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0
        char_offset = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len + 1 > self.max_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                start = text.find(current_sentences[0], char_offset)
                if start == -1:
                    start = char_offset
                end = start + len(chunk_text)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=end,
                        metadata=metadata or {},
                    )
                )
                idx += 1
                char_offset = end
                current_sentences = []
                current_len = 0

            current_sentences.append(sentence)
            current_len += sentence_len + 1

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = text.find(current_sentences[0], char_offset)
            if start == -1:
                start = char_offset
            end = start + len(chunk_text)

            # Merge with previous chunk if too small
            if len(chunk_text) < self.min_chunk_size and chunks:
                prev = chunks[-1]
                merged_text = prev.text + " " + chunk_text
                chunks[-1] = Chunk(
                    text=merged_text,
                    index=prev.index,
                    start_char=prev.start_char,
                    end_char=end,
                    metadata=metadata or {},
                )
            else:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=end,
                        metadata=metadata or {},
                    )
                )

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively splits text using a hierarchy of separators.

    Tries the first separator; if any resulting piece exceeds the max size,
    splits that piece with the next separator, and so on.

    Args:
        chunk_size: Maximum character count per chunk.
        overlap: Overlap between chunks at the leaf level.
        separators: Ordered list of separators from coarsest to finest.
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text fragments each within chunk_size.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Fallback: hard split
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            pieces = list(text)
        else:
            pieces = text.split(sep)

        result: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece) + (len(sep) if current else 0)

            if current_len + piece_len > self.chunk_size and current:
                merged = sep.join(current)
                if len(merged) > self.chunk_size:
                    result.extend(self._split_text(merged, remaining_seps))
                else:
                    result.append(merged)
                current = []
                current_len = 0

            current.append(piece)
            current_len += piece_len

        if current:
            merged = sep.join(current)
            if len(merged) > self.chunk_size:
                result.extend(self._split_text(merged, remaining_seps))
            else:
                result.append(merged)

        return [r for r in result if r.strip()]

    def chunk(self, text: str, metadata: dict[str, str | int | float] | None = None) -> list[Chunk]:
        """Split text recursively using separator hierarchy.

        Args:
            text: The input text.
            metadata: Optional metadata for each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        fragments = self._split_text(text, self.separators)
        chunks: list[Chunk] = []
        search_start = 0

        for idx, fragment in enumerate(fragments):
            start = text.find(fragment, search_start)
            if start == -1:
                start = search_start
            end = start + len(fragment)
            search_start = start + 1

            chunks.append(
                Chunk(
                    text=fragment,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata=metadata or {},
                )
            )

        return chunks
