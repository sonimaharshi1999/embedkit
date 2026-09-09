# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for chunking strategies."""

from __future__ import annotations

import pytest

from embedkit.chunkers import (
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self, sample_documents: list[str]) -> None:
        """Fixed-size chunker should produce chunks within the size limit."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(sample_documents[0])

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.text) <= 100
            assert chunk.index >= 0

    def test_overlap(self) -> None:
        """Consecutive chunks should overlap by the specified amount."""
        text = "A" * 200
        chunker = FixedSizeChunker(chunk_size=100, overlap=30)
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        # Second chunk should start at position 70 (100 - 30)
        assert chunks[1].start_char == 70

    def test_empty_text(self) -> None:
        """Empty text should produce no chunks."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_invalid_params(self) -> None:
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=-1)

    def test_metadata_passthrough(self) -> None:
        """Metadata should be attached to each chunk."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=0)
        meta = {"source": "test.txt"}
        chunks = chunker.chunk("Hello world, this is a test document with some content.", metadata=meta)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata == meta


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_sentence_boundaries(self) -> None:
        """Chunks should respect sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = SentenceChunker(max_chunk_size=50, min_chunk_size=10)
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        # Each chunk text should end with a period (complete sentence)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")

    def test_long_document(self, long_document: str) -> None:
        """SentenceChunker should handle long documents."""
        chunker = SentenceChunker(max_chunk_size=200)
        chunks = chunker.chunk(long_document)
        assert len(chunks) > 1

    def test_single_sentence(self) -> None:
        """A single short sentence should produce one chunk."""
        chunker = SentenceChunker(max_chunk_size=1000)
        chunks = chunker.chunk("Just one sentence.")
        assert len(chunks) == 1


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_splitting(self, long_document: str) -> None:
        """Recursive chunker should produce chunks within size limit."""
        chunker = RecursiveChunker(chunk_size=200, overlap=0)
        chunks = chunker.chunk(long_document)

        assert len(chunks) > 1
        for chunk in chunks:
            # Allow small overflow since splitting on boundaries is approximate
            assert len(chunk.text) <= 250  # small tolerance

    def test_respects_paragraph_breaks(self) -> None:
        """Recursive chunker should prefer splitting on paragraph breaks."""
        text = "Para one content here.\n\nPara two content here.\n\nPara three content here."
        chunker = RecursiveChunker(chunk_size=30, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_many(self, sample_documents: list[str]) -> None:
        """chunk_many should process multiple texts."""
        chunker = RecursiveChunker(chunk_size=150, overlap=0)
        results = chunker.chunk_many(sample_documents[:3])
        assert len(results) == 3
        for chunk_list in results:
            assert len(chunk_list) >= 1


class TestChunkingStrategy:
    """Tests for the ChunkingStrategy enum."""

    def test_enum_values(self) -> None:
        """Strategy enum should have the expected values."""
        assert ChunkingStrategy.FIXED.value == "fixed"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.RECURSIVE.value == "recursive"

    def test_from_string(self) -> None:
        """Strategy should be constructable from string."""
        assert ChunkingStrategy("fixed") == ChunkingStrategy.FIXED
        assert ChunkingStrategy("recursive") == ChunkingStrategy.RECURSIVE
