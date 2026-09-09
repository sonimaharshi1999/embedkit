# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Unified embedding pipeline.

Orchestrates the full flow: text -> chunking -> embedding -> storage -> search.
Provides a single high-level API that wires together all components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from embedkit.chunkers import (
    BaseChunker,
    Chunk,
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from embedkit.embedder import Embedder
from embedkit.hybrid import HybridSearcher
from embedkit.stores.base import SearchResult, VectorStore
from embedkit.stores.faiss_store import FAISSStore
from embedkit.stores.memory_store import InMemoryStore


@dataclass
class PipelineStats:
    """Statistics about the pipeline's indexed data.

    Attributes:
        total_documents: Number of documents ingested.
        total_chunks: Number of chunks created.
        embedding_dimension: Dimensionality of embeddings.
        store_type: Name of the vector store backend.
        cache_size: Number of cached embeddings.
        cache_disk_bytes: Disk usage of the embedding cache.
    """

    total_documents: int
    total_chunks: int
    embedding_dimension: int
    store_type: str
    cache_size: int
    cache_disk_bytes: int


class EmbedPipeline:
    """End-to-end embedding pipeline.

    Wires together chunking, embedding, and vector storage into a single
    coherent API. Supports hybrid search (vector + BM25) out of the box.

    Args:
        chunking_strategy: Which chunking approach to use.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        model_name: The sentence-transformers model to use.
        store_type: Vector store backend ("faiss" or "memory").
        cache_dir: Embedding cache directory. None disables caching.
        hybrid_alpha: Weight for vector search in hybrid mode (0-1).
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model_name: str = "all-MiniLM-L6-v2",
        store_type: Literal["faiss", "memory"] = "faiss",
        cache_dir: str | Path | None = ".embedkit_cache",
        hybrid_alpha: float = 0.7,
    ) -> None:
        # Resolve strategy
        if isinstance(chunking_strategy, str):
            chunking_strategy = ChunkingStrategy(chunking_strategy)

        self._chunker: BaseChunker = self._make_chunker(
            chunking_strategy, chunk_size, chunk_overlap
        )
        self._embedder = Embedder(
            model_name=model_name,
            cache_dir=cache_dir,
            show_progress=False,
        )

        # Initialize store
        dim = self._embedder.dimension
        self._store: VectorStore
        if store_type == "faiss":
            self._store = FAISSStore(dimension=dim)
        else:
            self._store = InMemoryStore(dimension=dim)

        self._store_type = store_type
        self._hybrid = HybridSearcher(
            vector_store=self._store,
            alpha=hybrid_alpha,
        )
        self._all_chunks: list[Chunk] = []
        self._doc_count: int = 0
        self._hybrid_indexed: bool = False

    @staticmethod
    def _make_chunker(
        strategy: ChunkingStrategy,
        chunk_size: int,
        overlap: int,
    ) -> BaseChunker:
        """Create a chunker instance for the given strategy.

        Args:
            strategy: The chunking strategy enum.
            chunk_size: Max chunk size.
            overlap: Overlap between chunks.

        Returns:
            A BaseChunker subclass instance.
        """
        if strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(max_chunk_size=chunk_size)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def embedder(self) -> Embedder:
        """Return the underlying Embedder instance."""
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """Return the underlying VectorStore instance."""
        return self._store

    def ingest(
        self,
        documents: list[str],
        metadata_list: list[dict[str, str | int | float]] | None = None,
    ) -> int:
        """Ingest documents through the full pipeline.

        Chunks each document, generates embeddings, and adds them to the
        vector store.

        Args:
            documents: List of document texts.
            metadata_list: Optional per-document metadata.

        Returns:
            The number of chunks created.
        """
        all_texts: list[str] = []
        all_metadata: list[dict[str, str | int | float]] = []

        for i, doc in enumerate(documents):
            meta = metadata_list[i] if metadata_list else {}
            chunks = self._chunker.chunk(doc, metadata=meta)
            self._all_chunks.extend(chunks)

            for chunk in chunks:
                all_texts.append(chunk.text)
                all_metadata.append(chunk.metadata)

        if not all_texts:
            return 0

        # Embed in batch
        embeddings = self._embedder.embed_batch(all_texts)

        # Add to vector store (not through hybrid -- store directly)
        self._store.add(all_texts, embeddings, all_metadata)

        # Rebuild hybrid BM25 index with all texts
        self._rebuild_hybrid_index()

        self._doc_count += len(documents)
        return len(all_texts)

    def _rebuild_hybrid_index(self) -> None:
        """Rebuild the BM25 component of hybrid search."""
        # The vector store already has the vectors; just re-index BM25
        # over all chunk texts accumulated so far.
        all_texts = [c.text for c in self._all_chunks]
        self._hybrid._bm25.index(all_texts)
        self._hybrid._texts = all_texts
        self._hybrid_indexed = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["vector", "hybrid"] = "hybrid",
    ) -> list[SearchResult]:
        """Search the indexed documents.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            mode: "vector" for pure vector search, "hybrid" for vector + BM25.

        Returns:
            A list of SearchResult objects.
        """
        query_embedding = self._embedder.embed(query)

        if mode == "hybrid" and self._hybrid_indexed:
            return self._hybrid.search(query, query_embedding, top_k=top_k)
        else:
            return self._store.search(query_embedding, top_k=top_k)

    def stats(self) -> PipelineStats:
        """Return statistics about the current pipeline state.

        Returns:
            A PipelineStats object.
        """
        cache = self._embedder.cache
        return PipelineStats(
            total_documents=self._doc_count,
            total_chunks=len(self._all_chunks),
            embedding_dimension=self._embedder.dimension,
            store_type=self._store_type,
            cache_size=cache.size if cache else 0,
            cache_disk_bytes=cache.disk_usage_bytes() if cache else 0,
        )

    def save(self, path: str) -> None:
        """Persist the pipeline's vector store to disk.

        Args:
            path: Base path for saved files.
        """
        self._store.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved vector store.

        Args:
            path: Base path of saved files.
        """
        self._store.load(path)
