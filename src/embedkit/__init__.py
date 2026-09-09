# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
EmbedKit - Production Embedding Pipeline Toolkit.

A pip-installable toolkit for building embedding pipelines with multiple
chunking strategies, cached embedding generation, and a unified vector
store interface supporting FAISS and in-memory backends.
"""

from embedkit.__version__ import __version__
from embedkit.chunkers import (
    ChunkingStrategy,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
)
from embedkit.embedder import Embedder
from embedkit.cache import EmbeddingCache
from embedkit.stores.base import VectorStore, SearchResult
from embedkit.stores.faiss_store import FAISSStore
from embedkit.stores.memory_store import InMemoryStore
from embedkit.hybrid import HybridSearcher
from embedkit.pipeline import EmbedPipeline

__all__: list[str] = [
    "__version__",
    # Chunkers
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    # Embedder
    "Embedder",
    # Cache
    "EmbeddingCache",
    # Stores
    "VectorStore",
    "SearchResult",
    "FAISSStore",
    "InMemoryStore",
    # Hybrid Search
    "HybridSearcher",
    # Pipeline
    "EmbedPipeline",
]
