# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for BM25 and hybrid search."""

from __future__ import annotations

import numpy as np
import pytest

from embedkit.hybrid import BM25Index, HybridSearcher
from embedkit.stores.memory_store import InMemoryStore


DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning algorithms process large datasets efficiently.",
    "Python programming language is popular for data science.",
    "Vector databases enable fast similarity search operations.",
    "Natural language processing transforms how we interact with computers.",
]

DIM = 384


class TestBM25Index:
    """Tests for the BM25 keyword index."""

    def test_basic_search(self) -> None:
        """BM25 should rank documents containing query terms higher."""
        bm25 = BM25Index()
        bm25.index(DOCUMENTS)

        results = bm25.search("machine learning algorithms", top_k=3)
        assert len(results) > 0
        # The ML document should rank first
        top_idx = results[0][0]
        assert top_idx == 1  # "Machine learning algorithms..."

    def test_no_match(self) -> None:
        """Query with no matching terms should return empty results."""
        bm25 = BM25Index()
        bm25.index(DOCUMENTS)

        results = bm25.search("xyznonexistent", top_k=5)
        assert len(results) == 0

    def test_scoring(self) -> None:
        """All BM25 scores should be non-negative."""
        bm25 = BM25Index()
        bm25.index(DOCUMENTS)

        scores = bm25.score("python programming")
        for s in scores:
            assert s >= 0

    def test_empty_index(self) -> None:
        """Searching an empty index should return empty."""
        bm25 = BM25Index()
        bm25.index([])
        results = bm25.search("test", top_k=5)
        assert results == []


class TestHybridSearcher:
    """Tests for hybrid search (vector + BM25)."""

    def test_hybrid_search_returns_results(self) -> None:
        """Hybrid search should return fused results."""
        store = InMemoryStore(dimension=DIM)
        hybrid = HybridSearcher(vector_store=store, alpha=0.7)

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((len(DOCUMENTS), DIM)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = (emb / norms).astype(np.float32)

        hybrid.index(DOCUMENTS, emb)

        query_emb = emb[1]  # Use the ML document's embedding as query
        results = hybrid.search("machine learning", query_emb, top_k=3)

        assert len(results) == 3
        # The ML document should appear in the results
        result_texts = [r.text for r in results]
        assert DOCUMENTS[1] in result_texts

    def test_alpha_boundaries(self) -> None:
        """Alpha outside [0,1] should raise ValueError."""
        store = InMemoryStore(dimension=DIM)
        with pytest.raises(ValueError):
            HybridSearcher(vector_store=store, alpha=1.5)
        with pytest.raises(ValueError):
            HybridSearcher(vector_store=store, alpha=-0.1)

    def test_pure_vector_alpha(self) -> None:
        """Alpha=1.0 should heavily weight vector results."""
        store = InMemoryStore(dimension=DIM)
        hybrid = HybridSearcher(vector_store=store, alpha=1.0)

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((len(DOCUMENTS), DIM)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = (emb / norms).astype(np.float32)

        hybrid.index(DOCUMENTS, emb)
        results = hybrid.search("anything", emb[0], top_k=1)
        assert len(results) >= 1
