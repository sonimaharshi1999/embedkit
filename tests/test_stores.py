# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for vector store backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from embedkit.stores.faiss_store import FAISSStore
from embedkit.stores.memory_store import InMemoryStore


DIM = 384


def _make_data(n: int = 5) -> tuple[list[str], NDArray[np.float32]]:
    """Create synthetic texts and random normalized embeddings."""
    texts = [f"Document number {i} about topic {chr(65 + i)}" for i in range(n)]
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return texts, emb


class TestFAISSStore:
    """Tests for the FAISS vector store."""

    def test_add_and_len(self) -> None:
        """Adding vectors should increase store length."""
        store = FAISSStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)
        assert len(store) == 5

    def test_search_returns_results(self) -> None:
        """Search should return scored results."""
        store = FAISSStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)

        results = store.search(emb[0], top_k=3)
        assert len(results) == 3
        # First result should be the query itself (highest similarity)
        assert results[0].text == texts[0]
        assert results[0].score > 0.9

    def test_search_empty_store(self) -> None:
        """Searching an empty store should return empty list."""
        store = FAISSStore(dimension=DIM)
        query = np.random.randn(DIM).astype(np.float32)
        assert store.search(query, top_k=5) == []

    def test_clear(self) -> None:
        """clear() should reset the store."""
        store = FAISSStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)
        store.clear()
        assert len(store) == 0

    def test_save_and_load(self, tmp_dir: Path) -> None:
        """Store should be persistable and loadable."""
        store = FAISSStore(dimension=DIM)
        texts, emb = _make_data(5)
        meta = [{"idx": i} for i in range(5)]
        store.add(texts, emb, metadata_list=meta)

        path = str(tmp_dir / "test_index")
        store.save(path)

        loaded = FAISSStore(dimension=DIM)
        loaded.load(path)
        assert len(loaded) == 5
        results = loaded.search(emb[0], top_k=1)
        assert results[0].text == texts[0]

    def test_dimension_mismatch(self) -> None:
        """Wrong dimension should raise ValueError."""
        store = FAISSStore(dimension=DIM)
        bad_emb = np.random.randn(5, DIM + 1).astype(np.float32)
        with pytest.raises(ValueError):
            store.add(["a", "b", "c", "d", "e"], bad_emb)


class TestInMemoryStore:
    """Tests for the in-memory vector store."""

    def test_add_and_len(self) -> None:
        """Adding vectors should increase store length."""
        store = InMemoryStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)
        assert len(store) == 5

    def test_cosine_similarity_search(self) -> None:
        """Search should rank the query vector's own document first."""
        store = InMemoryStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)

        results = store.search(emb[2], top_k=3)
        assert len(results) == 3
        assert results[0].text == texts[2]
        assert results[0].score > 0.9

    def test_metadata(self) -> None:
        """Metadata should be stored and returned with search results."""
        store = InMemoryStore(dimension=DIM)
        texts, emb = _make_data(3)
        meta = [{"source": f"file_{i}.txt"} for i in range(3)]
        store.add(texts, emb, metadata_list=meta)

        results = store.search(emb[0], top_k=1)
        assert results[0].metadata == {"source": "file_0.txt"}

    def test_save_and_load(self, tmp_dir: Path) -> None:
        """InMemoryStore should persist and reload correctly."""
        store = InMemoryStore(dimension=DIM)
        texts, emb = _make_data(5)
        store.add(texts, emb)

        path = str(tmp_dir / "mem_index")
        store.save(path)

        loaded = InMemoryStore(dimension=DIM)
        loaded.load(path)
        assert len(loaded) == 5

    def test_clear(self) -> None:
        """clear() should reset the store."""
        store = InMemoryStore(dimension=DIM)
        texts, emb = _make_data(3)
        store.add(texts, emb)
        store.clear()
        assert len(store) == 0
