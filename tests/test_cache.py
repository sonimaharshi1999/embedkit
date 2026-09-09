# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for the embedding cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embedkit.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_put_and_get(self, tmp_dir: Path) -> None:
        """Stored embeddings should be retrievable."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        embedding = np.random.randn(384).astype(np.float32)

        cache.put("hello world", "test-model", embedding)
        result = cache.get("hello world", "test-model")

        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding)

    def test_cache_miss(self, tmp_dir: Path) -> None:
        """Non-existent keys should return None."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        assert cache.get("nonexistent", "model") is None

    def test_contains(self, tmp_dir: Path) -> None:
        """contains() should reflect cached state."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        embedding = np.random.randn(384).astype(np.float32)

        assert not cache.contains("test", "model")
        cache.put("test", "model", embedding)
        assert cache.contains("test", "model")

    def test_clear(self, tmp_dir: Path) -> None:
        """clear() should remove all entries."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        embedding = np.random.randn(384).astype(np.float32)

        cache.put("text1", "model", embedding)
        cache.put("text2", "model", embedding)
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0
        assert cache.get("text1", "model") is None

    def test_different_models(self, tmp_dir: Path) -> None:
        """Same text with different models should have separate cache entries."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.zeros(384, dtype=np.float32)

        cache.put("same text", "model-a", emb1)
        cache.put("same text", "model-b", emb2)

        r1 = cache.get("same text", "model-a")
        r2 = cache.get("same text", "model-b")

        assert r1 is not None and r2 is not None
        np.testing.assert_array_almost_equal(r1, emb1)
        np.testing.assert_array_almost_equal(r2, emb2)

    def test_disk_usage(self, tmp_dir: Path) -> None:
        """disk_usage_bytes should be positive after caching."""
        cache = EmbeddingCache(cache_dir=tmp_dir / "cache")
        embedding = np.random.randn(384).astype(np.float32)
        cache.put("test", "model", embedding)
        assert cache.disk_usage_bytes() > 0

    def test_persistence(self, tmp_dir: Path) -> None:
        """Cache should survive re-instantiation from the same directory."""
        cache_dir = tmp_dir / "cache"
        embedding = np.random.randn(384).astype(np.float32)

        cache1 = EmbeddingCache(cache_dir=cache_dir)
        cache1.put("persist_test", "model", embedding)

        # Create a new cache instance pointing to the same directory
        cache2 = EmbeddingCache(cache_dir=cache_dir)
        result = cache2.get("persist_test", "model")
        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding)
