# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for the Embedder (requires sentence-transformers model download)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from embedkit.embedder import Embedder


def _model_cached() -> bool:
    """Check if the all-MiniLM-L6-v2 model weights are cached locally."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hf_cache / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not model_dir.exists():
        return False
    # Check if any snapshot has the actual model weights
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir():
            weights = snap / "model.safetensors"
            alt_weights = snap / "pytorch_model.bin"
            if weights.exists() or alt_weights.exists():
                return True
    return False


requires_model = pytest.mark.skipif(
    not _model_cached(),
    reason="all-MiniLM-L6-v2 model weights not cached locally (HuggingFace CDN may be down)",
)


@requires_model
class TestEmbedder:
    """Tests for embedding generation.

    These tests require the all-MiniLM-L6-v2 model to be available.
    The model is downloaded automatically on first use.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.embedder = Embedder(model_name="all-MiniLM-L6-v2", cache_dir=tmp_path / "cache")

    def test_dimension(self) -> None:
        """Model dimension should be 384 for all-MiniLM-L6-v2."""
        assert self.embedder.dimension == 384

    def test_single_embed(self) -> None:
        """embed() should return a 1-D array of the correct dimension."""
        result = self.embedder.embed("Hello world")
        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_batch_embed(self) -> None:
        """embed_batch() should return a 2-D array."""
        texts = ["First text", "Second text", "Third text"]
        result = self.embedder.embed_batch(texts)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_cache_hit(self) -> None:
        """Repeated embedding of the same text should use cache."""
        text = "Cache test document"
        emb1 = self.embedder.embed(text)
        emb2 = self.embedder.embed(text)
        np.testing.assert_array_equal(emb1, emb2)

    def test_similar_texts_have_high_similarity(self) -> None:
        """Semantically similar texts should have high cosine similarity."""
        emb1 = self.embedder.embed("The cat sat on the mat")
        emb2 = self.embedder.embed("A cat was sitting on a mat")

        cos_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        assert cos_sim > 0.7

    def test_dissimilar_texts_have_low_similarity(self) -> None:
        """Unrelated texts should have lower cosine similarity."""
        emb1 = self.embedder.embed("The stock market crashed yesterday")
        emb2 = self.embedder.embed("I love baking chocolate chip cookies")

        cos_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        assert cos_sim < 0.5

    def test_no_cache(self) -> None:
        """Embedder with cache_dir=None should work without caching."""
        emb = Embedder(model_name="all-MiniLM-L6-v2", cache_dir=None)
        result = emb.embed("No cache test")
        assert result.shape == (384,)
        assert emb.cache is None
