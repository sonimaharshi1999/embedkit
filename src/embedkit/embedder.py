# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Embedding generation with optional caching.

Wraps sentence-transformers to produce dense vector representations
of text chunks, with transparent disk caching to avoid recomputation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from embedkit.cache import EmbeddingCache


class Embedder:
    """Generate embeddings using sentence-transformers with optional caching.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Directory for the embedding cache. None disables caching.
        batch_size: Number of texts to encode in each batch.
        show_progress: Whether to show a progress bar during encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = ".embedkit_cache",
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = SentenceTransformer(model_name)
        self._cache: EmbeddingCache | None = None
        if cache_dir is not None:
            self._cache = EmbeddingCache(cache_dir)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimension for model {self.model_name}")
        return int(dim)

    @property
    def cache(self) -> EmbeddingCache | None:
        """Return the underlying cache instance, if any."""
        return self._cache

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string.

        Args:
            text: The input text.

        Returns:
            A 1-D float32 numpy array.
        """
        if self._cache is not None:
            cached = self._cache.get(text, self.model_name)
            if cached is not None:
                return cached

        embedding: NDArray[np.float32] = self._model.encode(
            text,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embedding = embedding.astype(np.float32)

        if self._cache is not None:
            self._cache.put(text, self.model_name, embedding)

        return embedding

    def embed_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed a batch of texts, leveraging cache where possible.

        Args:
            texts: Sequence of input texts.

        Returns:
            A 2-D float32 numpy array of shape (len(texts), dimension).
        """
        results: list[NDArray[np.float32] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            if self._cache is not None:
                cached = self._cache.get(text, self.model_name)
                if cached is not None:
                    results[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Encode uncached texts in batch
        if uncached_texts:
            new_embeddings: NDArray[np.float32] = self._model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
            )
            new_embeddings = new_embeddings.astype(np.float32)

            for j, idx in enumerate(uncached_indices):
                emb = new_embeddings[j]
                results[idx] = emb
                if self._cache is not None:
                    self._cache.put(texts[idx], self.model_name, emb)

        return np.stack([r for r in results if r is not None], axis=0)
