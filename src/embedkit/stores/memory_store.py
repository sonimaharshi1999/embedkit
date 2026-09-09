# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
In-memory vector store using numpy.

A simple, dependency-free vector store that computes cosine similarity
using numpy. Suitable for small datasets or testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from embedkit.stores.base import SearchResult, VectorStore


class InMemoryStore(VectorStore):
    """Pure-numpy in-memory vector store with cosine similarity search.

    All data lives in RAM. Suitable for prototyping, testing, and small
    datasets (up to ~100k vectors).

    Args:
        dimension: Dimensionality of the embedding vectors.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._embeddings: NDArray[np.float32] = np.empty((0, dimension), dtype=np.float32)
        self._texts: list[str] = []
        self._metadata: list[dict[str, str | int | float]] = []

    def add(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32],
        metadata_list: list[dict[str, str | int | float]] | None = None,
    ) -> None:
        """Add vectors to the in-memory store.

        Args:
            texts: The original text chunks.
            embeddings: A 2-D array of shape (n, dim).
            metadata_list: Optional per-text metadata.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts must match number of embeddings")

        self._embeddings = np.vstack([self._embeddings, embeddings.astype(np.float32)])
        self._texts.extend(texts)

        if metadata_list:
            self._metadata.extend(metadata_list)
        else:
            self._metadata.extend([{} for _ in texts])

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Brute-force cosine similarity search.

        Args:
            query_embedding: The query vector (1-D).
            top_k: Number of results to return.

        Returns:
            A list of SearchResult objects sorted by descending similarity.
        """
        if len(self) == 0:
            return []

        top_k = min(top_k, len(self))

        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Cosine similarity
        query_norm = np.linalg.norm(query)
        embed_norms = np.linalg.norm(self._embeddings, axis=1)

        if query_norm == 0:
            return []

        # Avoid division by zero for stored vectors
        safe_norms = np.where(embed_norms == 0, 1, embed_norms)
        similarities = (self._embeddings @ query.T).flatten() / (safe_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=float(similarities[idx]),
                    index=int(idx),
                    metadata=self._metadata[idx],
                )
            )
        return results

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self._texts)

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self._embeddings = np.empty((0, self.dimension), dtype=np.float32)
        self._texts.clear()
        self._metadata.clear()

    def save(self, path: str) -> None:
        """Persist the store to disk as .npz + .meta.json.

        Args:
            path: Base file path (without extension).
        """
        np.savez_compressed(f"{path}.npz", embeddings=self._embeddings)
        meta = {
            "dimension": self.dimension,
            "texts": self._texts,
            "metadata": self._metadata,
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load the store from disk.

        Args:
            path: Base file path (without extension).
        """
        npz_path = Path(f"{path}.npz")
        meta_path = Path(f"{path}.meta.json")

        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Store files not found at {path}")

        data = np.load(str(npz_path))
        self._embeddings = data["embeddings"].astype(np.float32)

        with open(str(meta_path), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dimension = meta["dimension"]
        self._texts = meta["texts"]
        self._metadata = meta["metadata"]
