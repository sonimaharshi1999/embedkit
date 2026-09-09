# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
FAISS-backed vector store.

Uses Facebook AI Similarity Search for fast approximate nearest-neighbor
retrieval. Supports inner-product (cosine after normalization) similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from embedkit.stores.base import SearchResult, VectorStore


class FAISSStore(VectorStore):
    """Vector store backed by FAISS IndexFlatIP (inner product).

    Vectors are L2-normalized before insertion so inner-product
    scores approximate cosine similarity.

    Args:
        dimension: Dimensionality of the embedding vectors.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._texts: list[str] = []
        self._metadata: list[dict[str, str | int | float]] = []

    def add(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32],
        metadata_list: list[dict[str, str | int | float]] | None = None,
    ) -> None:
        """Add normalized vectors to the FAISS index.

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

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (embeddings / norms).astype(np.float32)

        self._index.add(normalized)
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
        """Search for nearest neighbors using cosine similarity.

        Args:
            query_embedding: The query vector (1-D).
            top_k: Number of results to return.

        Returns:
            A list of SearchResult objects sorted by descending similarity.
        """
        if len(self) == 0:
            return []

        top_k = min(top_k, len(self))

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores, indices = self._index.search(query, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=float(score),
                    index=int(idx),
                    metadata=self._metadata[idx],
                )
            )
        return results

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal

    def clear(self) -> None:
        """Reset the index and all stored data."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._texts.clear()
        self._metadata.clear()

    def save(self, path: str) -> None:
        """Persist the FAISS index and metadata to disk.

        Creates two files: <path>.faiss and <path>.meta.json

        Args:
            path: Base file path (without extension).
        """
        faiss.write_index(self._index, f"{path}.faiss")
        meta = {
            "dimension": self.dimension,
            "texts": self._texts,
            "metadata": self._metadata,
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load a FAISS index and metadata from disk.

        Args:
            path: Base file path (without extension).
        """
        faiss_path = Path(f"{path}.faiss")
        meta_path = Path(f"{path}.meta.json")

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Store files not found at {path}")

        self._index = faiss.read_index(str(faiss_path))

        with open(str(meta_path), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dimension = meta["dimension"]
        self._texts = meta["texts"]
        self._metadata = meta["metadata"]
