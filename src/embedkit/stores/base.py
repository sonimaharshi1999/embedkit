# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Abstract vector store interface.

Defines the contract that all vector store backends must implement,
plus the SearchResult data class returned by similarity queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SearchResult:
    """A single search result from a vector store query.

    Attributes:
        text: The original text of the matched chunk.
        score: Similarity score (higher is more similar for cosine/IP).
        index: Position in the store's internal array.
        metadata: Arbitrary metadata associated with the stored chunk.
    """

    text: str
    score: float
    index: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32],
        metadata_list: list[dict[str, str | int | float]] | None = None,
    ) -> None:
        """Add texts and their embeddings to the store.

        Args:
            texts: The original text chunks.
            embeddings: A 2-D array of shape (n, dim).
            metadata_list: Optional per-text metadata.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find the top-k most similar items.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            A list of SearchResult objects, sorted by descending score.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored items."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the store."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk.

        Args:
            path: File or directory path to write to.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from disk.

        Args:
            path: File or directory path to read from.
        """
        ...
