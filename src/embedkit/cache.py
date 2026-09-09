# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Local disk cache for embedding vectors.

Stores embeddings keyed by a hash of the input text + model name,
avoiding redundant computation on repeated indexing runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EmbeddingCache:
    """Persistent on-disk cache for embedding vectors.

    Embeddings are stored as .npy files in a directory, keyed by a SHA-256
    hash of (model_name, text). A manifest JSON tracks metadata.

    Args:
        cache_dir: Directory to store cached embeddings.
    """

    def __init__(self, cache_dir: str | Path = ".embedkit_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load the manifest from disk.

        Returns:
            A dict mapping cache keys to metadata entries.
        """
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    @staticmethod
    def _compute_key(text: str, model_name: str) -> str:
        """Compute a cache key from text and model name.

        Args:
            text: The input text.
            model_name: Name of the embedding model.

        Returns:
            A hex digest string used as the cache key.
        """
        content = f"{model_name}::{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> NDArray[np.float32] | None:
        """Retrieve a cached embedding if available.

        Args:
            text: The original text.
            model_name: The model used to generate the embedding.

        Returns:
            The cached embedding array, or None if not found.
        """
        key = self._compute_key(text, model_name)
        if key not in self._manifest:
            return None

        npy_path = self.cache_dir / f"{key}.npy"
        if not npy_path.exists():
            # Stale manifest entry
            del self._manifest[key]
            self._save_manifest()
            return None

        return np.load(str(npy_path)).astype(np.float32)

    def put(self, text: str, model_name: str, embedding: NDArray[np.float32]) -> None:
        """Store an embedding in the cache.

        Args:
            text: The original text.
            model_name: The model used.
            embedding: The embedding vector to cache.
        """
        key = self._compute_key(text, model_name)
        npy_path = self.cache_dir / f"{key}.npy"
        np.save(str(npy_path), embedding)

        self._manifest[key] = {
            "model": model_name,
            "text_length": len(text),
            "dim": embedding.shape[0],
        }
        self._save_manifest()

    def contains(self, text: str, model_name: str) -> bool:
        """Check whether an embedding is cached.

        Args:
            text: The original text.
            model_name: The model name.

        Returns:
            True if a valid cache entry exists.
        """
        return self.get(text, model_name) is not None

    @property
    def size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._manifest)

    def clear(self) -> None:
        """Remove all cached embeddings and reset the manifest."""
        for key in list(self._manifest.keys()):
            npy_path = self.cache_dir / f"{key}.npy"
            if npy_path.exists():
                os.remove(str(npy_path))
        self._manifest.clear()
        self._save_manifest()

    def disk_usage_bytes(self) -> int:
        """Calculate total disk usage of the cache in bytes.

        Returns:
            Total size in bytes.
        """
        total = 0
        for path in self.cache_dir.iterdir():
            if path.is_file():
                total += path.stat().st_size
        return total
