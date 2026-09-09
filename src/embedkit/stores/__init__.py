# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Vector store backends for EmbedKit."""

from embedkit.stores.base import VectorStore, SearchResult
from embedkit.stores.faiss_store import FAISSStore
from embedkit.stores.memory_store import InMemoryStore

__all__: list[str] = ["VectorStore", "SearchResult", "FAISSStore", "InMemoryStore"]
