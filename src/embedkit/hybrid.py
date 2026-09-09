# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
Hybrid search combining vector similarity with BM25 keyword matching.

Provides a unified search interface that fuses dense (embedding-based)
and sparse (BM25) retrieval scores using reciprocal rank fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from embedkit.stores.base import SearchResult, VectorStore


@dataclass
class BM25Index:
    """A lightweight BM25 index for keyword matching.

    Implements the Okapi BM25 scoring formula over a set of documents.

    Attributes:
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.
    """

    k1: float = 1.5
    b: float = 0.75
    _doc_freqs: dict[str, int] = field(default_factory=dict)
    _doc_lens: list[int] = field(default_factory=list)
    _avg_doc_len: float = 0.0
    _doc_term_freqs: list[dict[str, int]] = field(default_factory=list)
    _n_docs: int = 0
    _texts: list[str] = field(default_factory=list)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing.

        Args:
            text: Input text.

        Returns:
            List of lowercase tokens.
        """
        return re.findall(r"\w+", text.lower())

    def index(self, texts: list[str]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            texts: The documents to index.
        """
        self._texts = texts
        self._n_docs = len(texts)
        self._doc_freqs = {}
        self._doc_lens = []
        self._doc_term_freqs = []

        for text in texts:
            tokens = self._tokenize(text)
            self._doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self._doc_term_freqs.append(dict(tf))

            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        total = sum(self._doc_lens)
        self._avg_doc_len = total / self._n_docs if self._n_docs > 0 else 0.0

    def score(self, query: str) -> list[float]:
        """Score all documents against a query using BM25.

        Args:
            query: The search query.

        Returns:
            A list of BM25 scores, one per indexed document.
        """
        query_tokens = self._tokenize(query)
        scores: list[float] = [0.0] * self._n_docs

        for token in query_tokens:
            if token not in self._doc_freqs:
                continue

            df = self._doc_freqs[token]
            # IDF component (with smoothing to avoid negative values)
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

            for i in range(self._n_docs):
                tf = self._doc_term_freqs[i].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self._doc_lens[i]
                # BM25 TF component
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                )
                scores[i] += idf * tf_norm

        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k documents by BM25 score.

        Args:
            query: The search query.
            top_k: Number of results.

        Returns:
            List of (doc_index, score) tuples sorted by descending score.
        """
        scores = self.score(query)
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


class HybridSearcher:
    """Combines vector similarity search with BM25 keyword matching.

    Uses reciprocal rank fusion (RRF) to merge results from both
    retrieval strategies into a single ranked list.

    Args:
        vector_store: The vector store backend.
        alpha: Weight for vector similarity (0-1). BM25 weight = 1 - alpha.
        rrf_k: Reciprocal rank fusion constant. Higher values smooth rankings.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.vector_store = vector_store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._texts: list[str] = []

    def index(self, texts: list[str], embeddings: NDArray[np.float32]) -> None:
        """Index texts for both vector and keyword search.

        Args:
            texts: The document texts.
            embeddings: Corresponding embedding vectors.
        """
        self._texts = texts
        self._bm25.index(texts)
        self.vector_store.add(texts, embeddings)

    def search(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector and BM25 retrieval.

        Uses reciprocal rank fusion to merge the two ranked lists.

        Args:
            query: The text query for BM25.
            query_embedding: The query vector for similarity search.
            top_k: Number of results to return.

        Returns:
            A list of SearchResult objects sorted by fused score.
        """
        # Get more candidates than needed for better fusion
        candidate_k = min(top_k * 3, len(self._texts)) if self._texts else top_k

        # Vector search
        vector_results = self.vector_store.search(query_embedding, top_k=candidate_k)

        # BM25 search
        bm25_results = self._bm25.search(query, top_k=candidate_k)

        # Reciprocal rank fusion
        rrf_scores: dict[int, float] = {}

        for rank, result in enumerate(vector_results):
            idx = result.index
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.alpha / (self.rrf_k + rank + 1)

        for rank, (idx, _score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.alpha) / (self.rrf_k + rank + 1)

        # Sort by fused score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in sorted_indices[:top_k]:
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=score,
                    index=idx,
                    metadata={},
                )
            )

        return results
