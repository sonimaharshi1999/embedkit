# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Shared fixtures and synthetic test data for EmbedKit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Synthetic documents for testing (no external downloads required)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS: list[str] = [
    (
        "Machine learning is a subset of artificial intelligence that focuses "
        "on building systems that learn from data. Instead of being explicitly "
        "programmed, these systems improve their performance through experience. "
        "Common approaches include supervised learning, unsupervised learning, "
        "and reinforcement learning."
    ),
    (
        "Natural language processing enables computers to understand and generate "
        "human language. Key tasks include text classification, named entity "
        "recognition, sentiment analysis, and machine translation. Modern NLP "
        "relies heavily on transformer architectures and large language models."
    ),
    (
        "Vector databases store high-dimensional vectors and support efficient "
        "similarity search operations. They are essential components of retrieval "
        "augmented generation systems. Popular implementations include FAISS, "
        "Pinecone, Weaviate, and Milvus."
    ),
    (
        "Python is a versatile programming language widely used in data science, "
        "web development, and automation. Its rich ecosystem of libraries like "
        "NumPy, pandas, and scikit-learn makes it the language of choice for "
        "many machine learning practitioners."
    ),
    (
        "The attention mechanism allows neural networks to focus on relevant parts "
        "of the input when producing output. Transformers use self-attention to "
        "process sequences in parallel, achieving significant speedups over "
        "recurrent architectures while maintaining or improving accuracy."
    ),
]

SHORT_DOCUMENT: str = "Hello world. This is a test."

LONG_DOCUMENT: str = " ".join(
    [
        f"Paragraph {i}: This is a detailed discussion about topic {i}. "
        f"It contains multiple sentences with varying content. "
        f"The information here is relevant to section {i} of the document. "
        f"We explore the implications and provide analysis."
        for i in range(50)
    ]
)


@pytest.fixture
def sample_documents() -> list[str]:
    """Return a list of synthetic test documents."""
    return SAMPLE_DOCUMENTS.copy()


@pytest.fixture
def short_document() -> str:
    """Return a short test document."""
    return SHORT_DOCUMENT


@pytest.fixture
def long_document() -> str:
    """Return a long test document with many paragraphs."""
    return LONG_DOCUMENT


@pytest.fixture
def random_embeddings() -> NDArray[np.float32]:
    """Return random embeddings with shape (5, 384) matching all-MiniLM-L6-v2 dim."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 384)).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / norms).astype(np.float32)


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
