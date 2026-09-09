# MIT License
# Copyright (c) 2024 Maharshi Soni

"""Tests for the unified EmbedPipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from embedkit.pipeline import EmbedPipeline


def _model_cached() -> bool:
    """Check if the all-MiniLM-L6-v2 model weights are cached locally."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hf_cache / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not model_dir.exists():
        return False
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
class TestEmbedPipeline:
    """Integration tests for the full pipeline."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.pipeline = EmbedPipeline(
            chunking_strategy="recursive",
            chunk_size=256,
            chunk_overlap=32,
            model_name="all-MiniLM-L6-v2",
            store_type="memory",
            cache_dir=str(tmp_path / "cache"),
        )
        docs = [
            "Machine learning is transforming healthcare with predictive analytics and medical imaging.",
            "Python is a popular programming language for web development and data science applications.",
            "Vector databases like FAISS enable fast similarity search for recommendation systems.",
        ]
        self.pipeline.ingest(docs)

    def test_ingest_creates_chunks(self) -> None:
        """Ingestion should create chunks in the store."""
        assert len(self.pipeline.store) > 0

    def test_vector_search(self) -> None:
        """Vector search should return relevant results."""
        results = self.pipeline.search("healthcare AI", top_k=2, mode="vector")
        assert len(results) > 0
        # The healthcare document should score highest
        assert "healthcare" in results[0].text.lower() or "machine" in results[0].text.lower()

    def test_hybrid_search(self) -> None:
        """Hybrid search should return results."""
        results = self.pipeline.search("Python programming", top_k=2, mode="hybrid")
        assert len(results) > 0

    def test_stats(self) -> None:
        """Pipeline stats should reflect the ingested data."""
        s = self.pipeline.stats()
        assert s.total_documents == 3
        assert s.total_chunks > 0
        assert s.embedding_dimension == 384
        assert s.store_type == "memory"

    def test_save_and_load(self) -> None:
        """Pipeline should be saveable and loadable."""
        with tempfile.TemporaryDirectory(prefix="embedkit_save_") as save_cache_dir:
            with tempfile.TemporaryDirectory(prefix="embedkit_out_") as out_dir:
                pipe = EmbedPipeline(
                    chunking_strategy="fixed",
                    chunk_size=200,
                    chunk_overlap=20,
                    model_name="all-MiniLM-L6-v2",
                    store_type="memory",
                    cache_dir=save_cache_dir,
                )
                pipe.ingest(["Test document for save and load."])
                path = str(Path(out_dir) / "test_store")
                pipe.save(path)

                # Load into a new pipeline with a fresh store
                pipe2 = EmbedPipeline(
                    model_name="all-MiniLM-L6-v2",
                    store_type="memory",
                    cache_dir=None,
                )
                pipe2.load(path)
                assert len(pipe2.store) > 0
