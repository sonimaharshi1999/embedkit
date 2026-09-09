# MIT License
# Copyright (c) 2024 Maharshi Soni

"""
CLI interface for EmbedKit.

Provides commands for indexing documents, searching, and viewing stats
via the ``embedkit`` command-line tool.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from embedkit.__version__ import __version__
from embedkit.chunkers import ChunkingStrategy
from embedkit.pipeline import EmbedPipeline


def _read_documents(source: str) -> list[tuple[str, str]]:
    """Read documents from a file or directory.

    Supports .txt and .json files. For directories, reads all .txt files.

    Args:
        source: Path to a file or directory.

    Returns:
        A list of (filename, content) tuples.
    """
    path = Path(source)
    documents: list[tuple[str, str]] = []

    if path.is_file():
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        documents.append((f"doc_{i}", item))
                    elif isinstance(item, dict) and "text" in item:
                        name = item.get("name", f"doc_{i}")
                        documents.append((str(name), item["text"]))
        else:
            content = path.read_text(encoding="utf-8")
            documents.append((path.name, content))
    elif path.is_dir():
        for txt_file in sorted(path.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            documents.append((txt_file.name, content))
    else:
        click.echo(f"Error: '{source}' is not a valid file or directory.", err=True)
        sys.exit(1)

    return documents


@click.group()
@click.version_option(version=__version__, prog_name="embedkit")
def cli() -> None:
    """EmbedKit - Production Embedding Pipeline Toolkit.

    Build embedding pipelines with multiple chunking strategies,
    cached embeddings, and unified vector store interfaces.
    """
    pass


@cli.command()
@click.argument("source")
@click.option(
    "--strategy",
    type=click.Choice(["fixed", "sentence", "recursive"]),
    default="recursive",
    help="Chunking strategy to use.",
)
@click.option("--chunk-size", type=int, default=512, help="Maximum chunk size in characters.")
@click.option("--overlap", type=int, default=64, help="Overlap between chunks.")
@click.option("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name.")
@click.option(
    "--store",
    type=click.Choice(["faiss", "memory"]),
    default="faiss",
    help="Vector store backend.",
)
@click.option("--output", type=str, default="embedkit_index", help="Output path for the index.")
@click.option("--cache-dir", type=str, default=".embedkit_cache", help="Embedding cache directory.")
def index(
    source: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    model: str,
    store: str,
    output: str,
    cache_dir: str,
) -> None:
    """Index documents from a file or directory.

    SOURCE can be a .txt file, .json file, or a directory of .txt files.
    """
    click.echo(f"EmbedKit v{__version__} - Indexing documents")
    click.echo(f"  Source:   {source}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Model:    {model}")
    click.echo(f"  Store:    {store}")
    click.echo()

    documents = _read_documents(source)
    if not documents:
        click.echo("No documents found.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(documents)} document(s). Indexing...")

    start = time.time()

    pipeline = EmbedPipeline(
        chunking_strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model_name=model,
        store_type=store,  # type: ignore[arg-type]
        cache_dir=cache_dir,
    )

    texts = [content for _name, content in documents]
    metadata = [{"source": name} for name, _content in documents]
    n_chunks = pipeline.ingest(texts, metadata_list=metadata)

    elapsed = time.time() - start

    pipeline.save(output)

    click.echo(f"Indexed {n_chunks} chunks from {len(documents)} documents in {elapsed:.2f}s")
    click.echo(f"Index saved to: {output}")

    stats = pipeline.stats()
    click.echo(f"  Embedding dimension: {stats.embedding_dimension}")
    click.echo(f"  Cache entries: {stats.cache_size}")


@cli.command()
@click.argument("query")
@click.option("--index-path", type=str, default="embedkit_index", help="Path to the saved index.")
@click.option("--top-k", type=int, default=5, help="Number of results to return.")
@click.option("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name.")
@click.option(
    "--store",
    type=click.Choice(["faiss", "memory"]),
    default="faiss",
    help="Vector store backend type.",
)
@click.option(
    "--mode",
    type=click.Choice(["vector", "hybrid"]),
    default="vector",
    help="Search mode.",
)
def search(
    query: str,
    index_path: str,
    top_k: int,
    model: str,
    store: str,
    mode: str,
) -> None:
    """Search indexed documents with a query.

    QUERY is the search string to find similar documents.
    """
    from embedkit.embedder import Embedder
    from embedkit.stores.faiss_store import FAISSStore
    from embedkit.stores.memory_store import InMemoryStore

    click.echo(f"Searching for: '{query}'")
    click.echo(f"  Index: {index_path}")
    click.echo(f"  Mode:  {mode}")
    click.echo()

    embedder = Embedder(model_name=model, cache_dir=None)
    dim = embedder.dimension

    vector_store: FAISSStore | InMemoryStore
    if store == "faiss":
        vector_store = FAISSStore(dimension=dim)
    else:
        vector_store = InMemoryStore(dimension=dim)

    try:
        vector_store.load(index_path)
    except FileNotFoundError:
        click.echo(f"Error: Index not found at '{index_path}'.", err=True)
        click.echo("Run 'embedkit index' first to create an index.", err=True)
        sys.exit(1)

    query_embedding = embedder.embed(query)
    results = vector_store.search(query_embedding, top_k=top_k)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Top {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        click.echo(f"  [{i}] Score: {result.score:.4f}")
        preview = result.text[:200].replace("\n", " ")
        click.echo(f"      {preview}")
        if result.metadata:
            click.echo(f"      Metadata: {result.metadata}")
        click.echo()


@cli.command()
@click.option("--index-path", type=str, default="embedkit_index", help="Path to the saved index.")
@click.option("--cache-dir", type=str, default=".embedkit_cache", help="Embedding cache directory.")
@click.option(
    "--store",
    type=click.Choice(["faiss", "memory"]),
    default="faiss",
    help="Vector store backend type.",
)
def stats(index_path: str, cache_dir: str, store: str) -> None:
    """Show statistics about an indexed dataset."""
    from embedkit.cache import EmbeddingCache
    from embedkit.stores.faiss_store import FAISSStore
    from embedkit.stores.memory_store import InMemoryStore

    click.echo(f"EmbedKit v{__version__} - Index Statistics")
    click.echo()

    # Check cache
    cache_path = Path(cache_dir)
    if cache_path.exists():
        cache = EmbeddingCache(cache_dir)
        click.echo(f"  Cache directory: {cache_dir}")
        click.echo(f"  Cached embeddings: {cache.size}")
        disk_bytes = cache.disk_usage_bytes()
        if disk_bytes > 1024 * 1024:
            click.echo(f"  Cache disk usage: {disk_bytes / (1024 * 1024):.2f} MB")
        else:
            click.echo(f"  Cache disk usage: {disk_bytes / 1024:.2f} KB")
    else:
        click.echo("  No cache directory found.")

    click.echo()

    # Check index
    if store == "faiss":
        faiss_path = Path(f"{index_path}.faiss")
        meta_path = Path(f"{index_path}.meta.json")
    else:
        faiss_path = Path(f"{index_path}.npz")
        meta_path = Path(f"{index_path}.meta.json")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        click.echo(f"  Index path: {index_path}")
        click.echo(f"  Store type: {store}")
        click.echo(f"  Dimension: {meta.get('dimension', 'unknown')}")
        click.echo(f"  Indexed vectors: {len(meta.get('texts', []))}")
    else:
        click.echo(f"  No index found at '{index_path}'.")


if __name__ == "__main__":
    cli()
