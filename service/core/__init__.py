"""Core modules: embedder backends and the Weaviate store."""

from .embedder import Embedder, GeminiEmbedder, SentenceTransformerEmbedder, build_embedder
from .weaviate_store import SearchHit, WeaviateStore

__all__ = [
    "Embedder",
    "GeminiEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "SearchHit",
    "WeaviateStore",
]
