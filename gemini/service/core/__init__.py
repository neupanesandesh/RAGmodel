"""
Core modules for the embedding service.
"""

from .chunking import chunk_text
from .embedder import GeminiEmbedder
from .vectorstore import QdrantStore

__all__ = ["chunk_text", "GeminiEmbedder", "QdrantStore"]
