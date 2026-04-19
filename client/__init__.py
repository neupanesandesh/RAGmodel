"""Python client for the Weaviate-backed RAG service."""

from .client import RAGClient, SearchHit

__version__ = "2.0.0"
__all__ = ["RAGClient", "SearchHit"]
