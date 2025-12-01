"""
Embedding Service Client

A simple Python client for the Embedding Service.
"""

from .client import EmbeddingClient, SearchResult, create_client

__version__ = "1.0.0"
__all__ = ["EmbeddingClient", "SearchResult", "create_client"]
