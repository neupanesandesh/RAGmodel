"""
RAGmodel

Production-grade multi-tenant RAG service on Qdrant with local open-source
embeddings, hybrid search, and native observability.
"""

__version__ = "2.0.0"

# Export main components for easier imports
try:
    from .config import settings, get_settings
    from .main import app
    __all__ = ["app", "settings", "get_settings", "__version__"]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed
    __all__ = ["__version__"]
