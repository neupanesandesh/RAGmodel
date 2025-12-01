"""
Embedding Service

A production-grade embedding service using Gemini and Qdrant.
"""

__version__ = "1.0.0"

# Export main components for easier imports
try:
    from .config import settings, get_settings
    from .main import app
    __all__ = ["app", "settings", "get_settings", "__version__"]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed
    __all__ = ["__version__"]
