"""Weaviate-backed RAG microservice."""

__version__ = "2.0.0"

try:
    from .config import get_settings, settings  # noqa: F401
    from .main import app  # noqa: F401

    __all__ = ["app", "settings", "get_settings", "__version__"]
except ImportError:
    __all__ = ["__version__"]
