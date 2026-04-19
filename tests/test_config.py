"""Unit tests for configuration validation."""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_config(env: dict[str, str]):
    """Reload service.config with a clean env snapshot."""
    for k in list(os.environ):
        if k.startswith(("WEAVIATE_", "EMBEDDER_", "GEMINI_", "API_KEY", "ENVIRONMENT", "EMBEDDING_")):
            os.environ.pop(k, None)
    os.environ.update(env)
    from service import config

    return importlib.reload(config)


def test_defaults_are_valid_in_dev():
    cfg = _reload_config({"ENVIRONMENT": "development"})
    cfg.validate_settings()  # should not raise


def test_production_requires_api_key():
    cfg = _reload_config({"ENVIRONMENT": "production"})
    with pytest.raises(ValueError, match="API_KEY"):
        cfg.validate_settings()


def test_gemini_backend_requires_api_key():
    cfg = _reload_config({"EMBEDDER_BACKEND": "gemini"})
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        cfg.validate_settings()


def test_unknown_backend_rejected():
    cfg = _reload_config({"EMBEDDER_BACKEND": "pinecone"})
    with pytest.raises(ValueError, match="EMBEDDER_BACKEND"):
        cfg.validate_settings()
