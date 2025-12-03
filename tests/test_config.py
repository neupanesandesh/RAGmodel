"""
Tests for configuration validation.

Catches missing env vars before deployment.
"""

import pytest
import os
from service.config import validate_settings


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self, monkeypatch):
        """Valid config should pass validation."""
        # Set valid environment variables
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        # Should not raise
        validate_settings()

    def test_missing_gemini_api_key(self, monkeypatch):
        """Missing API key should raise error."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            validate_settings()

    def test_invalid_embedding_dimension(self, monkeypatch):
        """Invalid dimension should raise error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "999")  # Invalid!
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with pytest.raises(ValueError, match="EMBEDDING_DIMENSION"):
            validate_settings()

    def test_valid_embedding_dimensions(self, monkeypatch):
        """Valid dimensions (768, 1536, 3072) should pass."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        for dimension in ["768", "1536", "3072"]:
            monkeypatch.setenv("EMBEDDING_DIMENSION", dimension)
            validate_settings()  # Should not raise

    def test_missing_qdrant_url(self, monkeypatch):
        """Missing Qdrant URL should raise error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        monkeypatch.delenv("QDRANT_URL", raising=False)

        with pytest.raises(ValueError, match="QDRANT_URL"):
            validate_settings()
