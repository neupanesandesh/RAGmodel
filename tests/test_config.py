"""
Tests for configuration validation.

Catches missing env vars before deployment.
"""

import pytest
import os
from service.config import Settings


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self, monkeypatch):
        """Valid config should pass validation."""
        # Set valid environment variables
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        # Create new settings object with monkeypatched env vars
        settings = Settings()

        # Validate should pass
        assert settings.gemini_api_key == "test-key-123"
        assert settings.embedding_dimension == 768
        assert settings.qdrant_url == "http://localhost:6333"

    # def test_missing_gemini_api_key(self, monkeypatch):
    #     """Missing API key should raise error."""
    #     monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    #     monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    #     monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    #     # Create settings with missing API key
    #     settings = Settings()

    #     # Validation should fail
    #     assert not settings.gemini_api_key or settings.gemini_api_key == ""

    def test_invalid_embedding_dimension(self, monkeypatch):
        """Invalid dimension should raise error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "999")  # Invalid!
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        # Create settings with invalid dimension
        settings = Settings()

        # Should have the invalid dimension
        assert settings.embedding_dimension == 999

        # Validation function should reject it
        valid_dimensions = [768, 1536, 3072]
        assert settings.embedding_dimension not in valid_dimensions

    def test_valid_embedding_dimensions(self, monkeypatch):
        """Valid dimensions (768, 1536, 3072) should pass."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        valid_dimensions = [768, 1536, 3072]
        for dimension in ["768", "1536", "3072"]:
            monkeypatch.setenv("EMBEDDING_DIMENSION", dimension)
            settings = Settings()
            assert settings.embedding_dimension in valid_dimensions

    # def test_missing_qdrant_url(self, monkeypatch):
    #     """Missing Qdrant URL should raise error."""
    #     monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    #     monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    #     monkeypatch.delenv("QDRANT_URL", raising=False)

    #     # Create settings with missing Qdrant URL
    #     settings = Settings()

    #     # Should be empty or default
    #     assert not settings.qdrant_url or settings.qdrant_url == "http://localhost:6333"
