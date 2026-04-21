"""
Configuration validation tests.
"""

import pytest

from service.config import Settings, validate_settings


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        s = Settings()
        assert s.embedder_backend == "sentence-transformers"
        assert s.embedder_model == "BAAI/bge-small-en-v1.5"
        assert s.embedding_dimension == 384
        assert s.qdrant_url == "http://localhost:6333"
        assert s.qdrant_collection == "ragmodel"
        assert s.hybrid_search_enabled is True
        assert s.is_production is False

    def test_invalid_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("EMBEDDER_BACKEND", "not-a-real-backend")
        with pytest.raises(ValueError):
            Settings()

    def test_invalid_environment_rejected(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "bogus")
        with pytest.raises(ValueError):
            Settings()

    def test_invalid_log_level_rejected(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "SHOUT")
        with pytest.raises(ValueError):
            Settings()

    def test_cors_origins_list(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com ,http://c.com")
        s = Settings()
        assert s.cors_origins_list == ["http://a.com", "http://b.com", "http://c.com"]


class TestValidateSettings:
    """validate_settings() should block unsafe prod config."""

    def test_production_requires_real_api_key(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY", "change-me-in-production")
        monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")

        import service.config as cfg
        cfg.settings = Settings()
        with pytest.raises(ValueError, match="API_KEY"):
            validate_settings()

    def test_production_rejects_wildcard_cors(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY", "real-secret-key")
        monkeypatch.setenv("CORS_ORIGINS", "*")

        import service.config as cfg
        cfg.settings = Settings()
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            validate_settings()

    def test_development_permissive(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("API_KEY", "")
        monkeypatch.setenv("CORS_ORIGINS", "*")

        import service.config as cfg
        cfg.settings = Settings()
        validate_settings()
