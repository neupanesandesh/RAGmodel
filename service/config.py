"""
Typed configuration loaded from environment / .env.

All settings flow through a single `Settings` instance. Validation is strict:
bad production config fails fast at startup rather than at first request.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- Embedder --------
    embedder_backend: str = Field(default="sentence-transformers")
    embedder_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimension: int = Field(default=384, ge=64, le=4096)
    embedder_device: Optional[str] = Field(default=None)

    # -------- Qdrant --------
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_timeout: int = Field(default=30, ge=1, le=600)
    qdrant_collection: str = Field(default="ragmodel")
    hybrid_search_enabled: bool = Field(default=True)
    sparse_model: str = Field(default="Qdrant/bm25")

    # -------- Service --------
    service_host: str = Field(default="0.0.0.0")
    service_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000")
    api_key: str = Field(default="")
    admin_api_key: str = Field(default="")
    rate_limit: str = Field(default="60/minute")

    # -------- Observability --------
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    metrics_enabled: bool = Field(default=True)

    @field_validator("embedder_backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        allowed = {"sentence-transformers", "fastembed"}
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"embedder_backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("environment")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"development", "staging", "production"}:
            raise ValueError("environment must be development|staging|production")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("invalid log_level")
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()


def validate_settings() -> None:
    """Raise if settings are unsafe for the selected environment."""
    errors: List[str] = []

    if not settings.qdrant_url:
        errors.append("QDRANT_URL is required")

    if settings.is_production:
        if not settings.api_key or settings.api_key == "change-me-in-production":
            errors.append("API_KEY must be set to a real value in production")
        if "*" in settings.cors_origins_list:
            errors.append("CORS_ORIGINS must not be '*' in production")

    if errors:
        raise ValueError(
            "Configuration invalid:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def get_settings() -> Settings:
    return settings
