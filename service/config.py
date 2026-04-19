"""Application configuration.

All runtime configuration is read from environment variables (and from .env
in development). Settings are validated at startup; an invalid configuration
fails fast rather than degrading silently at request time.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- Weaviate ---------------------------------------------------------
    weaviate_host: str = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_http_port: int = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    weaviate_grpc_port: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    weaviate_api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY") or None
    weaviate_collection: str = os.getenv("WEAVIATE_COLLECTION", "Document")

    # --- Embedder --------------------------------------------------------
    # "st" = sentence-transformers (local, open source, default).
    # "gemini" = Google Gemini (requires GEMINI_API_KEY).
    embedder_backend: str = os.getenv("EMBEDDER_BACKEND", "st")
    st_model: str = os.getenv("ST_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or None
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
    )
    gemini_generative_model: str = os.getenv(
        "GEMINI_GENERATIVE_MODEL", "gemini-2.5-flash"
    )

    # --- Service ---------------------------------------------------------
    service_host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("SERVICE_PORT", "8000"))
    api_key: Optional[str] = os.getenv("API_KEY") or None
    cors_allowed_origins: str = os.getenv("CORS_ALLOWED_ORIGINS", "")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

    # --- Observability ---------------------------------------------------
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    log_rotation_size: str = os.getenv("LOG_ROTATION_SIZE", "100 MB")
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


def validate_settings() -> None:
    """Validate configuration. Raises ValueError on misconfiguration."""
    errors: list[str] = []

    if settings.embedder_backend not in {"st", "gemini"}:
        errors.append(
            f"EMBEDDER_BACKEND must be 'st' or 'gemini', got {settings.embedder_backend!r}"
        )

    if settings.embedder_backend == "gemini" and not settings.gemini_api_key:
        errors.append("EMBEDDER_BACKEND=gemini requires GEMINI_API_KEY")

    if settings.embedding_dimension <= 0:
        errors.append("EMBEDDING_DIMENSION must be a positive integer")

    if not settings.weaviate_host:
        errors.append("WEAVIATE_HOST is required")

    if settings.environment == "production" and not settings.api_key:
        errors.append(
            "API_KEY is required when ENVIRONMENT=production (refuses to run unauthenticated)"
        )

    if errors:
        joined = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Configuration validation failed:\n{joined}")


def get_settings() -> Settings:
    return settings
