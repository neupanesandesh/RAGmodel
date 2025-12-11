"""
Configuration Management

Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):

    # Gemini Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "models/gemini-embedding-001")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)

    # Service Configuration
    service_host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("SERVICE_PORT", "8000"))

    # Logging Configuration
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    log_rotation_size: str = os.getenv("LOG_ROTATION_SIZE", "100 MB")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def validate_settings():

    errors = []

    # Check Gemini API key
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required but not set")

    # Check embedding dimension is valid
    valid_dimensions = [768, 1536, 3072]
    if settings.embedding_dimension not in valid_dimensions:
        errors.append(
            f"EMBEDDING_DIMENSION must be one of {valid_dimensions}, "
            f"got {settings.embedding_dimension}"
        )

    # Check Qdrant URL is set
    if not settings.qdrant_url:
        errors.append("QDRANT_URL is required but not set")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings instance
    """
    return settings



