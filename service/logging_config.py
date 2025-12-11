"""
Centralized logging configuration using Loguru.

This module configures structured logging for the RAGmodel microservice with:
- Environment-aware configuration (development vs production)
- Categorized log files (app, errors, requests, performance)
- Automatic log rotation and retention
- JSON formatting for production (easy parsing)
- Colored console output for development
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class LogConfig:
    """Logging configuration for the RAGmodel service."""

    def __init__(
        self,
        environment: str = "development",
        log_level: str = "INFO",
        log_dir: str = "./logs",
        retention_days: int = 30,
        rotation_size: str = "100 MB",
    ):
        """
        Initialize logging configuration.

        Args:
            environment: "development" or "production"
            log_level: Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            retention_days: Number of days to keep old logs
            rotation_size: Maximum size before rotation (e.g., "100 MB", "500 MB")
        """
        self.environment = environment.lower()
        self.log_level = log_level.upper()
        self.log_dir = Path(log_dir)
        self.retention = f"{retention_days} days"
        self.rotation = rotation_size

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track if logging has been configured
        self._configured = False

    def configure(self) -> None:
        """Configure loguru logger with environment-specific settings."""
        if self._configured:
            logger.warning("Logging already configured, skipping reconfiguration")
            return

        # Remove default handler
        logger.remove()

        # Configure console handler based on environment
        self._configure_console()

        # Configure file handlers
        self._configure_file_handlers()

        # Add custom log levels if needed
        self._configure_custom_levels()

        self._configured = True
        logger.info(
            "Logging configured",
            extra={
                "environment": self.environment,
                "log_level": self.log_level,
                "log_dir": str(self.log_dir),
            }
        )

    def _configure_console(self) -> None:
        """Configure console output based on environment."""
        if self.environment == "production":
            # Production: JSON format for structured logging
            logger.add(
                sys.stderr,
                level=self.log_level,
                format="{message}",
                serialize=True,  # JSON serialization
                backtrace=True,
                diagnose=False,  # Don't expose variable values in production
            )
        else:
            # Development: Colored, human-readable format
            logger.add(
                sys.stderr,
                level=self.log_level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                colorize=True,
                backtrace=True,
                diagnose=True,  # Show variable values in development
            )

    def _configure_file_handlers(self) -> None:
        """Configure file handlers for different log categories."""

        # 1. General application logs
        logger.add(
            self.log_dir / "app.log",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            enqueue=True,  # Thread-safe, async-safe
            backtrace=True,
            diagnose=self.environment != "production",
        )

        # 2. Error logs only (WARNING and above)
        logger.add(
            self.log_dir / "error.log",
            level="WARNING",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=self.environment != "production",
        )

        # 3. Request logs (filtered by context)
        logger.add(
            self.log_dir / "requests.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            enqueue=True,
            filter=lambda record: record.get("extra", {}).get("category") == "request",
        )

        # 4. Performance logs (filtered by context)
        logger.add(
            self.log_dir / "performance.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            enqueue=True,
            filter=lambda record: record.get("extra", {}).get("category") == "performance",
        )

        # 5. Production JSON logs (for log aggregation tools)
        if self.environment == "production":
            logger.add(
                self.log_dir / "app.json",
                level=self.log_level,
                format="{message}",
                serialize=True,  # JSON format
                rotation=self.rotation,
                retention=self.retention,
                compression="zip",
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )

    def _configure_custom_levels(self) -> None:
        """Add custom log levels if needed."""
        # Add a SUCCESS level for business logic milestones
        # (Loguru already has success() method, just documenting usage)
        pass

    def get_logger(self, name: Optional[str] = None):
        """
        Get a logger instance with optional name binding.

        Args:
            name: Module or component name for context

        Returns:
            Logger instance with bound context
        """
        if not self._configured:
            self.configure()

        if name:
            return logger.bind(component=name)
        return logger


# Convenience function for creating categorized loggers
def get_request_logger():
    """Get logger for HTTP requests."""
    return logger.bind(category="request")


def get_performance_logger():
    """Get logger for performance metrics."""
    return logger.bind(category="performance")


def get_component_logger(component_name: str):
    """
    Get logger for a specific component.

    Args:
        component_name: Name of the component (e.g., "embedder", "vectorstore")

    Returns:
        Logger instance with component context
    """
    return logger.bind(component=component_name)


# Global logging configuration instance
_log_config: Optional[LogConfig] = None


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: str = "./logs",
    retention_days: int = 30,
    rotation_size: str = "100 MB",
) -> None:
    """
    Setup logging for the application.

    This should be called once during application startup.

    Args:
        environment: "development" or "production"
        log_level: Minimum log level
        log_dir: Directory for log files
        retention_days: Number of days to keep old logs
        rotation_size: Maximum size before rotation
    """
    global _log_config

    if _log_config is not None:
        logger.warning("Logging already setup, skipping")
        return

    _log_config = LogConfig(
        environment=environment,
        log_level=log_level,
        log_dir=log_dir,
        retention_days=retention_days,
        rotation_size=rotation_size,
    )
    _log_config.configure()


def get_logger_instance():
    """Get the global logger instance."""
    if _log_config is None:
        # Auto-configure with defaults if not setup
        setup_logging()
    return logger
