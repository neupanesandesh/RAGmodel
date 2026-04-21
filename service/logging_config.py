"""
Structured logging — single-stream, cloud-native.

- development: colored, human-readable logs on stderr
- production: one JSON line per event on stdout (for log aggregators)

No file handlers: the container runtime (Docker, k8s, etc.) is responsible
for log collection. This keeps the service 12-factor-compliant.
"""

import sys
from typing import Optional
from loguru import logger


_configured = False


def setup_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """Configure loguru. Idempotent — safe to call multiple times."""
    global _configured
    if _configured:
        return

    logger.remove()
    level = log_level.upper()

    if environment.lower() == "production":
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[component]}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    logger.configure(extra={"component": "app"})
    _configured = True
    logger.bind(component="logging").info(
        "logging initialised", environment=environment, level=level
    )


def get_component_logger(component_name: str):
    """Return a logger bound to a component name for structured filtering."""
    if not _configured:
        setup_logging()
    return logger.bind(component=component_name)


def get_request_logger():
    return logger.bind(component="request", category="request")


def get_performance_logger():
    return logger.bind(component="performance", category="performance")


def get_logger_instance(name: Optional[str] = None):
    if not _configured:
        setup_logging()
    return logger.bind(component=name) if name else logger
