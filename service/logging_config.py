"""Logging setup: one stream, one format, shipper-friendly.

Design:
  * Single structured output. In production, that is JSON on stdout —
    trivial to ship to Datadog, Loki, CloudWatch, or Elastic without
    parsing. In development, that is a colorized single-line format on
    stdout for humans.
  * One optional rolling file sink (same JSON) so self-hosted deployments
    have a local audit trail without a log pipeline.
  * No per-category file split. Filtering by ``component`` /
    ``category`` / ``event`` happens downstream where the tooling is
    already good at it.
  * Every log record includes a stable set of fields: ``timestamp``,
    ``level``, ``component``, ``message``, plus any ``extra`` the caller
    passed. This gives you a clean dashboard schema.

Public API:
  * ``setup_logging(...)`` — call once at startup.
  * ``get_component_logger(name)`` — for module-level loggers.
  * ``get_request_logger()`` / ``get_performance_logger()`` — thin
    wrappers that add a ``category`` tag so downstream queries can
    split by class of event.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

_CONFIGURED = False


def _json_sink(message) -> None:
    """Serialize a loguru record as a compact JSON line on stdout.

    We render ourselves (instead of ``serialize=True``) so the field
    names are stable and the ``extra`` dict is hoisted to top level
    keys rather than nested.
    """
    record = message.record
    payload: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    # Promote extras (component, category, tenant, dataset_id, ...).
    extras = {k: v for k, v in record["extra"].items() if not k.startswith("_")}
    for k, v in extras.items():
        payload[k] = v
    if record["exception"]:
        payload["exception"] = str(record["exception"].value)

    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


_DEV_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<cyan>{extra[component]: <16}</cyan> "
    "<level>{message}</level>"
    "{extra[_context]}"
)


def _dev_formatter(record: dict) -> str:
    """Human-readable line with a compact context suffix."""
    record["extra"].setdefault("component", record["name"].split(".")[-1])
    ignore = {"component"}
    ctx = {
        k: v
        for k, v in record["extra"].items()
        if k not in ignore and not k.startswith("_")
    }
    record["extra"]["_context"] = (
        "  " + " ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
    )
    return _DEV_FORMAT + "\n{exception}"


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: str | None = "./logs",
    retention_days: int = 30,
    rotation_size: str = "100 MB",
) -> None:
    """Install the logging sinks. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    level = log_level.upper()

    if environment.lower() == "production":
        logger.add(
            _json_sink,
            level=level,
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            format=_dev_formatter,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Optional file sink (always JSON, safe for any environment).
    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.add(
            path / "service.jsonl",
            level=level,
            serialize=True,
            rotation=rotation_size,
            retention=f"{retention_days} days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    _CONFIGURED = True
    logger.bind(component="logging").info(
        "Logging configured",
        environment=environment,
        level=level,
        log_dir=log_dir or "(none)",
    )


# ---------------------------------------------------------------------------
# Logger factories
# ---------------------------------------------------------------------------
def get_component_logger(component_name: str):
    """Return a logger bound to a component name."""
    return logger.bind(component=component_name)


def get_request_logger():
    """Logger for HTTP request lifecycle events."""
    return logger.bind(component="http", category="request")


def get_performance_logger():
    """Logger for timing/performance events."""
    return logger.bind(component="perf", category="performance")
