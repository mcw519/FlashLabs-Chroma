from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class _ComponentFilter(logging.Filter):
    def __init__(self, component: str):
        super().__init__()
        self._component = component

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "component"):
            record.component = self._component
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, component: str):
        super().__init__()
        self._component = component

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", self._component),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _resolve_level(level_name: str | None, env_key: str, default_level: str) -> int:
    raw = (level_name or os.getenv(env_key, default_level)).upper()
    return getattr(logging, raw, logging.INFO)


def _build_formatter(component: str) -> logging.Formatter:
    fmt_name = os.getenv("CHROMA_LOG_FORMAT", "plain").strip().lower()
    if fmt_name == "json":
        return _JsonFormatter(component)
    return logging.Formatter("%(levelname)s | %(component)s | %(message)s")


def configure_component_logger(
    logger: logging.Logger,
    *,
    component: str,
    env_level_key: str = "CHROMA_LOG_LEVEL",
    default_level: str = "INFO",
    level_name: str | None = None,
) -> None:
    if getattr(logger, "_chroma_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_build_formatter(component))
    handler.addFilter(_ComponentFilter(component))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(_resolve_level(level_name, env_level_key, default_level))
    logger.propagate = False
    setattr(logger, "_chroma_configured", True)


def configure_root_logger(
    *,
    component: str,
    env_level_key: str = "CHROMA_LOG_LEVEL",
    default_level: str = "INFO",
    level_name: str | None = None,
) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(_build_formatter(component))
    handler.addFilter(_ComponentFilter(component))
    root.addHandler(handler)
    root.setLevel(_resolve_level(level_name, env_level_key, default_level))


__all__ = ["configure_component_logger", "configure_root_logger"]
