from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

RESET = "\033[0m"
COLOR_CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}
LEVEL_COLOR = {
    "DEBUG": "blue",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


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


def _paint(text: str, color: str | None, enabled: bool) -> str:
    if not enabled or not color:
        return text
    code = COLOR_CODES.get(color)
    if code is None:
        return text
    return f"{code}{text}{RESET}"


def _resolve_color_enabled() -> bool:
    color_mode = os.getenv("CHROMA_LOG_COLOR", "auto").strip().lower()
    if color_mode in {"never", "off", "false", "0"}:
        return False
    if color_mode in {"always", "on", "true", "1", "force"}:
        return True
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("TERM", "").strip().lower() == "dumb":
        return False
    return sys.stderr.isatty()


def _message_highlight_color(message: str) -> str | None:
    if message.startswith("user.input "):
        return "cyan"
    if message.startswith("input context ") and " query=" in message:
        return "cyan"
    if message.startswith("thinker.input_prompt "):
        return "yellow"
    if message.startswith("thinker.output "):
        return "magenta"
    return None


class _PlainFormatter(logging.Formatter):
    def __init__(self, component: str):
        super().__init__()
        self._component = component
        self._use_color = _resolve_color_enabled()

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        component = str(getattr(record, "component", self._component))
        level_name = record.levelname

        if self._use_color:
            level_name = _paint(
                level_name,
                LEVEL_COLOR.get(record.levelname, None),
                enabled=True,
            )
            component = _paint(component, "blue", enabled=True)
            message = _paint(
                message,
                _message_highlight_color(record.getMessage()),
                enabled=True,
            )

        rendered = f"{level_name} | {component} | {message}"
        if record.exc_info:
            rendered = f"{rendered}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            rendered = f"{rendered}\n{self.formatStack(record.stack_info)}"
        return rendered


def _resolve_level(level_name: str | None, env_key: str, default_level: str) -> int:
    raw = (level_name or os.getenv(env_key, default_level)).upper()
    return getattr(logging, raw, logging.INFO)


def _build_formatter(component: str) -> logging.Formatter:
    fmt_name = os.getenv("CHROMA_LOG_FORMAT", "plain").strip().lower()
    if fmt_name == "json":
        return _JsonFormatter(component)
    return _PlainFormatter(component)


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
