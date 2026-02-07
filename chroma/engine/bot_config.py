from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class BotConfig:
    system_prompt: str


def _parse_config_payload(path: Path, raw_text: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix == ".toml":
        payload = tomllib.loads(raw_text)
    else:
        try:
            payload = json.loads(raw_text)
        except Exception:
            try:
                payload = tomllib.loads(raw_text)
            except Exception as toml_exc:
                raise ValueError(
                    f"Unsupported bot config format for '{path}'. Use .json or .toml."
                ) from toml_exc

    if not isinstance(payload, dict):
        raise ValueError(f"Bot config at '{path}' must be a JSON/TOML object.")
    return payload


def _extract_system_prompt(payload: dict[str, Any]) -> str | None:
    direct_prompt = payload.get("system_prompt")
    if isinstance(direct_prompt, str):
        return direct_prompt

    bot_section = payload.get("bot")
    if isinstance(bot_section, dict):
        nested_prompt = bot_section.get("system_prompt")
        if isinstance(nested_prompt, str):
            return nested_prompt
    return None


def load_bot_config(config_path: str | Path) -> BotConfig:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Bot config not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    payload = _parse_config_payload(path, raw_text)
    system_prompt = _extract_system_prompt(payload)

    if system_prompt is None:
        raise ValueError(
            f"Bot config at '{path}' must include 'system_prompt' "
            "(or 'bot.system_prompt')."
        )
    normalized = system_prompt.strip()
    if not normalized:
        raise ValueError(f"Bot config at '{path}' has an empty 'system_prompt'.")

    return BotConfig(system_prompt=normalized)


__all__ = ["BotConfig", "load_bot_config"]
