from __future__ import annotations

import pytest

from chroma.engine.streaming_engine import StreamingVoicebotEngine
from chroma.engine.types import SessionConfig


def _dummy_engine() -> StreamingVoicebotEngine:
    engine = StreamingVoicebotEngine.__new__(StreamingVoicebotEngine)
    engine.default_speaker = "scarlett_johansson"
    engine.default_output_chunk_sec = 0.24
    return engine


def test_normalize_config_accepts_none_system_prompt() -> None:
    engine = _dummy_engine()
    normalized = engine._normalize_config(SessionConfig(system_prompt=None))
    assert normalized.system_prompt is None


def test_normalize_config_trims_system_prompt() -> None:
    engine = _dummy_engine()
    normalized = engine._normalize_config(SessionConfig(system_prompt="  You are a bot.  "))
    assert normalized.system_prompt == "You are a bot."


def test_normalize_config_rejects_empty_system_prompt() -> None:
    engine = _dummy_engine()
    with pytest.raises(ValueError, match="must not be empty"):
        engine._normalize_config(SessionConfig(system_prompt="   "))


def test_normalize_config_rejects_too_long_system_prompt() -> None:
    engine = _dummy_engine()
    with pytest.raises(ValueError, match="4000"):
        engine._normalize_config(SessionConfig(system_prompt="x" * 4001))
