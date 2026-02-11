from __future__ import annotations

import threading

import pytest

from chroma.engine.streaming_engine import StreamingVoicebotEngine, _SessionState
from chroma.engine.types import SessionConfig, TurnDetectionConfig


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


def test_normalize_turn_detection_mode_defaults_to_server_vad() -> None:
    engine = _dummy_engine()
    normalized = engine._normalize_config(
        SessionConfig(
            turn_detection=TurnDetectionConfig(mode="bad_mode"),  # type: ignore[arg-type]
        )
    )
    assert normalized.turn_detection.mode == "server_vad"


def test_normalize_turn_detection_threshold_bounds() -> None:
    engine = _dummy_engine()
    normalized = engine._normalize_config(
        SessionConfig(
            turn_detection=TurnDetectionConfig(
                mode="server_vad",
                threshold=999.0,
            )
        )
    )
    assert normalized.turn_detection.threshold == 0.5


def test_should_auto_commit_disabled_in_client_commit_mode() -> None:
    engine = _dummy_engine()
    engine.max_input_bytes = 16000
    engine._sessions_lock = threading.Lock()
    state = _SessionState(
        config=SessionConfig(
            turn_detection=TurnDetectionConfig(mode="client_commit"),
        )
    )
    state.audio_buffer.extend(b"\x01\x00" * 400)
    engine._sessions = {"s1": state}

    ready, reason, _, _ = engine.should_auto_commit("s1")
    assert ready is False
    assert reason == "mode_client_commit"
