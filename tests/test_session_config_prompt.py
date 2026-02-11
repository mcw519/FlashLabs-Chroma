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


def test_should_auto_commit_uses_incremental_vad_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _dummy_engine()
    engine.max_input_bytes = 400000
    engine._sessions_lock = threading.Lock()
    engine._vad_model = object()
    state = _SessionState(
        config=SessionConfig(
            turn_detection=TurnDetectionConfig(mode="server_vad"),
        )
    )
    state.audio_buffer.extend(b"\x00\x00" * 64000)
    engine._sessions = {"s1": state}
    created_iterators = []

    class _FakeVADIterator:
        def __init__(
            self,
            model,
            threshold: float,
            sampling_rate: int,
            min_silence_duration_ms: int,
            speech_pad_ms: int,
        ) -> None:
            self.model = model
            self.threshold = threshold
            self.sampling_rate = sampling_rate
            self.min_silence_duration_ms = min_silence_duration_ms
            self.speech_pad_ms = speech_pad_ms
            self.current_sample = 0
            self.calls = 0
            created_iterators.append(self)

        def reset_states(self) -> None:
            self.current_sample = 0

        def __call__(self, x):
            self.calls += 1
            self.current_sample += int(x.numel())
            return None

    monkeypatch.setattr("chroma.engine.streaming_engine.VADIterator", _FakeVADIterator)

    ready1, reason1, _, _ = engine.should_auto_commit("s1")
    assert ready1 is False
    assert reason1 == "no_speech"
    assert len(created_iterators) == 1
    first_calls = created_iterators[0].calls
    assert first_calls == 125

    engine.append_audio("s1", b"\x00\x00" * 1600)
    ready2, reason2, _, _ = engine.should_auto_commit("s1")
    assert ready2 is False
    assert reason2 == "no_speech"
    second_delta = created_iterators[0].calls - first_calls
    assert second_delta == 3


def test_should_auto_commit_skips_vad_when_no_new_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _dummy_engine()
    engine.max_input_bytes = 400000
    engine._sessions_lock = threading.Lock()
    engine._vad_model = object()
    state = _SessionState(
        config=SessionConfig(
            turn_detection=TurnDetectionConfig(mode="server_vad"),
        )
    )
    state.audio_buffer.extend(b"\x00\x00" * 16000)
    engine._sessions = {"s1": state}
    created_iterators = []

    class _FakeVADIterator:
        def __init__(
            self,
            model,
            threshold: float,
            sampling_rate: int,
            min_silence_duration_ms: int,
            speech_pad_ms: int,
        ) -> None:
            self.model = model
            self.threshold = threshold
            self.sampling_rate = sampling_rate
            self.min_silence_duration_ms = min_silence_duration_ms
            self.speech_pad_ms = speech_pad_ms
            self.current_sample = 0
            self.calls = 0
            created_iterators.append(self)

        def reset_states(self) -> None:
            self.current_sample = 0

        def __call__(self, x):
            self.calls += 1
            self.current_sample += int(x.numel())
            return None

    monkeypatch.setattr("chroma.engine.streaming_engine.VADIterator", _FakeVADIterator)

    ready1, _, _, _ = engine.should_auto_commit("s1")
    assert len(created_iterators) == 1
    first_calls = created_iterators[0].calls
    ready2, _, _, _ = engine.should_auto_commit("s1")
    assert ready1 is False
    assert ready2 is False
    assert created_iterators[0].calls == first_calls


def test_should_auto_commit_enforces_min_speech_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _dummy_engine()
    engine.max_input_bytes = 400000
    engine._sessions_lock = threading.Lock()
    engine._vad_model = object()
    state = _SessionState(
        config=SessionConfig(
            turn_detection=TurnDetectionConfig(
                mode="server_vad",
                min_speech_ms=250,
                min_silence_ms=200,
            ),
        )
    )
    state.audio_buffer.extend(b"\x00\x00" * 12000)
    engine._sessions = {"s1": state}

    class _FakeVADIterator:
        def __init__(
            self,
            model,
            threshold: float,
            sampling_rate: int,
            min_silence_duration_ms: int,
            speech_pad_ms: int,
        ) -> None:
            self.current_sample = 0
            self.calls = 0

        def reset_states(self) -> None:
            self.current_sample = 0
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            self.current_sample += int(x.numel())
            if self.calls == 1:
                return {"start": 0}
            if self.calls == 2:
                return {"end": 512}
            return None

    monkeypatch.setattr("chroma.engine.streaming_engine.VADIterator", _FakeVADIterator)

    ready, reason, _, _ = engine.should_auto_commit("s1")
    assert ready is False
    assert reason == "speech_too_short"
