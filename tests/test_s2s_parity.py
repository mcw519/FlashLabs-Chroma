from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from chroma.engine.streaming_engine import _EngineAudioStreamer, _SessionState
from chroma.engine.types import SessionConfig


class _CaptureCodecModel:
    def __init__(self) -> None:
        self.last_audio_codes = None

    def decode(self, audio_codes):
        self.last_audio_codes = audio_codes
        return SimpleNamespace(audio_values=torch.zeros((1, 16), dtype=torch.float32))


class _FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.codec_model = _CaptureCodecModel()
        self.config = SimpleNamespace(
            codebook_eos_token_id=0,
            decoder_config=SimpleNamespace(audio_num_codebooks=3),
            vocab_size=1024,
        )


class _CaptureProcessor:
    def __init__(self) -> None:
        self.last_conversation = None
        self.last_kwargs = None

    def __call__(self, conversation, **kwargs):
        self.last_conversation = conversation
        self.last_kwargs = kwargs
        return {
            "thinker_input_ids": torch.ones((1, 1), dtype=torch.long),
            "thinker_attention_mask": torch.ones((1, 1), dtype=torch.long),
        }


def test_audio_decode_clamp_matches_baseline_behavior() -> None:
    model = _FakeModel()
    streamer = _EngineAudioStreamer(
        model=model,
        frames_per_chunk=1,
        cancel_event=threading.Event(),
        on_audio_chunk=lambda _: None,
    )

    streamer._decode_frames([torch.tensor([8000, 8000, 8000])])
    assert model.codec_model.last_audio_codes is not None
    assert int(model.codec_model.last_audio_codes.max().item()) <= 1023


def test_prepare_inputs_uses_audio_first_query_shape() -> None:
    from chroma.engine.streaming_engine import StreamingVoicebotEngine

    engine = StreamingVoicebotEngine.__new__(StreamingVoicebotEngine)
    engine.processor = _CaptureProcessor()
    engine.model_dtype = torch.float32
    engine.device = torch.device("cpu")
    engine.system_prompt = "You are Chroma."
    engine.default_speaker = "scarlett_johansson"
    engine._load_prompt = MethodType(
        lambda self, speaker: (["prompt text"], ["prompt.wav"]),
        engine,
    )

    state = _SessionState(config=SessionConfig())
    audio = np.zeros((1600,), dtype=np.float32)
    inputs = engine._prepare_inputs(
        audio_np=audio,
        speaker="scarlett_johansson",
        state=state,
        session_id="s1",
        turn_id="1",
        query_text=None,
        include_transcript_in_query=False,
    )

    assert "thinker_input_ids" in inputs
    assert engine.processor.last_kwargs is not None
    assert engine.processor.last_kwargs["add_generation_prompt"] is True
    assert engine.processor.last_kwargs["tokenize"] is False
    assert engine.processor.last_conversation is not None
    conv = engine.processor.last_conversation[0]
    assert conv[0]["role"] == "system"
    assert conv[1]["role"] == "user"
    assert conv[1]["content"][0]["type"] == "audio"


def test_commit_turn_adds_user_memory_after_inputs_prepared() -> None:
    from chroma.engine.streaming_engine import StreamingVoicebotEngine

    class _MinimalCommitModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                codec_config=SimpleNamespace(frame_rate=12.5),
                decoder_config=SimpleNamespace(audio_num_codebooks=1),
                codebook_eos_token_id=0,
            )
            self._text_streamer = None

        def generate(self, **kwargs) -> None:
            streamer = kwargs["streamer"]
            streamer.end()

    engine = StreamingVoicebotEngine.__new__(StreamingVoicebotEngine)
    engine.model = _MinimalCommitModel()
    engine.enable_text = False
    engine.max_new_tokens = 16
    engine.max_text_new_tokens = 8
    engine.temperature = 0.7
    engine.top_p = 0.9
    engine.decode_mode = "full_turn"
    engine.overlap_frames = 2
    engine._generate_lock = threading.Lock()
    engine._sessions_lock = threading.Lock()

    state = _SessionState(config=SessionConfig(include_transcript_in_query=False))
    state.pending_user_text = "hello from transcript"
    state.audio_buffer.extend(b"\x00\x00" * 1600)  # 100ms at 16kHz PCM16 mono
    engine._sessions = {"s1": state}

    memory_sizes_seen: list[int] = []

    def _prepare_inputs_stub(
        self,
        audio_np,
        speaker,
        state,
        session_id,
        turn_id,
        query_text,
        include_transcript_in_query,
    ):
        memory_sizes_seen.append(len(state.memory))
        return {}

    engine._prepare_inputs = MethodType(_prepare_inputs_stub, engine)

    async def _drain_events() -> None:
        async for _event in engine.commit_turn("s1"):
            pass

    import asyncio

    asyncio.run(_drain_events())

    assert memory_sizes_seen == [0]
    assert len(state.memory) == 1
    assert state.memory[0].role == "user"
    assert state.memory[0].text == "hello from transcript"
