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
