from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chroma.engine.streaming_engine import GenerationCancelled, _EngineAudioStreamer


class _FakeCodecModel:
    def decode(self, audio_codes):
        frames = audio_codes.shape[-1]
        audio = torch.ones((1, frames * 10), dtype=torch.float32)
        return SimpleNamespace(audio_values=audio)


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.codec_model = _FakeCodecModel()
        self.config = SimpleNamespace(
            codebook_eos_token_id=0,
            decoder_config=SimpleNamespace(audio_num_codebooks=3),
        )


def test_streamer_chunking_and_flush() -> None:
    model = _FakeModel()
    cancel_event = threading.Event()
    chunks: list[np.ndarray] = []

    streamer = _EngineAudioStreamer(
        model=model,
        frames_per_chunk=2,
        cancel_event=cancel_event,
        on_audio_chunk=lambda chunk: chunks.append(chunk),
    )

    streamer.put(torch.tensor([1, 2, 3]))
    streamer.put(torch.tensor([1, 2, 3]))
    streamer.put(torch.tensor([1, 2, 3]))
    streamer.end()

    assert len(chunks) == 2
    assert chunks[0].shape[-1] == 20
    assert chunks[1].shape[-1] == 10


def test_streamer_cancel_raises() -> None:
    model = _FakeModel()
    cancel_event = threading.Event()
    cancel_event.set()

    streamer = _EngineAudioStreamer(
        model=model,
        frames_per_chunk=2,
        cancel_event=cancel_event,
        on_audio_chunk=lambda _: None,
    )

    with pytest.raises(GenerationCancelled):
        streamer.put(torch.tensor([1, 2, 3]))
