from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from chroma.obs_logging import configure_component_logger
from chroma.pretrained import get_pretrained_cache_dir

logger = logging.getLogger(__name__)
configure_component_logger(
    logger,
    component="server_asr",
    env_level_key="CHROMA_SERVER_LOG_LEVEL",
    default_level="INFO",
)

INPUT_SAMPLE_RATE = 16000


class AudioTranscriber(Protocol):
    def transcribe_pcm16_16k(self, pcm16_16k: bytes) -> str | None: ...


@dataclass(slots=True)
class ServerASRConfig:
    model_id: str
    language: str | None = None
    device: str = "auto"


class TransformersASRTranscriber:
    def __init__(self, config: ServerASRConfig):
        self.config = config
        self._lock = threading.Lock()
        device, torch_dtype = self._resolve_device(config.device)
        cache_dir = get_pretrained_cache_dir()
        try:
            from transformers import pipeline as hf_pipeline
        except Exception as exc:
            raise RuntimeError(
                "Server ASR requires 'transformers'. Install dependencies and retry."
            ) from exc
        logger.info(
            "Initializing server ASR model=%s device=%s dtype=%s cache_dir=%s",
            config.model_id,
            device,
            torch_dtype,
            cache_dir,
        )
        self._pipeline = hf_pipeline(
            task="automatic-speech-recognition",
            model=config.model_id,
            device=device,
            dtype=torch_dtype,
            model_kwargs={"cache_dir": str(cache_dir)},
        )

    def transcribe_pcm16_16k(self, pcm16_16k: bytes) -> str | None:
        if not pcm16_16k:
            return None
        audio = _pcm16le_bytes_to_float32_mono(pcm16_16k)
        if audio.size < INPUT_SAMPLE_RATE // 20:
            return None
        generate_kwargs = {"task": "transcribe"}
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        payload = {"array": audio, "sampling_rate": INPUT_SAMPLE_RATE}
        with self._lock:
            result = self._pipeline(
                payload,
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
        text = ""
        if isinstance(result, dict):
            text = str(result.get("text", "") or "")
        elif isinstance(result, str):
            text = result
        normalized = text.strip()
        return normalized or None

    @staticmethod
    def _resolve_device(device: str) -> tuple[int, torch.dtype]:
        key = (device or "auto").strip().lower()
        if key == "auto":
            key = "cuda" if torch.cuda.is_available() else "cpu"
        if key == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("ASR device=cuda requested but CUDA is not available")
            return 0, torch.float16
        if key == "cpu":
            return -1, torch.float32
        raise ValueError("Invalid ASR device; use one of: auto, cpu, cuda")

def build_server_asr_transcriber(
    *,
    model_id: str | None,
    language: str | None = None,
    device: str = "auto",
) -> AudioTranscriber | None:
    if model_id is None or not model_id.strip():
        return None
    config = ServerASRConfig(
        model_id=model_id.strip(),
        language=(language or "").strip() or None,
        device=device,
    )
    return TransformersASRTranscriber(config)


def _pcm16le_bytes_to_float32_mono(pcm16_16k: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(pcm16_16k, dtype=np.int16)
    if audio_i16.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (audio_i16.astype(np.float32) / 32768.0).copy()
