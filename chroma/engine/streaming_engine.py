from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import colorlog
import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from .bot_config import load_bot_config
from .types import (
    EngineEvent,
    ErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseCancelledEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
    ResponseTextDeltaEvent,
    SessionConfig,
)

logger = logging.getLogger(__name__)

_COLORLOG_CONFIGURED = False


def _configure_engine_logging() -> None:
    global _COLORLOG_CONFIGURED
    if _COLORLOG_CONFIGURED:
        return

    level_name = os.getenv("CHROMA_ENGINE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s | %(cyan)sengine%(reset)s | %(message)s"
    )
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _COLORLOG_CONFIGURED = True


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, np.ndarray):
        return str(value.shape)
    return type(value).__name__


def _clip_text(text: str | None, max_chars: int = 160) -> str:
    if not text:
        return ""
    stripped = text.strip().replace("\n", " ")
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3] + "..."


_ANSI_RESET = "\033[0m"
_ANSI_QUERY = "\033[38;5;214m"
_ANSI_HISTORY = "\033[38;5;45m"
_UNSET = object()


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_ANSI_RESET}"

PROMPT_SPEAKERS = [
    "scarlett_johansson",
    "ariana_grande",
    "donald_trump",
    "lebron_james",
]
SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


class GenerationCancelled(RuntimeError):
    pass


class AbortOnCancelCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self._cancel_event = cancel_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self._cancel_event.is_set()


@dataclass(slots=True)
class _MemoryItem:
    role: str
    text: str


@dataclass(slots=True)
class _SessionState:
    config: SessionConfig
    audio_buffer: bytearray = field(default_factory=bytearray)
    memory: list[_MemoryItem] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    turn_index: int = 0
    active_turn_id: str | None = None
    active_cancel_event: threading.Event | None = None
    active_thread: threading.Thread | None = None
    cancel_requested_at: float | None = None
    pending_user_text: str | None = None


class _EngineAudioStreamer(BaseStreamer):
    def __init__(
        self,
        model,
        frames_per_chunk: int,
        cancel_event: threading.Event,
        on_audio_chunk,
        log_prefix: str = "",
    ):
        self.model = model
        self.frames_per_chunk = frames_per_chunk
        self.cancel_event = cancel_event
        self.on_audio_chunk = on_audio_chunk
        self.log_prefix = log_prefix
        self.eos_token_id = model.config.codebook_eos_token_id
        self.num_codebooks = model.config.decoder_config.audio_num_codebooks
        self._buffer: list[torch.Tensor] = []
        self._closed = False
        logger.debug(
            "%sstreamer initialized frames_per_chunk=%s num_codebooks=%s",
            self.log_prefix,
            frames_per_chunk,
            self.num_codebooks,
        )

    def put(self, value) -> None:
        if self.cancel_event.is_set():
            logger.debug("%sstreamer cancel flag observed in put()", self.log_prefix)
            raise GenerationCancelled("Generation cancelled")
        if self._closed:
            logger.debug("%sstreamer ignored put() because streamer closed", self.log_prefix)
            return
        tokens = torch.as_tensor(value)
        logger.debug("%sstreamer put() token_shape=%s", self.log_prefix, _shape(tokens))
        if tokens.ndim == 2:
            if tokens.shape[0] == 1 and tokens.shape[1] == self.num_codebooks:
                tokens = tokens[0]
            elif tokens.shape[1] == self.num_codebooks:
                for row in tokens:
                    self._put_frame(row)
                return
            else:
                logger.debug(
                    "%sstreamer ignored 2D tokens with unexpected shape=%s",
                    self.log_prefix,
                    _shape(tokens),
                )
                return
        if tokens.ndim != 1:
            logger.debug("%sstreamer ignored tokens with ndim=%s", self.log_prefix, tokens.ndim)
            return
        if tokens.numel() != self.num_codebooks:
            logger.debug(
                "%sstreamer ignored tokens with numel=%s expected=%s",
                self.log_prefix,
                tokens.numel(),
                self.num_codebooks,
            )
            return
        self._put_frame(tokens)

    def _put_frame(self, tokens: torch.Tensor) -> None:
        if self.cancel_event.is_set():
            logger.debug("%sstreamer cancel flag observed in _put_frame()", self.log_prefix)
            raise GenerationCancelled("Generation cancelled")
        if (tokens == self.eos_token_id).all():
            logger.debug("%sstreamer EOS frame received; ending stream", self.log_prefix)
            self.end()
            return
        self._buffer.append(tokens)
        logger.debug(
            "%sstreamer buffered frame_count=%s/%s",
            self.log_prefix,
            len(self._buffer),
            self.frames_per_chunk,
        )
        if len(self._buffer) >= self.frames_per_chunk:
            self._emit_frames(self.frames_per_chunk)

    def end(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.debug(
            "%sstreamer end() flushing remaining_frames=%s",
            self.log_prefix,
            len(self._buffer),
        )
        if self._buffer:
            self._emit_frames(len(self._buffer))
            self._buffer.clear()

    def _emit_frames(self, count: int) -> None:
        logger.debug("%sstreamer emit_frames count=%s", self.log_prefix, count)
        frames = self._buffer[:count]
        del self._buffer[:count]
        audio_np = self._decode_frames(frames)
        if audio_np.size == 0:
            logger.debug("%sstreamer decoded empty audio chunk; skip emit", self.log_prefix)
            return
        logger.debug(
            "%sstreamer emit audio_chunk shape=%s samples=%s",
            self.log_prefix,
            _shape(audio_np),
            audio_np.shape[-1] if audio_np.ndim > 0 else 0,
        )
        self.on_audio_chunk(audio_np)

    @torch.no_grad()
    def _decode_frames(self, frames: list[torch.Tensor]) -> np.ndarray:
        logger.debug("%sstreamer decode_frames frame_count=%s", self.log_prefix, len(frames))
        audio_codes = torch.stack(frames).to(self.model.device)
        audio_values = self.model.codec_model.decode(
            audio_codes.transpose(0, 1).unsqueeze(0)
        ).audio_values
        audio_np = audio_values[0].detach().float().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        return audio_np


class StreamingVoicebotEngine:
    def __init__(
        self,
        model_path: str | None,
        *,
        use_half_precision: bool,
        max_new_tokens: int,
        max_text_new_tokens: int,
        temperature: float,
        top_p: float,
        output_chunk_sec: float = 0.24,
        default_speaker: str = "scarlett_johansson",
        enable_text: bool = True,
        max_sessions: int = 3,
        max_input_seconds: float = 30.0,
        warmup: bool = False,
        bot_config_path: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        _configure_engine_logging()
        logger.debug(
            "engine init start model_path=%s half_precision=%s max_new_tokens=%s max_text_new_tokens=%s "
            "temperature=%s top_p=%s output_chunk_sec=%s default_speaker=%s enable_text=%s max_sessions=%s "
            "max_input_seconds=%s warmup=%s bot_config_path=%s has_custom_system_prompt=%s",
            model_path,
            use_half_precision,
            max_new_tokens,
            max_text_new_tokens,
            temperature,
            top_p,
            output_chunk_sec,
            default_speaker,
            enable_text,
            max_sessions,
            max_input_seconds,
            warmup,
            bot_config_path,
            system_prompt is not None,
        )
        self.model, self.processor = self._load_chroma_model(
            from_local_path=model_path,
            use_half_precision=use_half_precision,
        )
        self.model.eval()
        self.device = self.model.device
        self.model_dtype = self.model.dtype
        self.max_new_tokens = max_new_tokens
        self.max_text_new_tokens = max_text_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.default_output_chunk_sec = output_chunk_sec
        self.default_speaker = (
            default_speaker if default_speaker in PROMPT_SPEAKERS else PROMPT_SPEAKERS[0]
        )
        self.enable_text = enable_text
        self.max_sessions = max_sessions
        self.max_input_bytes = int(max_input_seconds * INPUT_SAMPLE_RATE * 2)
        self.system_prompt = self._resolve_system_prompt(
            system_prompt=system_prompt,
            bot_config_path=bot_config_path,
        )

        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._prompt_cache: dict[str, tuple[list[str], list[str]]] = {}

        self._vad_model = load_silero_vad()
        logger.debug("silero VAD model loaded")

        if warmup:
            self.warmup()
        logger.info(
            "engine ready device=%s dtype=%s speaker=%s system_prompt_chars=%s",
            self.device,
            self.model_dtype,
            self.default_speaker,
            len(self.system_prompt),
        )

    def _resolve_system_prompt(
        self,
        *,
        system_prompt: str | None,
        bot_config_path: str | None,
    ) -> str:
        if system_prompt is not None:
            prompt = system_prompt.strip()
            if not prompt:
                raise ValueError("system_prompt must not be empty")
            logger.info("Using system prompt from runtime parameter.")
            return prompt

        if bot_config_path:
            config = load_bot_config(bot_config_path)
            logger.info("Using system prompt from bot config: %s", bot_config_path)
            return config.system_prompt

        return SYSTEM_PROMPT

    def warmup(self) -> None:
        logger.debug("warmup start")
        audio = np.zeros(INPUT_SAMPLE_RATE, dtype=np.float32)
        config = SessionConfig(speaker=self.default_speaker, output_chunk_sec=0.2, text_mode="none")
        self.create_session("__warmup__", config)
        self.append_audio("__warmup__", float32_to_pcm16le_bytes(audio))

        async def _run_warmup() -> None:
            async for _ in self.commit_turn("__warmup__"):
                pass

        try:
            asyncio.run(_run_warmup())
            logger.debug("warmup generation finished")
        except Exception:
            logger.exception("Warmup failed")
        finally:
            self.close_session("__warmup__")
            logger.debug("warmup end")

    def create_session(self, session_id: str, config: SessionConfig) -> None:
        logger.debug("create_session session_id=%s requested_config=%s", session_id, config)
        config = self._normalize_config(config)
        with self._sessions_lock:
            if session_id not in self._sessions and len(self._sessions) >= self.max_sessions:
                logger.error(
                    "create_session rejected session_id=%s active_sessions=%s max_sessions=%s",
                    session_id,
                    len(self._sessions),
                    self.max_sessions,
                )
                raise ValueError(f"Max sessions limit reached: {self.max_sessions}")
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = _SessionState(config=config)
                logger.debug(
                    "session created session_id=%s total_sessions=%s config=%s",
                    session_id,
                    len(self._sessions),
                    config,
                )
                return
        with state.lock:
            state.config = config
        logger.debug("session updated via create_session session_id=%s config=%s", session_id, config)

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        logger.debug("update_session session_id=%s kwargs=%s", session_id, kwargs)
        state = self._get_session(session_id)
        with state.lock:
            config = state.config
            speaker = kwargs.get("speaker")
            memory_turns = kwargs.get("memory_turns")
            output_chunk_sec = kwargs.get("output_chunk_sec")
            text_mode = kwargs.get("text_mode")
            include_transcript_in_query = kwargs.get("include_transcript_in_query")
            system_prompt = kwargs.get("system_prompt", _UNSET)
            merged = SessionConfig(
                speaker=config.speaker if speaker is None else speaker,
                memory_turns=config.memory_turns if memory_turns is None else memory_turns,
                output_chunk_sec=config.output_chunk_sec
                if output_chunk_sec is None
                else output_chunk_sec,
                text_mode=config.text_mode if text_mode is None else text_mode,
                include_transcript_in_query=(
                    config.include_transcript_in_query
                    if include_transcript_in_query is None
                    else include_transcript_in_query
                ),
                system_prompt=(
                    config.system_prompt if system_prompt is _UNSET else system_prompt
                ),
            )
            state.config = self._normalize_config(merged)
            logger.debug("session config updated session_id=%s config=%s", session_id, state.config)

    def get_session_config(self, session_id: str) -> SessionConfig:
        state = self._get_session(session_id)
        with state.lock:
            config = state.config
            return SessionConfig(
                speaker=config.speaker,
                memory_turns=config.memory_turns,
                output_chunk_sec=config.output_chunk_sec,
                text_mode=config.text_mode,
                include_transcript_in_query=config.include_transcript_in_query,
                system_prompt=config.system_prompt,
            )

    def set_pending_user_text(self, session_id: str, text: str | None) -> None:
        state = self._get_session(session_id)
        with state.lock:
            state.pending_user_text = (text or "").strip() or None
            logger.debug(
                "set_pending_user_text session_id=%s has_text=%s length=%s",
                session_id,
                state.pending_user_text is not None,
                len(state.pending_user_text) if state.pending_user_text else 0,
            )

    def append_audio(self, session_id: str, pcm16_16k: bytes) -> None:
        state = self._get_session(session_id)
        if not isinstance(pcm16_16k, (bytes, bytearray)):
            logger.error("append_audio invalid payload type=%s", type(pcm16_16k).__name__)
            raise TypeError("pcm16_16k must be bytes")
        with state.lock:
            prev_size = len(state.audio_buffer)
            state.audio_buffer.extend(pcm16_16k)
            overflow = len(state.audio_buffer) - self.max_input_bytes
            if overflow > 0:
                del state.audio_buffer[:overflow]
                logger.warning(
                    "append_audio overflow trimmed session_id=%s overflow_bytes=%s buffer_after=%s",
                    session_id,
                    overflow,
                    len(state.audio_buffer),
                )
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "append_audio session_id=%s bytes_in=%s buffer_before=%s buffer_after=%s",
                    session_id,
                    len(pcm16_16k),
                    prev_size,
                    len(state.audio_buffer),
                )

    def get_buffered_audio(self, session_id: str) -> bytes:
        state = self._get_session(session_id)
        with state.lock:
            return bytes(state.audio_buffer)

    def cancel_response(self, session_id: str) -> None:
        logger.info("cancel_response requested session_id=%s", session_id)
        state = self._get_session(session_id)
        with state.lock:
            state.cancel_requested_at = time.perf_counter()
            if state.active_cancel_event is not None:
                state.active_cancel_event.set()
                logger.info(
                    "cancel flag set session_id=%s turn_id=%s",
                    session_id,
                    state.active_turn_id,
                )
            else:
                logger.debug("no active turn to cancel session_id=%s", session_id)

    def close_session(self, session_id: str) -> None:
        logger.debug("close_session start session_id=%s", session_id)
        with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            logger.debug("close_session skipped missing session_id=%s", session_id)
            return
        with state.lock:
            if state.active_cancel_event is not None:
                state.active_cancel_event.set()
            thread = state.active_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
            logger.debug("close_session joined active thread session_id=%s", session_id)
        logger.debug("close_session done session_id=%s", session_id)

    async def commit_turn(self, session_id: str) -> AsyncIterator[EngineEvent]:
        logger.info("commit_turn start session_id=%s", session_id)
        try:
            state = self._get_session(session_id)
        except ValueError as exc:
            logger.error("commit_turn session not found session_id=%s error=%s", session_id, exc)
            yield ErrorEvent(session_id=session_id, code="session_not_found", message=str(exc))
            return

        with state.lock:
            if state.active_cancel_event is not None:
                logger.info(
                    "commit_turn session_id=%s preempting previous active turn_id=%s",
                    session_id,
                    state.active_turn_id,
                )
                state.active_cancel_event.set()
            config = self._normalize_config(state.config)
            audio_bytes = bytes(state.audio_buffer)
            state.audio_buffer.clear()
            user_text = state.pending_user_text
            state.pending_user_text = None
            state.turn_index += 1
            turn_id = str(state.turn_index)
            cancel_event = threading.Event()
            state.active_turn_id = turn_id
            state.active_cancel_event = cancel_event
            state.active_thread = None
            state.cancel_requested_at = None
            logger.debug(
                "commit_turn prepared session_id=%s turn_id=%s audio_bytes=%s pending_user_text=%s config=%s",
                session_id,
                turn_id,
                len(audio_bytes),
                user_text is not None,
                config,
            )

        if user_text:
            self._append_memory(state, "user", user_text)

        if not audio_bytes:
            logger.warning("commit_turn empty audio session_id=%s turn_id=%s", session_id, turn_id)
            yield ErrorEvent(
                session_id=session_id,
                code="empty_audio",
                message="No audio data buffered for this turn",
            )
            self._clear_active_turn(state, turn_id)
            return

        audio_np = pcm16le_bytes_to_float32_mono(audio_bytes)
        audio_16k = _resample_audio(audio_np, INPUT_SAMPLE_RATE, INPUT_SAMPLE_RATE)
        raw_seconds = audio_16k.shape[-1] / INPUT_SAMPLE_RATE
        logger.debug(
            "commit_turn audio decoded session_id=%s turn_id=%s shape=%s raw_seconds=%.3f",
            session_id,
            turn_id,
            _shape(audio_16k),
            raw_seconds,
        )
        audio_16k = self._trim_with_vad(audio_16k)
        trimmed_seconds = audio_16k.shape[-1] / INPUT_SAMPLE_RATE
        logger.debug(
            "commit_turn vad trimmed session_id=%s turn_id=%s trimmed_seconds=%.3f",
            session_id,
            turn_id,
            trimmed_seconds,
        )
        if audio_16k.size < INPUT_SAMPLE_RATE // 10:
            logger.warning(
                "commit_turn audio too short session_id=%s turn_id=%s raw=%.3fs trimmed=%.3fs",
                session_id,
                turn_id,
                raw_seconds,
                trimmed_seconds,
            )
            yield ErrorEvent(
                session_id=session_id,
                code="audio_too_short",
                message="Audio segment too short after VAD",
                details={"raw_seconds": raw_seconds, "trimmed_seconds": trimmed_seconds},
            )
            self._clear_active_turn(state, turn_id)
            return

        try:
            inputs = self._prepare_inputs(
                audio_16k,
                config.speaker,
                state,
                session_id=session_id,
                turn_id=turn_id,
                query_text=user_text,
                include_transcript_in_query=config.include_transcript_in_query,
            )
            logger.debug(
                "commit_turn inputs ready session_id=%s turn_id=%s keys=%s",
                session_id,
                turn_id,
                sorted(inputs.keys()),
            )
        except Exception as exc:
            logger.exception(
                "commit_turn input preparation failed session_id=%s turn_id=%s",
                session_id,
                turn_id,
            )
            yield ErrorEvent(
                session_id=session_id,
                code="input_preparation_failed",
                message=str(exc),
            )
            self._clear_active_turn(state, turn_id)
            return

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[EngineEvent | None] = asyncio.Queue()

        started_at = time.perf_counter()
        frame_rate = float(getattr(self.model.config.codec_config, "frame_rate", 12.5))
        num_codebooks = int(getattr(self.model.config.decoder_config, "audio_num_codebooks", 1))
        if num_codebooks <= 0:
            num_codebooks = 1
        metrics: dict[str, float] = {
            "raw_audio_sec": raw_seconds,
            "trimmed_audio_sec": trimmed_seconds,
            "ttfs_ms": -1.0,
            "first_decode_ms": -1.0,
            "chunk_gap_ms": -1.0,
            "cancel_to_stop_ms": -1.0,
            "audio_out_sec": 0.0,
            "audio_tokens": 0.0,
            "tokens_per_sec": -1.0,
            "tokens_per_sec_post_ttfs": -1.0,
        }
        first_chunk_ts: float | None = None
        last_chunk_ts: float | None = None
        chunk_gaps: list[float] = []
        total_output_samples = 0

        def emit(event: EngineEvent) -> None:
            logger.debug(
                "emit event session_id=%s turn_id=%s event_type=%s",
                session_id,
                turn_id,
                getattr(event, "type", type(event).__name__),
            )
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        def finalize(cancelled: bool, text_output: str | None = None) -> None:
            nonlocal first_chunk_ts, last_chunk_ts
            if first_chunk_ts is not None:
                metrics["ttfs_ms"] = max(0.0, (first_chunk_ts - started_at) * 1000.0)
                metrics["first_decode_ms"] = metrics["ttfs_ms"]
            if chunk_gaps:
                metrics["chunk_gap_ms"] = sum(chunk_gaps) / len(chunk_gaps)

            elapsed_sec = max(1e-9, time.perf_counter() - started_at)
            audio_out_sec = total_output_samples / float(OUTPUT_SAMPLE_RATE)
            audio_tokens = audio_out_sec * frame_rate * float(num_codebooks)
            metrics["audio_out_sec"] = audio_out_sec
            metrics["audio_tokens"] = audio_tokens
            metrics["tokens_per_sec"] = audio_tokens / elapsed_sec
            if metrics["ttfs_ms"] >= 0.0:
                post_ttfs_sec = elapsed_sec - (metrics["ttfs_ms"] / 1000.0)
                if post_ttfs_sec > 1e-9:
                    metrics["tokens_per_sec_post_ttfs"] = audio_tokens / post_ttfs_sec

            cancel_requested_at = None
            with state.lock:
                cancel_requested_at = state.cancel_requested_at
            if cancelled and cancel_requested_at is not None:
                metrics["cancel_to_stop_ms"] = max(
                    0.0, (time.perf_counter() - cancel_requested_at) * 1000.0
                )
            logger.info(
                "response complete session_id=%s turn_id=%s cancelled=%s text=\"%s\" metrics=%s",
                session_id,
                turn_id,
                cancelled,
                _clip_text(text_output),
                metrics,
            )
            logger.info(
                "audio throughput session_id=%s turn_id=%s audio_tokens=%.2f tokens_per_sec=%.2f post_ttfs=%.2f "
                "audio_out_sec=%.3f frame_rate=%.2f codebooks=%s",
                session_id,
                turn_id,
                metrics["audio_tokens"],
                metrics["tokens_per_sec"],
                metrics["tokens_per_sec_post_ttfs"],
                metrics["audio_out_sec"],
                frame_rate,
                num_codebooks,
            )

            if cancelled:
                emit(ResponseCancelledEvent(session_id=session_id, turn_id=turn_id))
            else:
                if text_output:
                    emit(
                        ResponseTextDeltaEvent(
                            session_id=session_id,
                            turn_id=turn_id,
                            text=text_output,
                        )
                    )
                emit(
                    ResponseDoneEvent(
                        session_id=session_id,
                        turn_id=turn_id,
                        text=text_output,
                        metrics=metrics,
                    )
                )
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

        def on_audio_chunk(audio_chunk: np.ndarray) -> None:
            nonlocal first_chunk_ts, last_chunk_ts, total_output_samples
            now = time.perf_counter()
            if first_chunk_ts is None:
                first_chunk_ts = now
                logger.info(
                    "first audio chunk session_id=%s turn_id=%s after_ms=%.2f",
                    session_id,
                    turn_id,
                    (now - started_at) * 1000.0,
                )
            if last_chunk_ts is not None:
                chunk_gaps.append((now - last_chunk_ts) * 1000.0)
            last_chunk_ts = now
            total_output_samples += int(audio_chunk.shape[-1])

            pcm16 = float32_to_pcm16le_bytes(audio_chunk[0])
            audio_b64 = base64.b64encode(pcm16).decode("ascii")
            logger.debug(
                "audio chunk prepared session_id=%s turn_id=%s samples=%s encoded_bytes=%s",
                session_id,
                turn_id,
                audio_chunk.shape[-1],
                len(audio_b64),
            )
            emit(
                ResponseAudioDeltaEvent(
                    session_id=session_id,
                    turn_id=turn_id,
                    audio_b64=audio_b64,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                )
            )

        def run_generation() -> None:
            frames_per_chunk = max(
                1,
                int(round(config.output_chunk_sec * frame_rate)),
            )
            logger.debug(
                "generation thread start session_id=%s turn_id=%s frame_rate=%s frames_per_chunk=%s",
                session_id,
                turn_id,
                frame_rate,
                frames_per_chunk,
            )
            streamer = _EngineAudioStreamer(
                model=self.model,
                frames_per_chunk=frames_per_chunk,
                cancel_event=cancel_event,
                on_audio_chunk=on_audio_chunk,
                log_prefix=f"[{session_id}:{turn_id}] ",
            )
            try:
                logger.debug("model.generate begin session_id=%s turn_id=%s", session_id, turn_id)
                self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    use_cache=True,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([AbortOnCancelCriteria(cancel_event)]),
                )
                logger.debug("model.generate end session_id=%s turn_id=%s", session_id, turn_id)
                streamer.end()
                if cancel_event.is_set():
                    logger.info("generation cancelled session_id=%s turn_id=%s", session_id, turn_id)
                    finalize(cancelled=True)
                    return

                text_output = self._generate_text(inputs, config.text_mode)
                if text_output:
                    self._append_memory(state, "assistant", text_output)
                finalize(cancelled=False, text_output=text_output)
            except GenerationCancelled:
                logger.info("generation aborted by cancel event session_id=%s turn_id=%s", session_id, turn_id)
                finalize(cancelled=True)
            except Exception as exc:
                logger.exception("Generation failed")
                emit(
                    ErrorEvent(
                        session_id=session_id,
                        code="generation_failed",
                        message=str(exc),
                    )
                )
                loop.call_soon_threadsafe(event_queue.put_nowait, None)
            finally:
                self._clear_active_turn(state, turn_id)
                logger.debug("generation thread cleanup done session_id=%s turn_id=%s", session_id, turn_id)

        worker = threading.Thread(target=run_generation, daemon=True)
        with state.lock:
            state.active_thread = worker
        emit(ResponseStartedEvent(session_id=session_id, turn_id=turn_id))
        logger.info("response started session_id=%s turn_id=%s", session_id, turn_id)
        worker.start()
        logger.debug("generation thread started session_id=%s turn_id=%s", session_id, turn_id)

        while True:
            event = await event_queue.get()
            if event is None:
                logger.debug("commit_turn end-of-stream marker session_id=%s turn_id=%s", session_id, turn_id)
                break
            logger.debug(
                "commit_turn yielding event session_id=%s turn_id=%s event_type=%s",
                session_id,
                turn_id,
                getattr(event, "type", type(event).__name__),
            )
            yield event
        logger.info("commit_turn done session_id=%s turn_id=%s", session_id, turn_id)

    def commit_turn_sync(self, session_id: str) -> Iterator[EngineEvent]:
        logger.debug("commit_turn_sync start session_id=%s", session_id)
        output_queue: queue.SimpleQueue[EngineEvent | None] = queue.SimpleQueue()

        def _runner() -> None:
            async def _run() -> None:
                async for event in self.commit_turn(session_id):
                    output_queue.put(event)

            try:
                asyncio.run(_run())
            except Exception as exc:
                logger.exception("commit_turn_sync runner failed session_id=%s", session_id)
                output_queue.put(
                    ErrorEvent(
                        session_id=session_id,
                        code="sync_bridge_failed",
                        message=str(exc),
                    )
                )
            finally:
                output_queue.put(None)
                logger.debug("commit_turn_sync runner finished session_id=%s", session_id)

        threading.Thread(target=_runner, daemon=True).start()

        while True:
            item = output_queue.get()
            if item is None:
                break
            yield item
        logger.debug("commit_turn_sync end session_id=%s", session_id)

    def _trim_with_vad(self, audio_16k: np.ndarray) -> np.ndarray:
        logger.debug("vad trim start audio_shape=%s", _shape(audio_16k))
        audio_tensor = torch.from_numpy(audio_16k)
        speech_timestamps = get_speech_timestamps(
            audio_tensor, self._vad_model, sampling_rate=INPUT_SAMPLE_RATE
        )
        if not speech_timestamps:
            logger.debug("vad found no speech; keep original audio")
            return audio_16k
        start = speech_timestamps[0]["start"]
        end = speech_timestamps[-1]["end"]
        trimmed = audio_tensor[start:end]
        if trimmed.numel() == 0:
            logger.warning("vad produced empty trim; keep original audio")
            return audio_16k
        logger.debug(
            "vad trim result start=%s end=%s kept_samples=%s",
            start,
            end,
            trimmed.numel(),
        )
        return trimmed.cpu().numpy()

    def _load_prompt(self, speaker: str) -> tuple[list[str], list[str]]:
        speaker = speaker if speaker in PROMPT_SPEAKERS else self.default_speaker
        if speaker in self._prompt_cache:
            logger.debug("prompt cache hit speaker=%s", speaker)
            return self._prompt_cache[speaker]
        repo_root = Path(__file__).resolve().parents[2]
        text_path = repo_root / "example" / "prompt_text" / f"{speaker}.txt"
        audio_path = repo_root / "example" / "prompt_audio" / f"{speaker}.wav"
        logger.debug("loading prompt speaker=%s text=%s audio=%s", speaker, text_path, audio_path)
        prompt_text = text_path.read_text(encoding="utf-8")
        payload = ([prompt_text], [str(audio_path)])
        self._prompt_cache[speaker] = payload
        return payload

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                logger.debug("move tensor float to device shape=%s dtype=%s", _shape(value), self.model_dtype)
                return value.to(device=self.device, dtype=self.model_dtype)
            logger.debug("move tensor int/bool to device shape=%s", _shape(value))
            return value.to(device=self.device)
        return value

    def _memory_context(self, state: _SessionState) -> str:
        if not state.memory or state.config.memory_turns <= 0:
            logger.debug("memory context empty")
            return ""
        max_entries = max(1, state.config.memory_turns * 2)
        clipped = state.memory[-max_entries:]
        logger.debug("memory context build entries=%s max_entries=%s", len(clipped), max_entries)
        lines = ["Recent conversation summary:"]
        for item in clipped:
            role = "User" if item.role == "user" else "Assistant"
            lines.append(f"- {role}: {item.text}")
        return "\n".join(lines)

    def _prepare_inputs(
        self,
        audio_np: np.ndarray,
        speaker: str,
        state: _SessionState,
        session_id: str,
        turn_id: str,
        query_text: str | None,
        include_transcript_in_query: bool,
    ) -> dict[str, torch.Tensor]:
        logger.debug("prepare_inputs start speaker=%s audio_shape=%s", speaker, _shape(audio_np))
        prompt_text, prompt_audio = self._load_prompt(speaker)
        memory_context = self._memory_context(state)
        self._log_input_context(
            session_id=session_id,
            turn_id=turn_id,
            memory_context=memory_context,
            query_text=query_text,
            include_transcript_in_query=include_transcript_in_query,
        )
        active_prompt = state.config.system_prompt or self.system_prompt
        system_text = active_prompt
        if memory_context:
            system_text = f"{active_prompt}\n\n{memory_context}"
        user_content: list[dict[str, Any]] = []
        if include_transcript_in_query and query_text:
            user_content.append({"type": "text", "text": query_text})
        user_content.append({"type": "audio", "audio": audio_np})
        conversation = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_text}],
                },
                {"role": "user", "content": user_content},
            ]
        ]
        inputs = self.processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )
        moved = {k: self._move_to_device(v) for k, v in inputs.items()}
        logger.debug(
            "prepare_inputs done keys=%s shapes=%s",
            sorted(moved.keys()),
            {k: _shape(v) for k, v in moved.items()},
        )
        return moved

    def _log_input_context(
        self,
        session_id: str,
        turn_id: str,
        memory_context: str,
        query_text: str | None,
        include_transcript_in_query: bool,
    ) -> None:
        if include_transcript_in_query and query_text:
            query_preview = _clip_text(query_text, max_chars=220)
            query_mode = "audio+transcript"
        else:
            query_preview = "(audio-only; no transcript)"
            query_mode = "audio-only"
        history_preview = (
            _clip_text(memory_context.replace("\n", " | "), max_chars=460)
            if memory_context
            else "(empty memory context)"
        )
        logger.info(
            "input context session_id=%s turn_id=%s %s",
            session_id,
            turn_id,
            _colorize(f"QUERY   [{query_mode}] {query_preview}", _ANSI_QUERY),
        )
        logger.info(
            "input context session_id=%s turn_id=%s %s",
            session_id,
            turn_id,
            _colorize(f"HISTORY {history_preview}", _ANSI_HISTORY),
        )

    @torch.no_grad()
    def _generate_text(self, inputs: dict[str, torch.Tensor], text_mode: str) -> str | None:
        logger.debug("generate_text start enabled=%s text_mode=%s", self.enable_text, text_mode)
        if not self.enable_text or text_mode == "none":
            logger.debug("generate_text skipped by config")
            return None
        thinker_input_ids = inputs.get("thinker_input_ids")
        if thinker_input_ids is None:
            logger.debug("generate_text skipped no thinker_input_ids")
            return None

        output_ids = self.model.thinker.generate(
            input_ids=thinker_input_ids,
            attention_mask=inputs.get("thinker_attention_mask"),
            input_features=inputs.get("thinker_input_features"),
            feature_attention_mask=inputs.get("thinker_feature_attention_mask"),
            max_new_tokens=self.max_text_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            use_cache=True,
        )
        prompt_len = thinker_input_ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        if generated_ids.numel() == 0:
            logger.debug("generate_text empty generated ids")
            return None
        text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            logger.debug("generate_text decoded empty text")
            return None
        if text_mode == "final":
            logger.debug("generate_text final mode length=%s", len(text))
            return text
        # sentence mode: keep first complete sentence if possible
        for delimiter in [". ", "! ", "? ", "。", "！", "？"]:
            if delimiter in text:
                sentence = text.split(delimiter, 1)[0].strip() + delimiter.strip()
                logger.debug("generate_text sentence mode length=%s", len(sentence))
                return sentence
        logger.debug("generate_text sentence fallback full length=%s", len(text))
        return text

    def _append_memory(self, state: _SessionState, role: str, text: str) -> None:
        if not text:
            return
        with state.lock:
            if state.config.memory_turns <= 0:
                state.memory.clear()
                logger.debug("append_memory skipped because memory_turns<=0")
                return
            state.memory.append(_MemoryItem(role=role, text=text))
            max_entries = max(1, state.config.memory_turns * 2)
            if len(state.memory) > max_entries:
                del state.memory[:-max_entries]
            logger.debug(
                "append_memory role=%s text_len=%s memory_size=%s max_entries=%s",
                role,
                len(text),
                len(state.memory),
                max_entries,
            )

    def _clear_active_turn(self, state: _SessionState, turn_id: str) -> None:
        with state.lock:
            if state.active_turn_id == turn_id:
                state.active_turn_id = None
                state.active_cancel_event = None
                state.active_thread = None
                logger.debug("cleared active turn turn_id=%s", turn_id)

    def _normalize_config(self, config: SessionConfig) -> SessionConfig:
        logger.debug("normalize_config input=%s", config)
        speaker = config.speaker
        if speaker not in PROMPT_SPEAKERS:
            logger.warning(
                "Invalid speaker '%s'; fallback to '%s'", speaker, self.default_speaker
            )
            speaker = self.default_speaker

        memory_turns = max(0, int(config.memory_turns))
        output_chunk_sec = float(config.output_chunk_sec)
        if output_chunk_sec <= 0:
            output_chunk_sec = self.default_output_chunk_sec

        text_mode = config.text_mode
        if text_mode not in {"none", "sentence", "final"}:
            text_mode = "sentence"

        include_transcript_in_query = self._coerce_bool(
            config.include_transcript_in_query,
            default=False,
        )
        system_prompt = self._normalize_session_prompt(config.system_prompt)

        normalized = SessionConfig(
            speaker=speaker,
            memory_turns=memory_turns,
            output_chunk_sec=output_chunk_sec,
            text_mode=text_mode,
            include_transcript_in_query=include_transcript_in_query,
            system_prompt=system_prompt,
        )
        logger.debug("normalize_config output=%s", normalized)
        return normalized

    @staticmethod
    def _normalize_session_prompt(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("system_prompt must be string or null")
        normalized = value.strip()
        if not normalized:
            raise ValueError("system_prompt must not be empty")
        if len(normalized) > 4000:
            raise ValueError("system_prompt exceeds 4000 characters")
        return normalized

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off", ""}:
                return False
        return default

    def _get_session(self, session_id: str) -> _SessionState:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            logger.error("session lookup failed session_id=%s", session_id)
            raise ValueError(f"Session '{session_id}' not found")
        logger.debug("session lookup hit session_id=%s", session_id)
        return state

    def _resolve_local_model_path(self, local_path: str) -> str:
        logger.debug("resolve_local_model_path input=%s", local_path)
        path = Path(local_path)
        if (path / "config.json").is_file():
            logger.debug("resolve_local_model_path direct config.json found")
            return str(path)

        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            ref_path = path / "refs" / "main"
            if ref_path.is_file():
                snapshot = snapshots_dir / ref_path.read_text().strip()
                if (snapshot / "config.json").is_file():
                    logger.debug("resolve_local_model_path resolved via refs/main")
                    return str(snapshot)

            snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if len(snapshot_dirs) == 1 and (snapshot_dirs[0] / "config.json").is_file():
                logger.debug("resolve_local_model_path resolved via single snapshot")
                return str(snapshot_dirs[0])

            raise ValueError(
                "Local model path looks like a Hugging Face cache; pass the snapshot "
                "directory (e.g. .../snapshots/<hash>)."
            )

        logger.debug("resolve_local_model_path fallback=%s", path)
        return str(path)

    def _load_chroma_model(
        self,
        from_local_path: str | None,
        *,
        use_half_precision: bool,
    ):
        logger.debug(
            "load_chroma_model start from_local_path=%s use_half_precision=%s",
            from_local_path,
            use_half_precision,
        )
        model_id = (
            self._resolve_local_model_path(from_local_path)
            if from_local_path
            else "FlashLabs/Chroma-4B"
        )

        cache_dir = None
        if not from_local_path:
            repo_root = Path(__file__).resolve().parents[2]
            cache_dir = repo_root / "pretrained_models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("load_chroma_model cache_dir=%s", cache_dir)

        torch_dtype = torch.float32
        if use_half_precision:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.debug("load_chroma_model using fp16")
            else:
                logger.warning(
                    "Half precision requested but CUDA is unavailable; using fp32."
                )

        logger.info("load_chroma_model loading model_id=%s dtype=%s", model_id, torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=str(cache_dir) if cache_dir else None,
            torch_dtype=torch_dtype,
        ).eval()
        logger.info("model loaded device=%s dtype=%s", model.device, model.dtype)

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        logger.info("processor loaded")

        return model, processor


def pcm16le_bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return (audio.astype(np.float32) / 32768.0).copy()


def float32_to_pcm16le_bytes(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32, copy=False)


def _resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    audio = _normalize_audio(audio)
    if sample_rate == target_sample_rate:
        return audio
    audio_tensor = torch.from_numpy(audio)
    resampled = torchaudio.functional.resample(
        audio_tensor,
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
    )
    return resampled.cpu().numpy()


__all__ = [
    "AbortOnCancelCriteria",
    "GenerationCancelled",
    "INPUT_SAMPLE_RATE",
    "OUTPUT_SAMPLE_RATE",
    "PROMPT_SPEAKERS",
    "SYSTEM_PROMPT",
    "StreamingVoicebotEngine",
    "float32_to_pcm16le_bytes",
    "pcm16le_bytes_to_float32_mono",
]
