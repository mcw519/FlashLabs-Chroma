from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

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
    ):
        self.model = model
        self.frames_per_chunk = frames_per_chunk
        self.cancel_event = cancel_event
        self.on_audio_chunk = on_audio_chunk
        self.eos_token_id = model.config.codebook_eos_token_id
        self.num_codebooks = model.config.decoder_config.audio_num_codebooks
        self._buffer: list[torch.Tensor] = []
        self._closed = False

    def put(self, value) -> None:
        if self.cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if self._closed:
            return
        tokens = torch.as_tensor(value)
        if tokens.ndim == 2:
            if tokens.shape[0] == 1 and tokens.shape[1] == self.num_codebooks:
                tokens = tokens[0]
            elif tokens.shape[1] == self.num_codebooks:
                for row in tokens:
                    self._put_frame(row)
                return
            else:
                return
        if tokens.ndim != 1:
            return
        if tokens.numel() != self.num_codebooks:
            return
        self._put_frame(tokens)

    def _put_frame(self, tokens: torch.Tensor) -> None:
        if self.cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if (tokens == self.eos_token_id).all():
            self.end()
            return
        self._buffer.append(tokens)
        if len(self._buffer) >= self.frames_per_chunk:
            self._emit_frames(self.frames_per_chunk)

    def end(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._buffer:
            self._emit_frames(len(self._buffer))
            self._buffer.clear()

    def _emit_frames(self, count: int) -> None:
        frames = self._buffer[:count]
        del self._buffer[:count]
        audio_np = self._decode_frames(frames)
        if audio_np.size == 0:
            return
        self.on_audio_chunk(audio_np)

    @torch.no_grad()
    def _decode_frames(self, frames: list[torch.Tensor]) -> np.ndarray:
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
    ) -> None:
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

        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._prompt_cache: dict[str, tuple[list[str], list[str]]] = {}

        self._vad_model = load_silero_vad()

        if warmup:
            self.warmup()

    def warmup(self) -> None:
        audio = np.zeros(INPUT_SAMPLE_RATE, dtype=np.float32)
        config = SessionConfig(speaker=self.default_speaker, output_chunk_sec=0.2, text_mode="none")
        self.create_session("__warmup__", config)
        self.append_audio("__warmup__", float32_to_pcm16le_bytes(audio))

        async def _run_warmup() -> None:
            async for _ in self.commit_turn("__warmup__"):
                pass

        try:
            asyncio.run(_run_warmup())
        except Exception:
            logger.exception("Warmup failed")
        finally:
            self.close_session("__warmup__")

    def create_session(self, session_id: str, config: SessionConfig) -> None:
        config = self._normalize_config(config)
        with self._sessions_lock:
            if session_id not in self._sessions and len(self._sessions) >= self.max_sessions:
                raise ValueError(f"Max sessions limit reached: {self.max_sessions}")
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = _SessionState(config=config)
                return
        with state.lock:
            state.config = config

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        state = self._get_session(session_id)
        with state.lock:
            config = state.config
            speaker = kwargs.get("speaker")
            memory_turns = kwargs.get("memory_turns")
            output_chunk_sec = kwargs.get("output_chunk_sec")
            text_mode = kwargs.get("text_mode")
            merged = SessionConfig(
                speaker=config.speaker if speaker is None else speaker,
                memory_turns=config.memory_turns if memory_turns is None else memory_turns,
                output_chunk_sec=config.output_chunk_sec
                if output_chunk_sec is None
                else output_chunk_sec,
                text_mode=config.text_mode if text_mode is None else text_mode,
            )
            state.config = self._normalize_config(merged)

    def set_pending_user_text(self, session_id: str, text: str | None) -> None:
        state = self._get_session(session_id)
        with state.lock:
            state.pending_user_text = (text or "").strip() or None

    def append_audio(self, session_id: str, pcm16_16k: bytes) -> None:
        state = self._get_session(session_id)
        if not isinstance(pcm16_16k, (bytes, bytearray)):
            raise TypeError("pcm16_16k must be bytes")
        with state.lock:
            state.audio_buffer.extend(pcm16_16k)
            overflow = len(state.audio_buffer) - self.max_input_bytes
            if overflow > 0:
                del state.audio_buffer[:overflow]

    def cancel_response(self, session_id: str) -> None:
        state = self._get_session(session_id)
        with state.lock:
            state.cancel_requested_at = time.perf_counter()
            if state.active_cancel_event is not None:
                state.active_cancel_event.set()

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return
        with state.lock:
            if state.active_cancel_event is not None:
                state.active_cancel_event.set()
            thread = state.active_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    async def commit_turn(self, session_id: str) -> AsyncIterator[EngineEvent]:
        try:
            state = self._get_session(session_id)
        except ValueError as exc:
            yield ErrorEvent(session_id=session_id, code="session_not_found", message=str(exc))
            return

        with state.lock:
            if state.active_cancel_event is not None:
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

        if user_text:
            self._append_memory(state, "user", user_text)

        if not audio_bytes:
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
        audio_16k = self._trim_with_vad(audio_16k)
        trimmed_seconds = audio_16k.shape[-1] / INPUT_SAMPLE_RATE
        if audio_16k.size < INPUT_SAMPLE_RATE // 10:
            yield ErrorEvent(
                session_id=session_id,
                code="audio_too_short",
                message="Audio segment too short after VAD",
                details={"raw_seconds": raw_seconds, "trimmed_seconds": trimmed_seconds},
            )
            self._clear_active_turn(state, turn_id)
            return

        try:
            inputs = self._prepare_inputs(audio_16k, config.speaker, state)
        except Exception as exc:
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
        metrics: dict[str, float] = {
            "raw_audio_sec": raw_seconds,
            "trimmed_audio_sec": trimmed_seconds,
            "ttfs_ms": -1.0,
            "first_decode_ms": -1.0,
            "chunk_gap_ms": -1.0,
            "cancel_to_stop_ms": -1.0,
        }
        first_chunk_ts: float | None = None
        last_chunk_ts: float | None = None
        chunk_gaps: list[float] = []

        def emit(event: EngineEvent) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        def finalize(cancelled: bool, text_output: str | None = None) -> None:
            nonlocal first_chunk_ts, last_chunk_ts
            if first_chunk_ts is not None:
                metrics["ttfs_ms"] = max(0.0, (first_chunk_ts - started_at) * 1000.0)
                metrics["first_decode_ms"] = metrics["ttfs_ms"]
            if chunk_gaps:
                metrics["chunk_gap_ms"] = sum(chunk_gaps) / len(chunk_gaps)
            cancel_requested_at = None
            with state.lock:
                cancel_requested_at = state.cancel_requested_at
            if cancelled and cancel_requested_at is not None:
                metrics["cancel_to_stop_ms"] = max(
                    0.0, (time.perf_counter() - cancel_requested_at) * 1000.0
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
            nonlocal first_chunk_ts, last_chunk_ts
            now = time.perf_counter()
            if first_chunk_ts is None:
                first_chunk_ts = now
            if last_chunk_ts is not None:
                chunk_gaps.append((now - last_chunk_ts) * 1000.0)
            last_chunk_ts = now

            pcm16 = float32_to_pcm16le_bytes(audio_chunk[0])
            audio_b64 = base64.b64encode(pcm16).decode("ascii")
            emit(
                ResponseAudioDeltaEvent(
                    session_id=session_id,
                    turn_id=turn_id,
                    audio_b64=audio_b64,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                )
            )

        def run_generation() -> None:
            streamer = _EngineAudioStreamer(
                model=self.model,
                frames_per_chunk=max(
                    1,
                    int(
                        round(
                            config.output_chunk_sec
                            * float(getattr(self.model.config.codec_config, "frame_rate", 12.5))
                        )
                    ),
                ),
                cancel_event=cancel_event,
                on_audio_chunk=on_audio_chunk,
            )
            try:
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
                streamer.end()
                if cancel_event.is_set():
                    finalize(cancelled=True)
                    return

                text_output = self._generate_text(inputs, config.text_mode)
                if text_output:
                    self._append_memory(state, "assistant", text_output)
                finalize(cancelled=False, text_output=text_output)
            except GenerationCancelled:
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

        worker = threading.Thread(target=run_generation, daemon=True)
        with state.lock:
            state.active_thread = worker
        emit(ResponseStartedEvent(session_id=session_id, turn_id=turn_id))
        worker.start()

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

    def commit_turn_sync(self, session_id: str) -> Iterator[EngineEvent]:
        output_queue: queue.SimpleQueue[EngineEvent | None] = queue.SimpleQueue()

        def _runner() -> None:
            async def _run() -> None:
                async for event in self.commit_turn(session_id):
                    output_queue.put(event)

            try:
                asyncio.run(_run())
            except Exception as exc:
                output_queue.put(
                    ErrorEvent(
                        session_id=session_id,
                        code="sync_bridge_failed",
                        message=str(exc),
                    )
                )
            finally:
                output_queue.put(None)

        threading.Thread(target=_runner, daemon=True).start()

        while True:
            item = output_queue.get()
            if item is None:
                break
            yield item

    def _trim_with_vad(self, audio_16k: np.ndarray) -> np.ndarray:
        audio_tensor = torch.from_numpy(audio_16k)
        speech_timestamps = get_speech_timestamps(
            audio_tensor, self._vad_model, sampling_rate=INPUT_SAMPLE_RATE
        )
        if not speech_timestamps:
            return audio_16k
        start = speech_timestamps[0]["start"]
        end = speech_timestamps[-1]["end"]
        trimmed = audio_tensor[start:end]
        if trimmed.numel() == 0:
            return audio_16k
        return trimmed.cpu().numpy()

    def _load_prompt(self, speaker: str) -> tuple[list[str], list[str]]:
        speaker = speaker if speaker in PROMPT_SPEAKERS else self.default_speaker
        if speaker in self._prompt_cache:
            return self._prompt_cache[speaker]
        repo_root = Path(__file__).resolve().parents[2]
        text_path = repo_root / "example" / "prompt_text" / f"{speaker}.txt"
        audio_path = repo_root / "example" / "prompt_audio" / f"{speaker}.wav"
        prompt_text = text_path.read_text(encoding="utf-8")
        payload = ([prompt_text], [str(audio_path)])
        self._prompt_cache[speaker] = payload
        return payload

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(device=self.device, dtype=self.model_dtype)
            return value.to(device=self.device)
        return value

    def _memory_context(self, state: _SessionState) -> str:
        if not state.memory or state.config.memory_turns <= 0:
            return ""
        max_entries = max(1, state.config.memory_turns * 2)
        clipped = state.memory[-max_entries:]
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
    ) -> dict[str, torch.Tensor]:
        prompt_text, prompt_audio = self._load_prompt(speaker)
        memory_context = self._memory_context(state)
        system_text = SYSTEM_PROMPT
        if memory_context:
            system_text = f"{SYSTEM_PROMPT}\n\n{memory_context}"
        conversation = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_text}],
                },
                {"role": "user", "content": [{"type": "audio", "audio": audio_np}]},
            ]
        ]
        inputs = self.processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )
        return {k: self._move_to_device(v) for k, v in inputs.items()}

    @torch.no_grad()
    def _generate_text(self, inputs: dict[str, torch.Tensor], text_mode: str) -> str | None:
        if not self.enable_text or text_mode == "none":
            return None
        thinker_input_ids = inputs.get("thinker_input_ids")
        if thinker_input_ids is None:
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
            return None
        text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            return None
        if text_mode == "final":
            return text
        # sentence mode: keep first complete sentence if possible
        for delimiter in [". ", "! ", "? ", "。", "！", "？"]:
            if delimiter in text:
                return text.split(delimiter, 1)[0].strip() + delimiter.strip()
        return text

    def _append_memory(self, state: _SessionState, role: str, text: str) -> None:
        if not text:
            return
        with state.lock:
            if state.config.memory_turns <= 0:
                state.memory.clear()
                return
            state.memory.append(_MemoryItem(role=role, text=text))
            max_entries = max(1, state.config.memory_turns * 2)
            if len(state.memory) > max_entries:
                del state.memory[:-max_entries]

    def _clear_active_turn(self, state: _SessionState, turn_id: str) -> None:
        with state.lock:
            if state.active_turn_id == turn_id:
                state.active_turn_id = None
                state.active_cancel_event = None
                state.active_thread = None

    def _normalize_config(self, config: SessionConfig) -> SessionConfig:
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

        return SessionConfig(
            speaker=speaker,
            memory_turns=memory_turns,
            output_chunk_sec=output_chunk_sec,
            text_mode=text_mode,
        )

    def _get_session(self, session_id: str) -> _SessionState:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session '{session_id}' not found")
        return state

    def _resolve_local_model_path(self, local_path: str) -> str:
        path = Path(local_path)
        if (path / "config.json").is_file():
            return str(path)

        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            ref_path = path / "refs" / "main"
            if ref_path.is_file():
                snapshot = snapshots_dir / ref_path.read_text().strip()
                if (snapshot / "config.json").is_file():
                    return str(snapshot)

            snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if len(snapshot_dirs) == 1 and (snapshot_dirs[0] / "config.json").is_file():
                return str(snapshot_dirs[0])

            raise ValueError(
                "Local model path looks like a Hugging Face cache; pass the snapshot "
                "directory (e.g. .../snapshots/<hash>)."
            )

        return str(path)

    def _load_chroma_model(
        self,
        from_local_path: str | None,
        *,
        use_half_precision: bool,
    ):
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

        torch_dtype = torch.float32
        if use_half_precision:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                logger.warning(
                    "Half precision requested but CUDA is unavailable; using fp32."
                )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=str(cache_dir) if cache_dir else None,
            torch_dtype=torch_dtype,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

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
