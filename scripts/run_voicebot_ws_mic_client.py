# run: python scripts/run_voicebot_ws_mic_client.py --url ws://127.0.0.1:8765

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sounddevice. Install with `python -m pip install sounddevice`."
    ) from exc

try:
    import websockets
    from websockets.exceptions import ConnectionClosedOK
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: websockets. Install with `python -m pip install websockets`."
    ) from exc

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
MAX_CROSSFADE_MS = 10.0
RESET = "\033[0m"
COLOR_CODES = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "white": "\033[37m",
}


class _ComponentFilter(logging.Filter):
    def __init__(self, component: str):
        super().__init__()
        self._component = component

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "component"):
            record.component = self._component
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, component: str):
        super().__init__()
        self._component = component

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", self._component),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


@dataclass
class TurnState:
    generating: bool = False
    in_turn: bool = False
    voice_ms: float = 0.0
    silence_ms: float = 0.0
    barge_in_sent: bool = False


def _is_barge_in_enabled(mode: str, has_local_playback: bool) -> bool:
    normalized = (mode or "auto").strip().lower()
    if normalized == "off":
        return False
    if normalized == "force":
        return True
    return not has_local_playback


class _PlaybackBuffer:
    def __init__(
        self,
        *,
        sample_rate: int,
        block_samples: int,
        jitter_buffer_ms: int,
        crossfade_ms: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = max(1, int(block_samples))
        self.jitter_target_samples = max(
            self.block_samples,
            int(round(sample_rate * (max(0, jitter_buffer_ms) / 1000.0))),
        )
        self.crossfade_samples = max(
            0,
            int(
                round(
                    sample_rate
                    * (max(0.0, min(crossfade_ms, MAX_CROSSFADE_MS)) / 1000.0)
                )
            ),
        )
        self._pending_chunk = np.zeros(0, dtype=np.float32)
        self._ready_chunks: deque[np.ndarray] = deque()
        self._ready_samples = 0
        self._primed = False
        self._last_chunk_monotonic = time.monotonic()

    @property
    def ready_samples(self) -> int:
        return self._ready_samples

    def push_pcm16_chunk(self, chunk_bytes: bytes) -> None:
        audio = _pcm16le_bytes_to_float32_mono(chunk_bytes)
        if audio.size == 0:
            return
        self._last_chunk_monotonic = time.monotonic()
        if self._pending_chunk.size == 0:
            self._pending_chunk = audio
            return

        emitted, pending = self._stitch_crossfade(self._pending_chunk, audio)
        self._enqueue(emitted)
        self._pending_chunk = pending

    def flush_boundary(self) -> None:
        if self._pending_chunk.size == 0:
            return
        self._enqueue(self._pending_chunk)
        self._pending_chunk = np.zeros(0, dtype=np.float32)

    def flush_if_idle(self, idle_sec: float) -> None:
        if self._pending_chunk.size == 0:
            return
        if time.monotonic() - self._last_chunk_monotonic >= idle_sec:
            self.flush_boundary()

    def pop_block_pcm16(self) -> bytes:
        if not self._primed:
            if self._ready_samples >= self.jitter_target_samples:
                self._primed = True
            else:
                return b"\x00" * (self.block_samples * 2)

        if self._ready_samples < self.block_samples:
            self._primed = False
            return b"\x00" * (self.block_samples * 2)

        audio = self._take_samples(self.block_samples)
        return _float32_to_pcm16le_bytes(audio)

    def _enqueue(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        chunk = audio.astype(np.float32, copy=False)
        self._ready_chunks.append(chunk)
        self._ready_samples += int(chunk.size)

    def _take_samples(self, sample_count: int) -> np.ndarray:
        output = np.empty(sample_count, dtype=np.float32)
        filled = 0

        while filled < sample_count and self._ready_chunks:
            head = self._ready_chunks[0]
            take = min(sample_count - filled, int(head.size))
            output[filled : filled + take] = head[:take]
            filled += take
            self._ready_samples -= take
            if take == int(head.size):
                self._ready_chunks.popleft()
            else:
                self._ready_chunks[0] = head[take:]

        if filled < sample_count:
            output[filled:] = 0.0
        return output

    def _stitch_crossfade(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        overlap = min(self.crossfade_samples, int(prev.size), int(curr.size))
        if overlap <= 0:
            return prev, curr

        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        fade_out = 1.0 - fade_in
        blended = (prev[-overlap:] * fade_out) + (curr[:overlap] * fade_in)
        emitted = np.concatenate((prev[:-overlap], blended)).astype(
            np.float32, copy=False
        )
        pending = curr[overlap:]
        return emitted, pending


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rms_from_pcm16le(chunk_bytes: bytes) -> float:
    if not chunk_bytes:
        return 0.0
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return 0.0
    audio = audio / 32768.0
    return float(math.sqrt(float(np.mean(audio * audio))))


def _pcm16le_bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    if len(audio_bytes) % 2 == 1:
        audio_bytes = audio_bytes[:-1]
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _float32_to_pcm16le_bytes(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


async def _send_json(ws, payload: dict[str, Any], send_lock: asyncio.Lock) -> None:
    async with send_lock:
        await ws.send(json.dumps(payload, ensure_ascii=True))


def _resolve_device(device: str | None) -> str | int | None:
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    return device


def _normalize_ws_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("ws://") or normalized.startswith("wss://"):
        return normalized
    return f"ws://{normalized}"


def _get_mic_chunk_with_timeout(
    mic_queue: queue.Queue[bytes], timeout_sec: float
) -> bytes | None:
    try:
        return mic_queue.get(timeout=timeout_sec)
    except queue.Empty:
        return None


def _should_use_color(no_color: bool) -> bool:
    if no_color:
        return False
    if os.getenv("CHROMA_LOG_FORMAT", "plain").strip().lower() == "json":
        return False
    if os.getenv("NO_COLOR") is not None:
        return False
    return sys.stderr.isatty()


def _paint(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    code = COLOR_CODES.get(color)
    if code is None:
        return text
    return f"{code}{text}{RESET}"


def _tag(name: str, color: str, enabled: bool) -> str:
    return _paint(f"[{name}]", color, enabled)


def _configure_logging(level_name: str) -> None:
    default_level = "INFO"
    env_level_key = "CHROMA_CLIENT_LOG_LEVEL"
    raw_level = (level_name or os.getenv(env_level_key, default_level)).upper()
    resolved_level = getattr(logging, raw_level, logging.INFO)

    log_format = os.getenv("CHROMA_LOG_FORMAT", "plain").strip().lower()
    if log_format == "json":
        formatter: logging.Formatter = _JsonFormatter("ws_client")
    else:
        formatter = logging.Formatter("%(levelname)s | %(component)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(_ComponentFilter("ws_client"))
    root.addHandler(handler)
    root.setLevel(resolved_level)


def _prompt_device_choice(
    *,
    title: str,
    devices: list[tuple[int, dict[str, Any]]],
    current: str | None,
    use_color: bool,
    allow_disable: bool = False,
) -> tuple[str | None, bool]:
    print()
    print(_paint(title, "cyan", use_color))
    for idx, dev in devices:
        name = dev.get("name", f"device-{idx}")
        hostapi = dev.get("hostapi")
        print(f"  [{idx}] {name} (hostapi={hostapi})")
    hint = "Enter index, Enter for default"
    if allow_disable:
        hint += ", or 'none' to disable playback"
    if current is not None:
        hint += f" (current={current})"

    valid_indices = {idx for idx, _ in devices}
    while True:
        raw = input(f"{hint}: ").strip()
        if not raw:
            return current, False
        lowered = raw.lower()
        if allow_disable and lowered in {"none", "n", "off", "disable"}:
            return None, True
        if raw.isdigit() and int(raw) in valid_indices:
            return raw, False
        print(_paint("Invalid selection, please try again.", "yellow", use_color))


def _interactive_device_setup(args: argparse.Namespace, use_color: bool) -> None:
    if args.no_device_prompt:
        return
    if not sys.stdin.isatty():
        logging.info("Non-interactive shell detected; skip device prompt")
        return
    try:
        all_devices = list(sd.query_devices())
    except Exception as exc:
        logging.warning("Unable to query audio devices: %s", exc)
        return

    input_devices = [
        (idx, dev)
        for idx, dev in enumerate(all_devices)
        if int(dev.get("max_input_channels", 0)) > 0
    ]
    output_devices = [
        (idx, dev)
        for idx, dev in enumerate(all_devices)
        if int(dev.get("max_output_channels", 0)) > 0
    ]

    print(_paint("Audio device setup", "blue", use_color))
    if input_devices:
        args.input_device, _ = _prompt_device_choice(
            title="Select microphone device",
            devices=input_devices,
            current=args.input_device,
            use_color=use_color,
            allow_disable=False,
        )
    else:
        print(
            _paint("No input devices found; use system default.", "yellow", use_color)
        )

    if output_devices:
        output_choice, disable = _prompt_device_choice(
            title="Select speaker device",
            devices=output_devices,
            current=args.output_device,
            use_color=use_color,
            allow_disable=True,
        )
        args.disable_playback = disable
        args.output_device = output_choice
    else:
        print(
            _paint("No output devices found; playback disabled.", "yellow", use_color)
        )
        args.disable_playback = True


async def run_client(args: argparse.Namespace) -> None:
    use_color = _should_use_color(args.no_color)
    session_id = args.session_id or f"mic-{uuid.uuid4().hex[:8]}"
    ws_url = _normalize_ws_url(args.url)
    client_chunk_ms = float(args.client_chunk_ms)
    chunk_frames = max(1, int(INPUT_SAMPLE_RATE * (client_chunk_ms / 1000.0)))
    output_chunk_frames = max(1, int(OUTPUT_SAMPLE_RATE * (client_chunk_ms / 1000.0)))
    pre_roll_chunks = max(1, int(args.client_pre_roll_ms / client_chunk_ms))
    jitter_buffer_ms = max(0, int(args.jitter_buffer_ms))
    crossfade_ms = float(args.crossfade_ms)
    if crossfade_ms < 0.0 or crossfade_ms > MAX_CROSSFADE_MS:
        logging.warning(
            "%s crossfade-ms=%.2f out of range, clamping to [0, %.1f]",
            _tag("AUDIO", "yellow", use_color),
            crossfade_ms,
            MAX_CROSSFADE_MS,
        )
    crossfade_ms = max(0.0, min(crossfade_ms, MAX_CROSSFADE_MS))

    logging.info(
        "%s Connecting to %s session_id=%s",
        _tag("CONNECT", "blue", use_color),
        ws_url,
        session_id,
    )

    mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=512)
    stop_event = asyncio.Event()
    send_lock = asyncio.Lock()
    state = TurnState()

    playback_queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
    io_stop = threading.Event()
    session_end_sent = False

    def mic_reader_loop(stream: sd.RawInputStream) -> None:
        while not io_stop.is_set():
            chunk, overflowed = stream.read(chunk_frames)
            if overflowed:
                logging.warning(
                    "%s Input overflow detected", _tag("AUDIO", "yellow", use_color)
                )
            try:
                mic_queue.put_nowait(bytes(chunk))
            except queue.Full:
                logging.warning(
                    "%s Mic queue full; dropping chunk",
                    _tag("AUDIO", "yellow", use_color),
                )

    def playback_writer_loop(stream: sd.RawOutputStream) -> None:
        playback_buffer = _PlaybackBuffer(
            sample_rate=OUTPUT_SAMPLE_RATE,
            block_samples=output_chunk_frames,
            jitter_buffer_ms=jitter_buffer_ms,
            crossfade_ms=crossfade_ms,
        )
        while not io_stop.is_set():
            deadline = time.monotonic() + 0.02
            while time.monotonic() < deadline:
                try:
                    payload = playback_queue.get_nowait()
                except queue.Empty:
                    break
                if payload is None:
                    playback_buffer.flush_boundary()
                else:
                    playback_buffer.push_pcm16_chunk(payload)
            playback_buffer.flush_if_idle(0.08)
            stream.write(playback_buffer.pop_block_pcm16())

    input_stream = sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_frames,
        device=_resolve_device(args.input_device),
    )

    output_stream = None
    if not args.disable_playback:
        output_stream = sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=output_chunk_frames,
            device=_resolve_device(args.output_device),
        )
    barge_in_enabled = _is_barge_in_enabled(
        args.barge_in_mode,
        has_local_playback=output_stream is not None,
    )

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:

        async def close_session() -> None:
            nonlocal session_end_sent
            if session_end_sent:
                return
            session_end_sent = True
            try:
                await _send_json(
                    ws,
                    {"type": "session.close", "session_id": session_id},
                    send_lock,
                )
            except Exception:
                pass

        session_config: dict[str, Any] = {
            "speaker": args.speaker,
            "memory_turns": args.memory_turns,
            "output_chunk_sec": args.output_chunk_sec,
            "text_mode": args.text_mode,
            "include_transcript_in_query": args.include_transcript_in_query,
            "trim_with_vad": args.trim_with_vad,
            "turn_detection": {
                "mode": args.turn_detection_mode,
                "threshold": args.turn_threshold,
                "min_speech_ms": args.turn_min_speech_ms,
                "min_silence_ms": args.turn_min_silence_ms,
                "speech_pad_ms": args.turn_speech_pad_ms,
            },
        }
        if args.system_prompt is not None:
            session_config["system_prompt"] = args.system_prompt

        await _send_json(
            ws,
            {
                "type": "session.open",
                "session_id": session_id,
                "config": session_config,
            },
            send_lock,
        )

        first = json.loads(await ws.recv())
        if first.get("type") != "session.opened":
            raise RuntimeError(f"Failed to start session: {first}")
        logging.info(
            "%s Session opened: %s", _tag("SESSION", "cyan", use_color), first
        )

        async def receiver() -> None:
            while not stop_event.is_set():
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    if not stop_event.is_set():
                        logging.info(
                            "%s Connection closed by server",
                            _tag("CONNECT", "yellow", use_color),
                        )
                    stop_event.set()
                    break
                msg = json.loads(raw)
                event_type = msg.get("type")

                if event_type == "response.stream.opened":
                    logging.info(
                        "%s Turn stream started", _tag("TURN", "blue", use_color)
                    )
                elif event_type == "response.started":
                    state.generating = True
                    state.barge_in_sent = False
                    if output_stream is not None:
                        playback_queue.put(None)
                    logging.info(
                        "%s Response started turn_id=%s",
                        _tag("TURN", "blue", use_color),
                        msg.get("turn_id"),
                    )
                elif event_type == "response.audio.delta":
                    sample_rate = int(msg.get("sample_rate", OUTPUT_SAMPLE_RATE))
                    channels = int(msg.get("channels", 1))
                    mime_type = str(msg.get("mime_type", ""))
                    if sample_rate != OUTPUT_SAMPLE_RATE or channels != 1:
                        logging.warning(
                            "%s Dropping non-24k mono chunk sr=%s channels=%s",
                            _tag("AUDIO", "yellow", use_color),
                            sample_rate,
                            channels,
                        )
                        continue
                    if mime_type and "audio/pcm" not in mime_type:
                        logging.warning(
                            "%s Dropping non-PCM chunk mime=%s",
                            _tag("AUDIO", "yellow", use_color),
                            mime_type,
                        )
                        continue
                    try:
                        audio_bytes = base64.b64decode(msg["audio_b64"])
                    except Exception:
                        logging.warning(
                            "%s Dropping invalid audio_b64 chunk",
                            _tag("AUDIO", "yellow", use_color),
                        )
                        continue
                    if len(audio_bytes) % 2 == 1:
                        audio_bytes = audio_bytes[:-1]
                    if not audio_bytes:
                        continue
                    if output_stream is not None:
                        playback_queue.put(audio_bytes)
                elif event_type == "response.text.delta":
                    assistant_label = _paint("ASSISTANT", "magenta", use_color)
                    print(f"{assistant_label}: {msg.get('text', '')}")
                elif event_type == "response.done":
                    state.generating = False
                    state.barge_in_sent = False
                    if output_stream is not None:
                        playback_queue.put(None)
                    metrics = msg.get("metrics") or {}
                    logging.info(
                        "%s Response done metrics=%s",
                        _tag("TURN", "blue", use_color),
                        metrics,
                    )
                elif event_type == "response.cancelled":
                    state.generating = False
                    state.barge_in_sent = False
                    if output_stream is not None:
                        playback_queue.put(None)
                    logging.info(
                        "%s Response cancelled", _tag("TURN", "blue", use_color)
                    )
                elif event_type == "error":
                    logging.error(
                        "%s Server error: %s", _tag("SERVER", "red", use_color), msg
                    )
                else:
                    logging.debug(
                        "%s Event: %s", _tag("SERVER", "white", use_color), msg
                    )

        async def mic_sender() -> None:
            pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)

            while not stop_event.is_set():
                chunk = await asyncio.to_thread(
                    _get_mic_chunk_with_timeout, mic_queue, 0.2
                )
                if chunk is None:
                    continue
                rms = _rms_from_pcm16le(chunk)
                is_voice = rms >= args.client_vad_threshold
                if args.turn_detection_mode == "server_vad":
                    if (
                        barge_in_enabled
                        and state.generating
                        and is_voice
                        and not state.barge_in_sent
                    ):
                        await _send_json(
                            ws,
                            {"type": "response.cancel", "session_id": session_id},
                            send_lock,
                        )
                        state.barge_in_sent = True
                        logging.info(
                            "%s Barge-in detected: sent response.cancel",
                            _tag("INTERRUPT", "yellow", use_color),
                        )
                    await _send_json(
                        ws,
                        {
                            "type": "input.audio.append",
                            "session_id": session_id,
                            "audio_b64": base64.b64encode(chunk).decode("ascii"),
                        },
                        send_lock,
                    )
                    continue

                if not state.in_turn:
                    pre_roll.append(chunk)
                    if not is_voice:
                        continue

                    if (
                        barge_in_enabled
                        and state.generating
                        and not state.barge_in_sent
                    ):
                        await _send_json(
                            ws,
                            {"type": "response.cancel", "session_id": session_id},
                            send_lock,
                        )
                        state.barge_in_sent = True
                        logging.info(
                            "%s Barge-in detected: sent response.cancel",
                            _tag("INTERRUPT", "yellow", use_color),
                        )

                    state.in_turn = True
                    state.voice_ms = float(args.client_chunk_ms)
                    state.silence_ms = 0.0

                    while pre_roll:
                        frame = pre_roll.popleft()
                        await _send_json(
                            ws,
                            {
                                "type": "input.audio.append",
                                "session_id": session_id,
                                "audio_b64": base64.b64encode(frame).decode("ascii"),
                            },
                            send_lock,
                        )
                    continue

                await _send_json(
                    ws,
                    {
                        "type": "input.audio.append",
                        "session_id": session_id,
                        "audio_b64": base64.b64encode(chunk).decode("ascii"),
                    },
                    send_lock,
                )

                if is_voice:
                    state.voice_ms += float(args.client_chunk_ms)
                    state.silence_ms = 0.0
                else:
                    state.silence_ms += float(args.client_chunk_ms)

                if state.voice_ms >= float(
                    args.client_min_speech_ms
                ) and state.silence_ms >= float(args.client_pause_ms):
                    await _send_json(
                        ws,
                        {"type": "input.turn.commit", "session_id": session_id},
                        send_lock,
                    )
                    logging.info(
                        "%s Committed turn voice_ms=%.0f silence_ms=%.0f",
                        _tag("TURN", "blue", use_color),
                        state.voice_ms,
                        state.silence_ms,
                    )
                    state.in_turn = False
                    state.voice_ms = 0.0
                    state.silence_ms = 0.0
                    pre_roll.clear()

        async def command_loop() -> None:
            prompt = _paint("Command (/quit to disconnect): ", "white", use_color)
            while not stop_event.is_set():
                try:
                    raw = await asyncio.to_thread(input, prompt)
                except EOFError:
                    raw = "/quit"
                command = raw.strip()
                if not command:
                    continue
                if command == "/quit":
                    logging.info(
                        "%s Received /quit, disconnecting...",
                        _tag("CMD", "yellow", use_color),
                    )
                    stop_event.set()
                    io_stop.set()
                    await close_session()
                    await ws.close()
                    break
                logging.info(
                    "%s Unknown command: %s", _tag("CMD", "yellow", use_color), command
                )

        input_stream.start()
        mic_thread = threading.Thread(
            target=mic_reader_loop,
            args=(input_stream,),
            daemon=True,
        )
        mic_thread.start()

        playback_thread = None
        if output_stream is not None:
            logging.info(
                "%s Playback pipeline: PCM16 mono %dHz, jitter=%dms, crossfade=%.1fms",
                _tag("AUDIO", "cyan", use_color),
                OUTPUT_SAMPLE_RATE,
                jitter_buffer_ms,
                crossfade_ms,
            )
            output_stream.start()
            playback_thread = threading.Thread(
                target=playback_writer_loop,
                args=(output_stream,),
                daemon=True,
            )
            playback_thread.start()

        logging.info(
            "%s Mic streaming started (%s, chunk=%sms). Press Ctrl+C to stop.",
            _tag("AUDIO", "cyan", use_color),
            args.turn_detection_mode,
            args.client_chunk_ms,
        )
        if args.turn_detection_mode == "client_commit":
            logging.info(
                "%s Client VAD settings pause=%sms threshold=%.4f.",
                _tag("AUDIO", "cyan", use_color),
                args.client_pause_ms,
                args.client_vad_threshold,
            )
        else:
            logging.info(
                "%s Server turn detection threshold=%.2f min_speech=%sms min_silence=%sms pad=%sms.",
                _tag("AUDIO", "cyan", use_color),
                args.turn_threshold,
                args.turn_min_speech_ms,
                args.turn_min_silence_ms,
                args.turn_speech_pad_ms,
            )
        if args.turn_detection_mode == "server_vad":
            logging.info(
                "%s Barge-in mode=%s enabled=%s (local_playback=%s)",
                _tag("INTERRUPT", "cyan", use_color),
                args.barge_in_mode,
                barge_in_enabled,
                output_stream is not None,
            )
        logging.info(
            "%s Type /quit then Enter to disconnect", _tag("CMD", "white", use_color)
        )

        receiver_task = asyncio.create_task(receiver())
        sender_task = asyncio.create_task(mic_sender())
        command_task = asyncio.create_task(command_loop())

        try:
            done, pending = await asyncio.wait(
                {receiver_task, sender_task, command_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    if isinstance(exc, ConnectionClosedOK):
                        logging.info(
                            "%s Connection closed cleanly",
                            _tag("CONNECT", "yellow", use_color),
                        )
                        continue
                    raise exc
            stop_event.set()
            io_stop.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        finally:
            stop_event.set()
            io_stop.set()
            receiver_task.cancel()
            sender_task.cancel()
            command_task.cancel()
            mic_thread.join(timeout=1.0)
            if playback_thread is not None:
                playback_thread.join(timeout=1.0)
            input_stream.stop()
            input_stream.close()
            if output_stream is not None:
                output_stream.stop()
                output_stream.close()

            await close_session()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime microphone streaming client for Chroma voicebot WS server"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://127.0.0.1:8765",
        help="WebSocket URL of the voicebot server",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session_id to use for the connection (default: random)",
    )

    parser.add_argument(
        "--speaker",
        type=str,
        default="scarlett_johansson",
        help="Speaker/voice name to use for the assistant",
        choices=[
            "scarlett_johansson",
            "ariana_grande",
            "donald_trump",
            "lebron_james",
            "ben_vyin",
        ],
    )
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=6,
        help="Number of recent turns to include in context memory. (text only)",
    )
    parser.add_argument(
        "--output-chunk-sec",
        type=float,
        default=0.24,
        help="Duration of each output audio chunk in seconds. Lower values can reduce latency but increase risk of underflows.",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default="final",
        choices=["none", "sentence", "final"],
        help="When to send partial transcript text from the current turn. 'sentence' sends completed sentences, 'final' sends only the final transcript at the end of the turn, and 'none' disables transcript updates.",
    )
    parser.add_argument(
        "--include-transcript-in-query",
        action="store_true",
        help="Include transcript text in the same-turn user query (text + audio)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional per-session system prompt override",
    )
    parser.add_argument(
        "--trim-with-vad",
        action="store_true",
        help="Enable VAD trim before server inference",
    )
    parser.add_argument(
        "--turn-detection-mode",
        type=str,
        default="client_commit",
        choices=["client_commit", "server_vad"],
        help="Select who owns turn segmentation",
    )
    parser.add_argument(
        "--barge-in-mode",
        type=str,
        default="auto",
        choices=["auto", "off", "force"],
        help="Barge-in cancel policy: auto disables cancel when local playback is on; off never cancels; force always cancels.",
    )
    parser.add_argument(
        "--turn-threshold",
        type=float,
        default=0.5,
        help="Server-side turn detector threshold",
    )
    parser.add_argument(
        "--turn-min-speech-ms",
        type=int,
        default=250,
        help="Server-side minimum speech duration in milliseconds",
    )
    parser.add_argument(
        "--turn-min-silence-ms",
        type=int,
        default=500,
        help="Server-side silence duration before auto commit",
    )
    parser.add_argument(
        "--turn-speech-pad-ms",
        type=int,
        default=200,
        help="Server-side VAD padding in milliseconds",
    )

    parser.add_argument(
        "--client-chunk-ms",
        type=int,
        default=40,
        help="Client chunk duration in milliseconds",
    )
    parser.add_argument(
        "--client-pre-roll-ms",
        type=int,
        default=200,
        help="Client pre-roll before speech trigger in milliseconds",
    )
    parser.add_argument(
        "--client-min-speech-ms",
        type=int,
        default=280,
        help="Client-side minimum speech duration in milliseconds",
    )
    parser.add_argument(
        "--client-pause-ms",
        type=int,
        default=500,
        help="Client-side silence duration to commit a turn in milliseconds",
    )
    parser.add_argument(
        "--client-vad-threshold",
        type=float,
        default=0.015,
        help="Client-side RMS VAD threshold",
    )

    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="Audio input device index or name (default: system default)",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        help="Audio output device index or name (default: system default)",
    )
    parser.add_argument(
        "--disable-playback", action="store_true", help="Disable audio playback"
    )
    parser.add_argument(
        "--jitter-buffer-ms",
        type=int,
        default=300,
        help="Jitter buffer duration in milliseconds",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=8.0,
        help=f"Chunk boundary crossfade duration (0-{int(MAX_CROSSFADE_MS)} ms)",
    )
    parser.add_argument(
        "--no-device-prompt",
        action="store_true",
        help="Disable interactive audio device selection prompt on startup",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()

    use_color = _should_use_color(args.no_color)
    _configure_logging(args.log_level)
    _interactive_device_setup(args, use_color)

    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        logging.info("Stopped by user at %d", _now_ms())


if __name__ == "__main__":
    main()
