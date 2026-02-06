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
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any
import sys

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sounddevice. Install with `python -m pip install sounddevice`."
    ) from exc

try:
    import websockets
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: websockets. Install with `python -m pip install websockets`."
    ) from exc

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
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


@dataclass
class TurnState:
    generating: bool = False
    in_turn: bool = False
    voice_ms: float = 0.0
    silence_ms: float = 0.0
    barge_in_sent: bool = False


class _ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def __init__(self, use_color: bool) -> None:
        super().__init__("%(levelname)s | %(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color:
            color_name = self.LEVEL_COLORS.get(record.levelno)
            if color_name:
                record.levelname = (
                    f"{COLOR_CODES[color_name]}{original_levelname}{RESET}"
                )
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime microphone streaming client for Chroma voicebot WS server"
    )
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765")
    parser.add_argument("--session-id", type=str, default=None)

    parser.add_argument("--speaker", type=str, default="scarlett_johansson")
    parser.add_argument("--memory-turns", type=int, default=6)
    parser.add_argument("--output-chunk-sec", type=float, default=0.5)
    parser.add_argument(
        "--text-mode",
        type=str,
        default="sentence",
        choices=["none", "sentence", "final"],
    )

    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--pre-roll-ms", type=int, default=200)
    parser.add_argument("--min-speech-ms", type=int, default=280)
    parser.add_argument("--pause-ms", type=int, default=500)
    parser.add_argument("--vad-threshold", type=float, default=0.015)

    parser.add_argument("--input-device", type=str, default=None)
    parser.add_argument("--output-device", type=str, default=None)
    parser.add_argument("--disable-playback", action="store_true")
    parser.add_argument("--no-device-prompt", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


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


def _get_mic_chunk_with_timeout(mic_queue: queue.Queue[bytes], timeout_sec: float) -> bytes | None:
    try:
        return mic_queue.get(timeout=timeout_sec)
    except queue.Empty:
        return None


def _should_use_color(no_color: bool) -> bool:
    if no_color:
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


def _configure_logging(level_name: str, use_color: bool) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter(use_color))
    root.addHandler(handler)
    root.setLevel(level)


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
        (idx, dev) for idx, dev in enumerate(all_devices) if int(dev.get("max_input_channels", 0)) > 0
    ]
    output_devices = [
        (idx, dev) for idx, dev in enumerate(all_devices) if int(dev.get("max_output_channels", 0)) > 0
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
        print(_paint("No input devices found; use system default.", "yellow", use_color))

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
        print(_paint("No output devices found; playback disabled.", "yellow", use_color))
        args.disable_playback = True


async def run_client(args: argparse.Namespace) -> None:
    use_color = _should_use_color(args.no_color)
    session_id = args.session_id or f"mic-{uuid.uuid4().hex[:8]}"
    ws_url = _normalize_ws_url(args.url)
    chunk_frames = max(1, int(INPUT_SAMPLE_RATE * (args.chunk_ms / 1000.0)))
    output_chunk_frames = max(1, int(OUTPUT_SAMPLE_RATE * (args.chunk_ms / 1000.0)))
    pre_roll_chunks = max(1, int(args.pre_roll_ms / args.chunk_ms))

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

    playback_queue: queue.SimpleQueue[bytes] = queue.SimpleQueue()
    io_stop = threading.Event()
    session_end_sent = False

    def mic_reader_loop(stream: sd.RawInputStream) -> None:
        while not io_stop.is_set():
            chunk, overflowed = stream.read(chunk_frames)
            if overflowed:
                logging.warning("%s Input overflow detected", _tag("AUDIO", "yellow", use_color))
            try:
                mic_queue.put_nowait(bytes(chunk))
            except queue.Full:
                logging.warning("%s Mic queue full; dropping chunk", _tag("AUDIO", "yellow", use_color))

    def playback_writer_loop(stream: sd.RawOutputStream) -> None:
        target_bytes = output_chunk_frames * 2
        silence = b"\x00" * target_bytes
        buffer = bytearray()
        while not io_stop.is_set():
            deadline = time.time() + 0.06
            while len(buffer) < target_bytes and time.time() < deadline:
                try:
                    buffer.extend(playback_queue.get_nowait())
                except queue.Empty:
                    break
            if len(buffer) >= target_bytes:
                stream.write(bytes(buffer[:target_bytes]))
                del buffer[:target_bytes]
            else:
                stream.write(silence)

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

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        async def close_session() -> None:
            nonlocal session_end_sent
            if session_end_sent:
                return
            session_end_sent = True
            try:
                await _send_json(
                    ws,
                    {"type": "session.end", "session_id": session_id},
                    send_lock,
                )
            except Exception:
                pass

        await _send_json(
            ws,
            {
                "type": "session.start",
                "session_id": session_id,
                "config": {
                    "speaker": args.speaker,
                    "memory_turns": args.memory_turns,
                    "output_chunk_sec": args.output_chunk_sec,
                    "text_mode": args.text_mode,
                },
            },
            send_lock,
        )

        first = json.loads(await ws.recv())
        if first.get("type") != "session.started":
            raise RuntimeError(f"Failed to start session: {first}")
        logging.info("%s Session started: %s", _tag("SESSION", "cyan", use_color), first)

        async def receiver() -> None:
            while not stop_event.is_set():
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    if not stop_event.is_set():
                        logging.info("%s Connection closed by server", _tag("CONNECT", "yellow", use_color))
                    stop_event.set()
                    break
                msg = json.loads(raw)
                event_type = msg.get("type")

                if event_type == "response.stream.started":
                    logging.info("%s Turn stream started", _tag("TURN", "blue", use_color))
                elif event_type == "response.started":
                    state.generating = True
                    logging.info(
                        "%s Response started turn_id=%s",
                        _tag("TURN", "blue", use_color),
                        msg.get("turn_id"),
                    )
                elif event_type == "response.audio.delta":
                    audio_bytes = base64.b64decode(msg["audio_b64"])
                    if output_stream is not None:
                        playback_queue.put(audio_bytes)
                elif event_type == "response.text.delta":
                    assistant_label = _paint("ASSISTANT", "magenta", use_color)
                    print(f"{assistant_label}: {msg.get('text', '')}")
                elif event_type == "response.done":
                    state.generating = False
                    state.barge_in_sent = False
                    metrics = msg.get("metrics") or {}
                    logging.info(
                        "%s Response done metrics=%s",
                        _tag("TURN", "blue", use_color),
                        metrics,
                    )
                elif event_type == "response.cancelled":
                    state.generating = False
                    state.barge_in_sent = False
                    logging.info("%s Response cancelled", _tag("TURN", "blue", use_color))
                elif event_type == "error":
                    logging.error("%s Server error: %s", _tag("SERVER", "red", use_color), msg)
                else:
                    logging.debug("%s Event: %s", _tag("SERVER", "white", use_color), msg)

        async def mic_sender() -> None:
            pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)

            while not stop_event.is_set():
                chunk = await asyncio.to_thread(_get_mic_chunk_with_timeout, mic_queue, 0.2)
                if chunk is None:
                    continue
                rms = _rms_from_pcm16le(chunk)
                is_voice = rms >= args.vad_threshold

                if not state.in_turn:
                    pre_roll.append(chunk)
                    if not is_voice:
                        continue

                    if state.generating and not state.barge_in_sent:
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
                    state.voice_ms = float(args.chunk_ms)
                    state.silence_ms = 0.0

                    while pre_roll:
                        frame = pre_roll.popleft()
                        await _send_json(
                            ws,
                            {
                                "type": "audio.append",
                                "session_id": session_id,
                                "audio_b64": base64.b64encode(frame).decode("ascii"),
                            },
                            send_lock,
                        )
                    continue

                await _send_json(
                    ws,
                    {
                        "type": "audio.append",
                        "session_id": session_id,
                        "audio_b64": base64.b64encode(chunk).decode("ascii"),
                    },
                    send_lock,
                )

                if is_voice:
                    state.voice_ms += float(args.chunk_ms)
                    state.silence_ms = 0.0
                else:
                    state.silence_ms += float(args.chunk_ms)

                if state.voice_ms >= float(
                    args.min_speech_ms
                ) and state.silence_ms >= float(args.pause_ms):
                    await _send_json(
                        ws,
                        {"type": "audio.commit", "session_id": session_id},
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
                    logging.info("%s Received /quit, disconnecting...", _tag("CMD", "yellow", use_color))
                    stop_event.set()
                    io_stop.set()
                    await close_session()
                    await ws.close()
                    break
                logging.info("%s Unknown command: %s", _tag("CMD", "yellow", use_color), command)

        input_stream.start()
        mic_thread = threading.Thread(
            target=mic_reader_loop,
            args=(input_stream,),
            daemon=True,
        )
        mic_thread.start()

        playback_thread = None
        if output_stream is not None:
            output_stream.start()
            playback_thread = threading.Thread(
                target=playback_writer_loop,
                args=(output_stream,),
                daemon=True,
            )
            playback_thread.start()

        logging.info(
            "%s Mic streaming started (chunk=%sms pause=%sms vad=%.4f). Press Ctrl+C to stop.",
            _tag("AUDIO", "cyan", use_color),
            args.chunk_ms,
            args.pause_ms,
            args.vad_threshold,
        )
        logging.info("%s Type /quit then Enter to disconnect", _tag("CMD", "white", use_color))

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
    args = parse_args()
    use_color = _should_use_color(args.no_color)
    _configure_logging(args.log_level, use_color)
    _interactive_device_setup(args, use_color)

    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        logging.info("Stopped by user at %d", _now_ms())


if __name__ == "__main__":
    main()
