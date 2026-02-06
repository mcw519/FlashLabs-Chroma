# run: python scripts/run_voicebot_ws_mic_client.py --url ws://127.0.0.1:8765

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import queue
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

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


@dataclass
class TurnState:
    generating: bool = False
    in_turn: bool = False
    voice_ms: float = 0.0
    silence_ms: float = 0.0
    barge_in_sent: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime microphone streaming client for Chroma voicebot WS server"
    )
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765")
    parser.add_argument("--session-id", type=str, default=None)

    parser.add_argument("--speaker", type=str, default="scarlett_johansson")
    parser.add_argument("--memory-turns", type=int, default=6)
    parser.add_argument("--output-chunk-sec", type=float, default=0.24)
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


async def run_client(args: argparse.Namespace) -> None:
    session_id = args.session_id or f"mic-{uuid.uuid4().hex[:8]}"
    chunk_frames = max(1, int(INPUT_SAMPLE_RATE * (args.chunk_ms / 1000.0)))
    pre_roll_chunks = max(1, int(args.pre_roll_ms / args.chunk_ms))

    logging.info("Connecting to %s session_id=%s", args.url, session_id)

    mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=512)
    stop_event = asyncio.Event()
    send_lock = asyncio.Lock()
    state = TurnState()

    playback_queue: queue.SimpleQueue[bytes] = queue.SimpleQueue()
    playback_buffer = bytearray()

    def on_input(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            logging.warning("Mic status: %s", status)
        chunk = bytes(indata)
        try:
            mic_queue.put_nowait(chunk)
        except queue.Full:
            logging.warning("Mic queue full; dropping chunk")

    def on_output(outdata, frames, time_info, status) -> None:
        del time_info
        if status:
            logging.warning("Playback status: %s", status)
        needed = frames * 2
        while len(playback_buffer) < needed:
            try:
                playback_buffer.extend(playback_queue.get_nowait())
            except queue.Empty:
                break
        if len(playback_buffer) >= needed:
            outdata[:] = bytes(playback_buffer[:needed])
            del playback_buffer[:needed]
        else:
            have = len(playback_buffer)
            outdata[:have] = bytes(playback_buffer)
            outdata[have:] = b"\x00" * (needed - have)
            playback_buffer.clear()

    input_stream = sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_frames,
        callback=on_input,
        device=_resolve_device(args.input_device),
    )

    output_stream = None
    if not args.disable_playback:
        output_stream = sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=on_output,
            device=_resolve_device(args.output_device),
        )

    async with websockets.connect(args.url, max_size=8 * 1024 * 1024) as ws:
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
        logging.info("Session started: %s", first)

        async def receiver() -> None:
            while not stop_event.is_set():
                raw = await ws.recv()
                msg = json.loads(raw)
                event_type = msg.get("type")

                if event_type == "response.stream.started":
                    logging.info("Turn stream started")
                elif event_type == "response.started":
                    state.generating = True
                    logging.info("Response started turn_id=%s", msg.get("turn_id"))
                elif event_type == "response.audio.delta":
                    audio_bytes = base64.b64decode(msg["audio_b64"])
                    if output_stream is not None:
                        playback_queue.put(audio_bytes)
                elif event_type == "response.text.delta":
                    print(f"ASSISTANT: {msg.get('text', '')}")
                elif event_type == "response.done":
                    state.generating = False
                    state.barge_in_sent = False
                    metrics = msg.get("metrics") or {}
                    logging.info("Response done metrics=%s", metrics)
                elif event_type == "response.cancelled":
                    state.generating = False
                    state.barge_in_sent = False
                    logging.info("Response cancelled")
                elif event_type == "error":
                    logging.error("Server error: %s", msg)
                else:
                    logging.debug("Server event: %s", msg)

        async def mic_sender() -> None:
            pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)

            while not stop_event.is_set():
                chunk = await asyncio.to_thread(mic_queue.get)
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
                        logging.info("Barge-in detected: sent response.cancel")

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

                if (
                    state.voice_ms >= float(args.min_speech_ms)
                    and state.silence_ms >= float(args.pause_ms)
                ):
                    await _send_json(
                        ws,
                        {"type": "audio.commit", "session_id": session_id},
                        send_lock,
                    )
                    logging.info(
                        "Committed turn voice_ms=%.0f silence_ms=%.0f",
                        state.voice_ms,
                        state.silence_ms,
                    )
                    state.in_turn = False
                    state.voice_ms = 0.0
                    state.silence_ms = 0.0
                    pre_roll.clear()

        input_stream.start()
        if output_stream is not None:
            output_stream.start()

        logging.info(
            "Mic streaming started (chunk=%sms pause=%sms vad=%.4f). Press Ctrl+C to stop.",
            args.chunk_ms,
            args.pause_ms,
            args.vad_threshold,
        )

        receiver_task = asyncio.create_task(receiver())
        sender_task = asyncio.create_task(mic_sender())

        try:
            await asyncio.gather(receiver_task, sender_task)
        except asyncio.CancelledError:
            raise
        finally:
            stop_event.set()
            receiver_task.cancel()
            sender_task.cancel()
            input_stream.stop()
            input_stream.close()
            if output_stream is not None:
                output_stream.stop()
                output_stream.close()

            try:
                await _send_json(
                    ws,
                    {"type": "session.end", "session_id": session_id},
                    send_lock,
                )
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        logging.info("Stopped by user at %d", _now_ms())


if __name__ == "__main__":
    main()
