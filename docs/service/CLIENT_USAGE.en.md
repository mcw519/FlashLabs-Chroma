# Chroma WS Client Usage (V2)

## 1. Prerequisites
- Project dependencies installed (recommended: `uv sync`)
- Server can access Chroma model (default: `FlashLabs/Chroma-4B`)
- Microphone client requires working `sounddevice` audio devices

## 2. Fastest End-to-End Setup

### 2.1 Terminal A: start server
```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --decode-mode full_turn \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

### 2.2 Terminal B: start microphone client
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

### 2.3 Disconnect cleanly
- Type `/quit` in the client and press Enter (sends `session.close`).

## 3. Common Startup Recipes

### 3.1 Lower-latency streaming (`overlap_stream`)
Server:
```bash
python scripts/run_voicebot_ws.py \
  --decode-mode overlap_stream \
  --overlap-frames 2 \
  --output-chunk-sec 0.20
```

Client:
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --output-chunk-sec 0.20
```

### 3.2 Server-owned turn detection (`server_vad`)
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --turn-detection-mode server_vad \
  --turn-threshold 0.5 \
  --turn-min-speech-ms 250 \
  --turn-min-silence-ms 500 \
  --turn-speech-pad-ms 200
```

### 3.3 Client-owned turn detection (`client_commit`)
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --turn-detection-mode client_commit \
  --client-vad-threshold 0.015 \
  --client-min-speech-ms 280 \
  --client-pause-ms 500
```

### 3.4 Disable text output and persisted logs
```bash
python scripts/run_voicebot_ws.py \
  --disable-text \
  --disable-session-log
```

### 3.5 Enable server-side ASR
```bash
python scripts/run_voicebot_ws.py \
  --server-asr-model openai/whisper-small \
  --server-asr-language en \
  --server-asr-timeout-sec 1.2
```

## 4. Parameter Map

### 4.1 Server (`scripts/run_voicebot_ws.py`)
Main knobs:
- inference: `--max-new-tokens`, `--temperature`, `--top-p`
- streaming: `--decode-mode`, `--overlap-frames`, `--output-chunk-sec`
- service: `--host`, `--port`, `--max-message-size`, `--max-sessions`
- features: `--disable-text`, `--warmup`, `--server-asr-*`, `--disable-session-log`

### 4.2 Mic client flags mapped into session config
- `--speaker` -> `speaker`
- `--memory-turns` -> `memory_turns`
- `--output-chunk-sec` -> `output_chunk_sec`
- `--text-mode` -> `text_mode`
- `--include-transcript-in-query` -> `include_transcript_in_query`
- `--system-prompt` -> `system_prompt`
- `--trim-with-vad` -> `trim_with_vad`
- `--turn-detection-mode` / `--turn-*` -> `turn_detection.*`

Transcript timing:
- `include_transcript_in_query=false`: transcript is not used in the same-turn query; it is added to short-term memory for later turns.
- `include_transcript_in_query=true`: transcript is included in same-turn query (`text + audio`).

### 4.3 Mic client local-only behavior (not sent to server)
- VAD/send loop: `--client-*`
- device selection: `--input-device`, `--output-device`
- playback smoothing: `--jitter-buffer-ms`, `--crossfade-ms`
- UX/logging: `--no-device-prompt`, `--no-color`, `--log-level`

## 5. Raw WebSocket Integration Example (Python)

```python
import asyncio
import base64
import json
import wave
import websockets


def load_pcm16_16k_mono_b64(path: str) -> str:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        pcm = wf.readframes(wf.getnframes())
    return base64.b64encode(pcm).decode("ascii")


async def main():
    session_id = "demo-s1"
    audio_b64 = load_pcm16_16k_mono_b64("example/make_taco.wav")

    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({
            "type": "session.open",
            "session_id": session_id,
            "config": {
                "speaker": "scarlett_johansson",
                "text_mode": "final",
                "turn_detection": {"mode": "client_commit"}
            }
        }))
        print(await ws.recv())  # session.opened

        await ws.send(json.dumps({
            "type": "input.audio.append",
            "session_id": session_id,
            "audio_b64": audio_b64
        }))
        print(await ws.recv())  # input.audio.accepted

        await ws.send(json.dumps({
            "type": "input.turn.commit",
            "session_id": session_id,
            "transcript": "please summarize this audio"
        }))

        while True:
            msg = json.loads(await ws.recv())
            print(msg["type"])
            if msg["type"] in {"response.done", "response.cancelled", "error"}:
                break

        await ws.send(json.dumps({"type": "session.close", "session_id": session_id}))
        print(await ws.recv())  # session.closed


asyncio.run(main())
```

## 6. Barge-In Behavior
To interrupt ongoing generation, send:
```json
{"type":"response.cancel","session_id":"..."}
```
If cancellation succeeds, you will get `response.cancelled`.

## 7. Troubleshooting
- `invalid_base64`
  - `audio_b64` is not valid base64 or not encoded PCM bytes.
- `audio_too_short`
  - Effective speech after trim is too short; increase speech duration or tune VAD.
- Audio events arrive but no sound playback
  - Ensure playback is enabled and `--output-device` is correct.
- `Max sessions limit reached`
  - Increase server `--max-sessions` or close stale sessions.

## 8. Related Docs
- Service protocol: `docs/service/STREAMING_SERVICE_SPEC.en.md`
- Model runtime algorithm: `docs/model/S2S_PIPELINE.en.md`
