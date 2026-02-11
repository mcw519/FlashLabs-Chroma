# Chroma WS Client Usage (V2)

## 1. Start Server
```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

## 2. Start Microphone Client
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

## 3. Turn Detection Modes
### Client-owned turn segmentation
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --turn-detection-mode client_commit \
  --client-vad-threshold 0.015 \
  --client-min-speech-ms 280 \
  --client-pause-ms 500
```

### Server-owned turn segmentation
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --turn-detection-mode server_vad \
  --turn-threshold 0.5 \
  --turn-min-speech-ms 250 \
  --turn-min-silence-ms 500 \
  --turn-speech-pad-ms 200
```

## 4. Client Flags
- Session: `--session-id`, `--speaker`, `--memory-turns`, `--output-chunk-sec`
- Prompt/response: `--text-mode`, `--include-transcript-in-query`, `--system-prompt`
- Server inference: `--trim-with-vad`, `--turn-*`
- Client VAD/input: `--client-*`
- Audio devices/playback: `--input-device`, `--output-device`, `--disable-playback`

## 5. Detailed Parameter Reference

### 5.1 Server Script (`scripts/run_voicebot_ws.py`)

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--model-path` | `string` | `None` | Local model path. If omitted, uses default HF model id. |
| `--bot-config` | `string` | `None` | JSON/TOML config; can provide startup `system_prompt`. |
| `--prompt-speaker` | `string` | `scarlett_johansson` | Default speaker used when session does not override `speaker`. |
| `--max-new-tokens` | `int` | `1000` | Max generated audio token steps per turn. |
| `--max-text-new-tokens` | `int` | `64` | Max tokens for thinker text branch. |
| `--temperature` | `float` | `0.7` | Sampling temperature for generation. |
| `--top-p` | `float` | `0.9` | Nucleus sampling threshold. |
| `--output-chunk-sec` | `float` | `0.24` | Default chunk duration if session omits `output_chunk_sec`. |
| `--use-half-precision` | `flag` | `false` | Requests fp16 when CUDA is available. |
| `--disable-text` | `flag` | `false` | Disables thinker text output events. |
| `--max-sessions` | `int` | `3` | Maximum active sessions in engine. |
| `--host` | `string` | `0.0.0.0` | WebSocket bind host. |
| `--port` | `int` | `8765` | WebSocket bind port. |
| `--max-message-size` | `int` | `8388608` | Max WS frame size (bytes). |
| `--warmup` | `flag` | `false` | Runs one warmup inference at startup. |
| `--server-asr-model` | `string` | `""` | Enables server-side ASR for commit transcript. |
| `--server-asr-language` | `string` | `""` | Optional ASR language hint. |
| `--server-asr-device` | `auto\|cpu\|cuda` | `auto` | ASR runtime device. |
| `--server-asr-timeout-sec` | `float` | `1.2` | ASR timeout per committed turn. |

### 5.2 Mic Client Session Flags (`scripts/run_voicebot_ws_mic_client.py`)

These flags map directly into `session.open.config`.

| Flag | Config Field | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `--speaker` | `speaker` | `string` | `scarlett_johansson` | Persona speaker for prompt style. |
| `--memory-turns` | `memory_turns` | `int` | `6` | `<0` normalized to `0` by server. |
| `--output-chunk-sec` | `output_chunk_sec` | `float` | `0.24` | `<=0` normalized to server default. |
| `--text-mode` | `text_mode` | `enum` | `sentence` | `none`, `sentence`, `final`. |
| `--include-transcript-in-query` | `include_transcript_in_query` | `bool` | `false` | Adds transcript text into same-turn model query. |
| `--system-prompt` | `system_prompt` | `string\|null` | `None` | Trimmed; must be non-empty if provided. |
| `--trim-with-vad` | `trim_with_vad` | `bool` | `false` | Enables pre-inference VAD trimming. |
| `--turn-detection-mode` | `turn_detection.mode` | `enum` | `client_commit` | Turn owner selector. |
| `--turn-threshold` | `turn_detection.threshold` | `float` | `0.5` | Used when mode is `server_vad`. |
| `--turn-min-speech-ms` | `turn_detection.min_speech_ms` | `int` | `250` | Used when mode is `server_vad`. |
| `--turn-min-silence-ms` | `turn_detection.min_silence_ms` | `int` | `500` | Used when mode is `server_vad`. |
| `--turn-speech-pad-ms` | `turn_detection.speech_pad_ms` | `int` | `200` | Used when mode is `server_vad`. |

### 5.3 Mic Client Local Input/Playback Flags

These flags are local client behavior and are not sent to server config.

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--client-chunk-ms` | `int` | `40` | Mic capture chunk size in ms. |
| `--client-pre-roll-ms` | `int` | `200` | Pre-roll buffer before client VAD start. |
| `--client-min-speech-ms` | `int` | `280` | Client-side commit threshold (mode=`client_commit`). |
| `--client-pause-ms` | `int` | `500` | Silence threshold to commit (mode=`client_commit`). |
| `--client-vad-threshold` | `float` | `0.015` | RMS voice activity threshold (mode=`client_commit`). |
| `--input-device` | `string` | `None` | Input device name or numeric index. |
| `--output-device` | `string` | `None` | Output device name or numeric index. |
| `--disable-playback` | `flag` | `false` | Disable local playback. |
| `--jitter-buffer-ms` | `int` | `300` | Playback jitter buffer size. |
| `--crossfade-ms` | `float` | `8.0` | Chunk boundary smoothing (`0` to `10`). |
| `--no-device-prompt` | `flag` | `false` | Skips interactive device picker. |
| `--no-color` | `flag` | `false` | Disables colored local tags. |
| `--log-level` | `string` | `INFO` | Local logger level. |

## 6. Runtime Commands
- Type `/quit` and press Enter to send `session.close` and disconnect.
