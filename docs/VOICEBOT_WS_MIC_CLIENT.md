# Voicebot WebSocket + Microphone Client Guide

This document explains how to run and use the streaming voicebot server with the local microphone client.

## 1. Components

- Server: `scripts/run_voicebot_ws.py`
- Mic client: `scripts/run_voicebot_ws_mic_client.py`
- Shared engine: `chroma/engine/streaming_engine.py`
- WebSocket transport: `chroma/transport/ws_server.py`

## 2. Audio Formats

- Client -> Server (`audio.append.audio_b64`): PCM16LE, 16kHz, mono
- Server -> Client (`response.audio.delta.audio_b64`): PCM16LE, 24kHz, mono

## 3. Start Server

```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

Common options:

- `--model-path`: local model path (optional)
- `--max-new-tokens`: audio length control
- `--max-text-new-tokens`: text output length
- `--output-chunk-sec`: response chunk cadence (default: `0.24`)
- `--disable-text`: audio-only mode
- `--max-sessions`: concurrent session limit

## 4. Start Microphone Client

Install client dependencies (if missing):

```bash
python -m pip install sounddevice websockets
```

Run:

```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

At startup, the client will prompt for:
- Microphone device
- Speaker device (or `none` to disable playback)

During runtime:
- Type `/quit` and press Enter to disconnect cleanly.

What the mic client does:

- Captures microphone in real time (16k mono PCM16)
- Uses simple RMS-based voice activity detection
- Sends `audio.commit` automatically when pause is detected
- Plays streamed response audio locally
- Sends `response.cancel` automatically on barge-in

## 5. Client Parameters

Session/output:

- `--session-id`: fixed session id (optional)
- `--speaker`: prompt speaker
- `--memory-turns`: context turns (default `6`)
- `--output-chunk-sec`: response chunk cadence
- `--text-mode`: `none|sentence|final`
- `--include-transcript-in-query`: include transcript in same-turn query (`text + audio`, default off)

VAD/turn segmentation:

- `--chunk-ms`: input frame size (default `40`)
- `--pre-roll-ms`: pre-roll audio before speech start (default `200`)
- `--min-speech-ms`: minimum speech duration to commit (default `280`)
- `--pause-ms`: silence duration threshold to commit (default `500`)
- `--vad-threshold`: RMS threshold (default `0.015`)

Audio devices:

- `--input-device`: input device name or index
- `--output-device`: output device name or index
- `--disable-playback`: disable local playback
- `--no-device-prompt`: skip startup interactive device selection
- `--no-color`: disable colorized console output

## 6. WebSocket Event Flow

Typical flow per session:

1. `session.start`
2. `audio.append` (0..n times)
3. `audio.commit`
4. Server emits:
   - `response.stream.started`
   - `response.started`
   - `response.audio.delta` (0..n times)
   - `response.text.delta` (optional)
   - `response.done`

Barge-in flow:

1. Client detects user speech during model generation
2. Client sends `response.cancel`
3. Client starts appending new speech for next turn

## 7. Event Examples

Client -> Server:

```json
{"type":"session.start","session_id":"mic-demo","config":{"speaker":"scarlett_johansson","memory_turns":6,"output_chunk_sec":0.24,"text_mode":"sentence","include_transcript_in_query":false}}
```

```json
{"type":"audio.append","session_id":"mic-demo","audio_b64":"..."}
```

```json
{"type":"audio.commit","session_id":"mic-demo"}
```

```json
{"type":"response.cancel","session_id":"mic-demo"}
```

Server -> Client:

```json
{"type":"response.audio.delta","session_id":"mic-demo","turn_id":"3","audio_b64":"...","sample_rate":24000,"channels":1}
```

```json
{"type":"response.done","session_id":"mic-demo","turn_id":"3","text":"...","metrics":{"ttfs_ms":420.0,"chunk_gap_ms":190.0}}
```

## 8. Tuning Tips

- Too sensitive (false trigger): increase `--vad-threshold` (e.g. `0.02`)
- Missing speech start: decrease `--vad-threshold` (e.g. `0.01`)
- Commit too early: increase `--pause-ms`
- Commit too late: decrease `--pause-ms`
- Choppy response playback: increase server/client `--output-chunk-sec` slightly

## 9. Troubleshooting

- `Port already in use`: change server `--port`
- `ModuleNotFoundError: sounddevice`: install with pip
- No mic sound: verify input device (`--input-device`)
- No speaker output: verify output device (`--output-device`) or remove `--disable-playback`
- Server `error` events: check payload schema and base64 validity

## 10. Quick Smoke Test

1. Start server.
2. Start mic client.
3. Say one short sentence and pause.
4. Confirm logs show:
   - `Committed turn ...`
   - `response.started`
   - `response.done`
5. Speak again while assistant is talking.
6. Confirm log shows `Barge-in detected: sent response.cancel`.
