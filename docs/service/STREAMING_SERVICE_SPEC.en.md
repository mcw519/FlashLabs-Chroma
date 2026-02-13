# Chroma Streaming Service Spec (WebSocket V2)

## 1. Scope
This document is an implementation-aligned spec for `scripts/run_voicebot_ws.py`.
It covers:
- server startup and runtime capabilities
- session config schema and normalization rules
- client/server event contracts and timing
- server-side VAD auto-commit and cancel behavior
- persisted session log artifacts

## 2. Quick Server Startup

### 2.1 Minimal startup
```bash
python scripts/run_voicebot_ws.py \
  --host 0.0.0.0 \
  --port 8765 \
  --decode-mode full_turn \
  --prompt-speaker scarlett_johansson
```

### 2.2 Low-latency incremental decode
```bash
python scripts/run_voicebot_ws.py \
  --decode-mode overlap_stream \
  --overlap-frames 2 \
  --output-chunk-sec 0.24
```

### 2.3 Enable server-side ASR
```bash
python scripts/run_voicebot_ws.py \
  --server-asr-model openai/whisper-small \
  --server-asr-language en \
  --server-asr-device auto \
  --server-asr-timeout-sec 1.2
```

### 2.4 Disable text branch and persisted session logs
```bash
python scripts/run_voicebot_ws.py \
  --disable-text \
  --disable-session-log
```

## 3. Audio Contract
- Client -> Server (`input.audio.append.audio_b64`)
  - PCM16LE, 16kHz, mono
- Server -> Client (`response.audio.delta.audio_b64`)
  - PCM16LE, 24kHz, mono
  - `mime_type=audio/pcm;rate=24000;encoding=s16le`

## 4. Session Config Schema
```json
{
  "speaker": "scarlett_johansson",
  "memory_turns": 6,
  "output_chunk_sec": 0.24,
  "text_mode": "final",
  "include_transcript_in_query": false,
  "system_prompt": null,
  "trim_with_vad": false,
  "turn_detection": {
    "mode": "server_vad",
    "threshold": 0.5,
    "min_speech_ms": 250,
    "min_silence_ms": 500,
    "speech_pad_ms": 200
  }
}
```

Notes:
- `SessionConfigV2` dataclass default `text_mode` is `final`.
- The mic client script defaults to `text_mode=sentence` and sends that explicitly.

## 5. Server Normalization Rules

| Field | Rule |
| --- | --- |
| `speaker` | Unknown speaker falls back to server startup default (`--prompt-speaker`). |
| `memory_turns` | Values `<0` normalize to `0`. |
| `output_chunk_sec` | Values `<=0` normalize to server default (`--output-chunk-sec`). |
| `text_mode` | Only `none/sentence/final` accepted, otherwise normalized to `sentence`. |
| `include_transcript_in_query` | Boolean coercion: `1/true/yes/on -> true`; `0/false/no/off/"" -> false`. `false`: transcript is excluded from same-turn query and appended to memory after input preparation (affects later turns). `true`: transcript is included in same-turn user query (`text + audio`). |
| `system_prompt` | `null` clears override; string is trimmed, must be non-empty, max 4000 chars. |
| `trim_with_vad` | Same boolean coercion behavior. |
| `turn_detection.mode` | Only `client_commit/server_vad`, otherwise normalized to `server_vad`. |
| `turn_detection.threshold` | Must be `(0,1]`, otherwise normalized to `0.5`. |
| `turn_detection.min_speech_ms` | Values `<0` normalize to `0`. |
| `turn_detection.min_silence_ms` | Values `<0` normalize to `0`. |
| `turn_detection.speech_pad_ms` | Values `<0` normalize to `0`. |

## 6. Turn Ownership Modes
- `turn_detection.mode=client_commit`
  - Client must send `input.turn.commit`.
  - Server does not auto-commit.
- `turn_detection.mode=server_vad`
  - Server continuously evaluates incoming audio with Silero `VADIterator`.
  - Server auto-commits when speech/silence criteria are satisfied.

## 7. Event Protocol

### 7.1 Client -> Server
1. `session.open`
2. `session.update`
3. `input.audio.append`
4. `input.turn.commit`
5. `response.cancel`
6. `session.close`

### 7.2 Server -> Client
1. `session.opened`
2. `session.updated`
3. `input.audio.accepted`
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta`
7. `response.text.delta` (optional)
8. `response.done`
9. `response.cancelled`
10. `session.closed`
11. `error`

## 8. Payload Notes

### 8.1 `session.open`
```json
{"type":"session.open","session_id":"s1","config":{...}}
```
- `session_id` is optional; server generates UUID when omitted.
- `session.opened` includes normalized config.

### 8.2 `input.audio.append`
```json
{"type":"input.audio.append","session_id":"s1","audio_b64":"..."}
```
- `audio_b64` is required and must be valid base64.
- Success returns `input.audio.accepted` with `num_bytes`.

### 8.3 `input.turn.commit`
```json
{"type":"input.turn.commit","session_id":"s1","transcript":"hello"}
```
- `transcript` is optional.
- If server ASR is enabled and succeeds, server transcript overrides client transcript.

### 8.4 `response.cancel`
```json
{"type":"response.cancel","session_id":"s1"}
```
- Asynchronous request; no dedicated ack event is guaranteed.
- Successful interruption eventually yields `response.cancelled`.

### 8.5 `response.done`
Core fields:
- `session_id`, `turn_id`
- `text` (possibly `null`)
- `metrics` (`ttfs_ms`, `audio_out_sec`, `tokens_per_sec`, etc.)

## 9. Typical Timelines

### 9.1 `client_commit`
1. `session.open`
2. `input.audio.append` x N
3. `input.turn.commit`
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.text.delta` (optional)
8. `response.done`

### 9.2 `server_vad`
1. `session.open` with `turn_detection.mode=server_vad`
2. continuous `input.audio.append`
3. server auto-commit trigger
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.done` or `response.cancelled`

## 10. Error Codes
Common `error.code` values:
- `bad_json`
- `missing_type`
- `unknown_type`
- `invalid_audio_payload`
- `invalid_base64`
- `request_failed`
- `audio_commit_failed`
- engine-originated codes may also appear, such as `empty_audio`, `audio_too_short`, `generation_failed`

## 11. Persisted Session Logs
Enabled by default (`--disable-session-log` turns it off).

Directory: `logs/YYYY-MM-DD/<session_id>/`
- Session id is sanitized for filesystem safety when needed.

Artifacts:
- `user_0001.wav`, `user_0002.wav`, ... (16kHz PCM16 mono)
- `bot_0001.wav`, `bot_0002.wav`, ... (24kHz PCM16 mono)
- `conversation_log.json` (turn-level event/text/metrics/error records)

## 12. Server Startup Flags (`scripts/run_voicebot_ws.py`)

| Flag | Default | Description |
| --- | --- | --- |
| `--model-path` | `None` | Local model path; default behavior uses HF model id. |
| `--bot-config` | `None` | JSON/TOML bot config (can provide system prompt). |
| `--prompt-speaker` | `scarlett_johansson` | Session speaker fallback. |
| `--max-new-tokens` | `1000` | Max generation steps per turn. |
| `--max-text-new-tokens` | `64` | Thinker text branch cap. |
| `--temperature` | `0.7` | Sampling temperature. |
| `--top-p` | `0.9` | Nucleus sampling threshold. |
| `--output-chunk-sec` | `0.24` | Default audio chunk seconds. |
| `--decode-mode` | `full_turn` | `full_turn` or `overlap_stream`. |
| `--overlap-frames` | `2` | Used only in `overlap_stream`. |
| `--use-half-precision` | `false` | Request fp16 when CUDA is available. |
| `--disable-text` | `false` | Disable text output events. |
| `--max-sessions` | `3` | Maximum active sessions. |
| `--host` | `0.0.0.0` | Bind host. |
| `--port` | `8765` | Bind port. |
| `--max-message-size` | `8388608` | Max WebSocket message size in bytes. |
| `--warmup` | `false` | Run one warmup turn at startup. |
| `--server-asr-model` | `""` | Server-side ASR model id/path. |
| `--server-asr-language` | `""` | Optional ASR language hint. |
| `--server-asr-device` | `auto` | `auto/cpu/cuda`. |
| `--server-asr-timeout-sec` | `1.2` | ASR timeout per turn. |
| `--session-log-root` | `None` | Session log root (`<repo>/logs` by default). |
| `--disable-session-log` | `false` | Disable persisted session logs. |

### 12.1 LLM Generation Step Profiler (Environment Variables)
To observe smoothness at the per-generation-step level, enable the built-in profiler in `chroma/generation_chroma.py`:

- `CHROMA_GEN_STEP_PROFILE`
  - Enable with `1/true/yes/on`; disabled by default.
- `CHROMA_GEN_STEP_PROFILE_INTERVAL`
  - Emit one profiling log every N steps, default `10`.
- `CHROMA_GEN_STEP_PROFILE_CUDA_SYNC`
  - Whether to call `torch.cuda.synchronize()` before/after timing, enabled by default (`1`).
  - Keep this enabled on GPU for accurate step latency.

Example:
```bash
CHROMA_GEN_STEP_PROFILE=1 \
CHROMA_GEN_STEP_PROFILE_INTERVAL=10 \
CHROMA_GEN_STEP_PROFILE_CUDA_SYNC=1 \
python scripts/run_voicebot_ws.py --decode-mode overlap_stream --overlap-frames 2 --output-chunk-sec 0.24
```

`gen-step ...` log fields:
- `step_ms`: total latency of one generation step
- `forward_ms`: backbone forward latency
- `sample_ms`: token sampling latency
- `decoder_ms`: depth decoder generation latency
- `put_ms`: streamer handoff latency
- `step_p50` / `step_p95`: running median / p95 over observed steps

The end of each turn emits a `gen-step summary` line with overall `step_p50/step_p95` and stage-level p50 values.
