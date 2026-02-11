# Chroma Streaming Service Spec (V2)

## 1. Scope
This document defines the WebSocket protocol for Chroma streaming inference.
V2 is **breaking** and replaces previous event names.

## 2. Audio Contract
- Client -> Server (`input.audio.append.audio_b64`): PCM16LE, 16kHz, mono
- Server -> Client (`response.audio.delta.audio_b64`): PCM16LE, 24kHz, mono

## 3. Session Config Schema
```json
{
  "speaker": "scarlett_johansson",
  "memory_turns": 6,
  "output_chunk_sec": 0.24,
  "text_mode": "sentence",
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

## 4. Session Config Parameters

| Field | Type | Default | Allowed / Normalization | Behavior |
| --- | --- | --- | --- | --- |
| `speaker` | `string` | `scarlett_johansson` | If unknown, server falls back to startup default speaker. | Selects prompt audio/text persona for style. |
| `memory_turns` | `int` | `6` | `<0` is normalized to `0`. | Number of recent user/assistant turns kept in prompt memory context. |
| `output_chunk_sec` | `float` | `0.24` | `<=0` is normalized to `0.24`. | Target audio chunk duration emitted in `response.audio.delta`. |
| `text_mode` | `none \| sentence \| final` | `sentence` | Invalid values become `sentence`. | Controls text emission policy (`response.text.delta`). |
| `include_transcript_in_query` | `bool` | `false` | String/number booleans are coerced by server (`true/1/yes/on`, `false/0/no/off`). | If `true`, transcript text is injected into same-turn model query with audio. |
| `system_prompt` | `string \| null` | `null` | If string: trimmed, must be non-empty, max `4000` chars. | Per-session system prompt override. `null` clears override. |
| `trim_with_vad` | `bool` | `false` | Boolean coercion applies. | If `true`, pre-inference audio is trimmed with VAD boundaries. |
| `turn_detection.mode` | `client_commit \| server_vad` | `server_vad` | Invalid values normalize to `server_vad`. | Decides who owns turn boundary commit. |
| `turn_detection.threshold` | `float` | `0.5` | Must be `(0, 1]`; else normalized to `0.5`. | Silero VAD sensitivity for server-owned turn detection. |
| `turn_detection.min_speech_ms` | `int` | `250` | `<0` normalized to `0`. | Minimum speech duration for a valid speech region. |
| `turn_detection.min_silence_ms` | `int` | `500` | `<0` normalized to `0`. | Required trailing silence before server auto-commit. |
| `turn_detection.speech_pad_ms` | `int` | `200` | `<0` normalized to `0`. | Speech boundary padding around detected speech segments. |

## 5. Turn Ownership
- `turn_detection.mode=client_commit`: client sends `input.turn.commit`; server does not auto-commit.
- `turn_detection.mode=server_vad`: server auto-commits with Silero `VADIterator` in incremental streaming mode.

## 6. Client -> Server Events
1. `session.open`
2. `session.update`
3. `input.audio.append`
4. `input.turn.commit`
5. `response.cancel`
6. `session.close`

### 6.1 Event Payload Notes
- `session.open`: `session_id` optional; server generates one if omitted. `config` optional.
- `session.update`: requires `session_id`; `config` is partial patch and merged with existing session config.
- `input.audio.append`: requires `session_id` + non-empty `audio_b64` (base64 PCM16/16k/mono bytes).
- `input.turn.commit`: requires `session_id`; optional `transcript` (`string`).
- `response.cancel`: requires `session_id`; asynchronous request, no explicit acknowledgement event.
- `session.close`: requires `session_id`; server responds with `session.closed`.

## 7. Server -> Client Events
1. `session.opened`
2. `session.updated`
3. `input.audio.accepted`
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta`
7. `response.text.delta`
8. `response.done`
9. `response.cancelled`
10. `session.closed`
11. `error`

### 7.1 Event Payload Notes
- `session.opened` / `session.updated`: include normalized `config`.
- `input.audio.accepted`: includes `num_bytes` for accepted chunk size.
- `response.stream.opened`: emitted before model events for a committed turn.
- `response.audio.delta`: always mono 24k PCM payload (`audio_b64`).
- `response.done`: includes `metrics` (`ttfs_ms`, `chunk_gap_ms`, throughput fields, etc.).
- `error`: shape `{type, session_id, code, message, details?}`.

## 8. Typical Flow
1. `session.open`
2. `input.audio.append` x N
3. (`input.turn.commit` in `client_commit` mode)
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.text.delta` (optional)
8. `response.done`

## 9. Error Codes
- `bad_json`
- `missing_type`
- `unknown_type`
- `invalid_audio_payload`
- `invalid_base64`
- `request_failed`
- `audio_commit_failed`

## 10. V1 -> V2 Migration Guide (Breaking)

This upgrade is V2-only and does not provide a V1 adapter.  
If a client still sends V1 event names, the server returns `unknown_type`.

### 10.1 Event Name Mapping

| V1 | V2 | Notes |
| --- | --- | --- |
| `session.start` | `session.open` | Create session. |
| `session.started` | `session.opened` | Session created event. |
| `audio.append` | `input.audio.append` | Upload audio chunk. |
| `audio.appended` | `input.audio.accepted` | Server accepted audio chunk. |
| `audio.commit` | `input.turn.commit` | Commit turn and trigger inference. |
| `response.stream.started` | `response.stream.opened` | Stream-open marker before model events. |
| `response.cancel` | `response.cancel` | Name unchanged, response behavior changed. |
| `session.end` | `session.close` | Close session. |
| `session.ended` | `session.closed` | Session closed event. |
| `session.update` | `session.update` | Name unchanged. |

### 10.2 Session Config Mapping

| V1 Field | V2 Field | Mapping Rule |
| --- | --- | --- |
| `auto_commit` | `turn_detection.mode` | `true -> server_vad`; `false -> client_commit`. |
| `vad_threshold` | `turn_detection.threshold` | Direct mapping. |
| `vad_min_speech_ms` | `turn_detection.min_speech_ms` | Direct mapping. |
| `vad_min_silence_ms` | `turn_detection.min_silence_ms` | Direct mapping. |
| `vad_speech_pad_ms` | `turn_detection.speech_pad_ms` | Direct mapping. |

Notes:
- V2 uses `turn_detection` as the only source of turn-segmentation settings.
- V1 flat fields are not applied by the V2 schema; keeping them becomes a silent no-op and can cause config drift.

### 10.3 Behavioral Changes You Must Handle

1. `response.cancel` no longer emits `response.cancelled.requested`.
2. Successful cancellation is still surfaced by `response.cancelled` in the model event stream.
3. Turn ownership is mutually exclusive:
   - `turn_detection.mode=client_commit`
   - `turn_detection.mode=server_vad`
4. `session.open` / `session.update` return normalized `config`; treat this as the source of truth.

### 10.4 Upgrade Checklist

1. Rename all V1 events to V2 names (see 10.1).
2. Replace flat `auto_commit` + `vad_*` fields with `turn_detection` object (see 10.2).
3. Remove any dependency on `response.cancelled.requested`; use `response.cancelled` / `response.done` as terminal signals.
4. Ensure client chooses exactly one turn mode (`client_commit` or `server_vad`) and does not mix dual-side turn segmentation.
