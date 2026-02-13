# Chroma Streaming Service 規格（WebSocket V2）

## 1. 文件目的
本文是 `scripts/run_voicebot_ws.py` 對外服務協議的實作對照文件，涵蓋：
- 服務啟動與能力範圍
- session config schema 與正規化規則
- Client/Server 事件格式與時序
- server-side VAD auto-commit 與 cancel 行為
- session log 落地規格

## 2. 快速啟動（Server）

### 2.1 最小啟動
```bash
python scripts/run_voicebot_ws.py \
  --host 0.0.0.0 \
  --port 8765 \
  --decode-mode full_turn \
  --prompt-speaker scarlett_johansson
```

### 2.2 低延遲串流模式（增量 decode）
```bash
python scripts/run_voicebot_ws.py \
  --decode-mode overlap_stream \
  --overlap-frames 2 \
  --output-chunk-sec 0.24
```

### 2.3 啟用 server-side ASR
```bash
python scripts/run_voicebot_ws.py \
  --server-asr-model openai/whisper-small \
  --server-asr-language en \
  --server-asr-device auto \
  --server-asr-timeout-sec 1.2
```

### 2.4 關閉文字分支與落地 log
```bash
python scripts/run_voicebot_ws.py \
  --disable-text \
  --disable-session-log
```

## 3. 音訊契約
- Client -> Server（`input.audio.append.audio_b64`）
  - `PCM16LE`, `16kHz`, `mono`
- Server -> Client（`response.audio.delta.audio_b64`）
  - `PCM16LE`, `24kHz`, `mono`
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

註：
- `SessionConfigV2` dataclass 的 `text_mode` 預設是 `final`。
- mic client (`run_voicebot_ws_mic_client.py`) 預設送的是 `text_mode=sentence`。

## 5. Config 正規化規則（server 實作）

| 欄位 | 規則 |
| --- | --- |
| `speaker` | 不在允許清單時，fallback 到 server 啟動時的 `--prompt-speaker`。 |
| `memory_turns` | `<0` 正規化為 `0`。 |
| `output_chunk_sec` | `<=0` 回退到 server 預設（啟動參數 `--output-chunk-sec`）。 |
| `text_mode` | 僅接受 `none/sentence/final`，非法值變 `sentence`。 |
| `include_transcript_in_query` | 支援 bool coercion：`1/true/yes/on -> true`；`0/false/no/off/"" -> false`。`false`：transcript 不進當回合 query，會在 input 準備後寫入 memory（影響後續回合）。`true`：transcript 會進當回合 user query（`text + audio`）。 |
| `system_prompt` | `null` 代表清除覆蓋；字串會 `strip`，不可空字串，長度上限 4000。 |
| `trim_with_vad` | 同 bool coercion。 |
| `turn_detection.mode` | 僅接受 `client_commit/server_vad`，非法值變 `server_vad`。 |
| `turn_detection.threshold` | 需在 `(0,1]`，否則回退 `0.5`。 |
| `turn_detection.min_speech_ms` | `<0` 正規化為 `0`。 |
| `turn_detection.min_silence_ms` | `<0` 正規化為 `0`。 |
| `turn_detection.speech_pad_ms` | `<0` 正規化為 `0`。 |

## 6. Turn 主導模式
- `turn_detection.mode=client_commit`
  - client 必須發 `input.turn.commit`
  - server 不會自動切 turn
- `turn_detection.mode=server_vad`
  - server 用 Silero `VADIterator` 連續掃描 `input.audio.append`
  - 達到語音/靜音條件後自動 commit

## 7. 事件協議

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
7. `response.text.delta`（可選）
8. `response.done`
9. `response.cancelled`
10. `session.closed`
11. `error`

## 8. 事件 payload 重點

### 8.1 `session.open`
```json
{"type":"session.open","session_id":"s1","config":{...}}
```
- `session_id` 可省略，省略時 server 自動產生 UUID。
- 回應 `session.opened` 會帶「正規化後」config。

### 8.2 `input.audio.append`
```json
{"type":"input.audio.append","session_id":"s1","audio_b64":"..."}
```
- `audio_b64` 必填且必須為合法 base64。
- 成功回 `input.audio.accepted`，並帶 `num_bytes`。

### 8.3 `input.turn.commit`
```json
{"type":"input.turn.commit","session_id":"s1","transcript":"hello"}
```
- `transcript` 可選。
- 若 server ASR 啟用且成功，server transcript 會覆蓋 client transcript。

### 8.4 `response.cancel`
```json
{"type":"response.cancel","session_id":"s1"}
```
- 非同步，不保證回 ack 事件。
- 成功中斷時，後續事件流會出現 `response.cancelled`。

### 8.5 `response.done`
核心欄位：
- `session_id`, `turn_id`
- `text`（可能為 `null`）
- `metrics`（`ttfs_ms`, `audio_out_sec`, `tokens_per_sec` 等）

## 9. 典型時序

### 9.1 `client_commit` 模式
1. `session.open`
2. `input.audio.append` x N
3. `input.turn.commit`
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.text.delta`（可選）
8. `response.done`

### 9.2 `server_vad` 模式
1. `session.open`（`turn_detection.mode=server_vad`）
2. `input.audio.append` 持續串流
3. server 內部判斷 ready 後自動 commit
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.done` 或 `response.cancelled`

## 10. 錯誤碼
常見 `error.code`：
- `bad_json`
- `missing_type`
- `unknown_type`
- `invalid_audio_payload`
- `invalid_base64`
- `request_failed`
- `audio_commit_failed`
- engine 內部也可能回：`empty_audio`、`audio_too_short`、`generation_failed` 等

## 11. Session Log 落地
預設開啟（可用 `--disable-session-log` 關閉）。

目錄：`logs/YYYY-MM-DD/<session_id>/`
- 若 session_id 含非法檔名字元會被正規化成 `_`

檔案：
- `user_0001.wav`, `user_0002.wav`...（16kHz PCM16 mono）
- `bot_0001.wav`, `bot_0002.wav`...（24kHz PCM16 mono）
- `conversation_log.json`（每 turn 的事件、文字、metrics、error）

## 12. 啟動參數總表（Server）
腳本：`scripts/run_voicebot_ws.py`

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--model-path` | `None` | 本地模型路徑；不帶則用預設 HF model id。 |
| `--bot-config` | `None` | JSON/TOML bot 設定（可含 system prompt）。 |
| `--prompt-speaker` | `scarlett_johansson` | session speaker fallback。 |
| `--max-new-tokens` | `1000` | 每 turn 生成步數上限。 |
| `--max-text-new-tokens` | `64` | thinker 文字分支上限。 |
| `--temperature` | `0.7` | 取樣溫度。 |
| `--top-p` | `0.9` | nucleus sampling。 |
| `--output-chunk-sec` | `0.24` | 預設輸出 chunk 秒數。 |
| `--decode-mode` | `full_turn` | `full_turn` 或 `overlap_stream`。 |
| `--overlap-frames` | `2` | 僅 `overlap_stream` 生效。 |
| `--use-half-precision` | `false` | CUDA 可用時使用 fp16。 |
| `--disable-text` | `false` | 關閉文字事件輸出。 |
| `--max-sessions` | `3` | 最大同時 session。 |
| `--host` | `0.0.0.0` | 監聽位址。 |
| `--port` | `8765` | 監聽連接埠。 |
| `--max-message-size` | `8388608` | WS 單訊息大小上限（bytes）。 |
| `--warmup` | `false` | 啟動時跑一輪暖機。 |
| `--server-asr-model` | `""` | server ASR 模型 id/path。 |
| `--server-asr-language` | `""` | ASR 語言提示。 |
| `--server-asr-device` | `auto` | `auto/cpu/cuda`。 |
| `--server-asr-timeout-sec` | `1.2` | 每 turn ASR timeout。 |
| `--session-log-root` | `None` | log 根目錄（預設 `<repo>/logs`）。 |
| `--disable-session-log` | `false` | 關閉 session log 落地。 |

### 12.1 LLM generation step profiler（環境變數）
若要觀察「每個 generation step」是否順暢，可用環境變數啟用 `chroma/generation_chroma.py` 內建 profiler：

- `CHROMA_GEN_STEP_PROFILE`
  - `1/true/yes/on` 啟用；預設關閉。
- `CHROMA_GEN_STEP_PROFILE_INTERVAL`
  - 每 N step 輸出一次統計，預設 `10`。
- `CHROMA_GEN_STEP_PROFILE_CUDA_SYNC`
  - 是否在計時前後做 `torch.cuda.synchronize()`，預設開啟（`1`）。
  - GPU 下建議保持開啟，否則 step 時間可能偏小。

範例：
```bash
CHROMA_GEN_STEP_PROFILE=1 \
CHROMA_GEN_STEP_PROFILE_INTERVAL=10 \
CHROMA_GEN_STEP_PROFILE_CUDA_SYNC=1 \
python scripts/run_voicebot_ws.py --decode-mode overlap_stream --overlap-frames 2 --output-chunk-sec 0.24
```

log 欄位（`gen-step ...`）：
- `step_ms`：單一步總耗時
- `forward_ms`：backbone forward 耗時
- `sample_ms`：sampling 耗時
- `decoder_ms`：depth decoder generate 耗時
- `put_ms`：送入 streamer 耗時
- `step_p50` / `step_p95`：目前累積步數的中位數 / 95 分位

收尾會輸出 `gen-step summary`，包含整輪 `step_p50/step_p95` 與各子階段 p50。
