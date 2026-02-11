# Chroma 串流服務規格（V2）

## 1. 範圍
本文定義 Chroma streaming inference 的 WebSocket 協議。  
V2 為**破壞性升級**，已取代舊事件命名。

## 2. 音訊契約
- Client -> Server（`input.audio.append.audio_b64`）：PCM16LE, 16kHz, mono
- Server -> Client（`response.audio.delta.audio_b64`）：PCM16LE, 24kHz, mono

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

## 4. Session Config 參數說明

| 欄位 | 型別 | 預設值 | 合法範圍 / 正規化規則 | 行為影響 |
| --- | --- | --- | --- | --- |
| `speaker` | `string` | `scarlett_johansson` | 若 speaker 不存在，server 會 fallback 到啟動時的預設 speaker。 | 決定 prompt audio/text 的 persona 風格。 |
| `memory_turns` | `int` | `6` | `<0` 會被正規化成 `0`。 | 控制寫入 prompt memory 的歷史 turn 數量。 |
| `output_chunk_sec` | `float` | `0.24` | `<=0` 會回退到 `0.24`。 | 控制 `response.audio.delta` 的目標分塊秒數。 |
| `text_mode` | `none \| sentence \| final` | `sentence` | 非法值會回退為 `sentence`。 | 控制 `response.text.delta` 的輸出策略。 |
| `include_transcript_in_query` | `bool` | `false` | 支援字串/數字布林轉換（如 `true/1/yes/on`、`false/0/no/off`）。 | `true` 時 transcript 會與 audio 一起進入同回合 query。 |
| `system_prompt` | `string \| null` | `null` | 若為字串：會 `strip`、不可空字串、上限 `4000` 字元。 | 覆蓋每個 session 的 system prompt；`null` 代表清除覆蓋。 |
| `trim_with_vad` | `bool` | `false` | 會做布林轉換。 | 是否在推論前先做 VAD 裁切。 |
| `turn_detection.mode` | `client_commit \| server_vad` | `server_vad` | 非法值會回退為 `server_vad`。 | 決定誰主導 turn 邊界。 |
| `turn_detection.threshold` | `float` | `0.5` | 需為 `(0, 1]`；超出會回退為 `0.5`。 | server 端 Silero VAD 靈敏度。 |
| `turn_detection.min_speech_ms` | `int` | `250` | `<0` 會被正規化成 `0`。 | 判定為有效語音所需最短語音長度。 |
| `turn_detection.min_silence_ms` | `int` | `500` | `<0` 會被正規化成 `0`。 | server auto-commit 前需要的尾端靜音時長。 |
| `turn_detection.speech_pad_ms` | `int` | `200` | `<0` 會被正規化成 `0`。 | VAD 語音片段前後 padding。 |

## 5. Turn 主導權
- `turn_detection.mode=client_commit`：由 client 發 `input.turn.commit`，server 不自動切 turn。
- `turn_detection.mode=server_vad`：由 server 透過 Silero VAD 自動 commit。

## 6. Client -> Server 事件
1. `session.open`
2. `session.update`
3. `input.audio.append`
4. `input.turn.commit`
5. `response.cancel`
6. `session.close`

### 6.1 事件 payload 補充
- `session.open`：`session_id` 可省略；省略時由 server 產生。`config` 可省略。
- `session.update`：必須帶 `session_id`；`config` 為局部更新，與既有 config 合併。
- `input.audio.append`：必須帶 `session_id` 與非空 `audio_b64`（base64 PCM16/16k/mono）。
- `input.turn.commit`：必須帶 `session_id`；`transcript` 可選（字串）。
- `response.cancel`：必須帶 `session_id`；為非同步請求，不會額外回傳 ack 事件。
- `session.close`：必須帶 `session_id`；server 會回 `session.closed`。

## 7. Server -> Client 事件
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

### 7.1 事件 payload 補充
- `session.opened` / `session.updated`：都會回傳「正規化後」的 `config`。
- `input.audio.accepted`：會帶 `num_bytes`，表示本次收進去的音訊大小。
- `response.stream.opened`：在單回合推論事件前先送出。
- `response.audio.delta`：固定為 24k 單聲道 PCM（base64）。
- `response.done`：包含 `metrics`（如 `ttfs_ms`、`chunk_gap_ms`、throughput 欄位）。
- `error`：格式 `{type, session_id, code, message, details?}`。

## 8. 典型流程
1. `session.open`
2. `input.audio.append` x N
3. （`client_commit` 模式下）`input.turn.commit`
4. `response.stream.opened`
5. `response.started`
6. `response.audio.delta` x N
7. `response.text.delta`（可選）
8. `response.done`

## 9. 錯誤碼
- `bad_json`
- `missing_type`
- `unknown_type`
- `invalid_audio_payload`
- `invalid_base64`
- `request_failed`
- `audio_commit_failed`
