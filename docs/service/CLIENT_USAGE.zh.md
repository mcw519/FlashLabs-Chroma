# Chroma WS Client 使用指南（V2）

## 1. 啟動 Server
```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

## 2. 啟動麥克風 Client
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

## 3. Turn Detection 模式
### Client 主導切 turn
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --turn-detection-mode client_commit \
  --client-vad-threshold 0.015 \
  --client-min-speech-ms 280 \
  --client-pause-ms 500
```

### Server 主導切 turn
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --turn-detection-mode server_vad \
  --turn-threshold 0.5 \
  --turn-min-speech-ms 250 \
  --turn-min-silence-ms 500 \
  --turn-speech-pad-ms 200
```

## 4. 主要參數分類
- Session：`--session-id`、`--speaker`、`--memory-turns`、`--output-chunk-sec`
- Prompt/輸出：`--text-mode`、`--include-transcript-in-query`、`--system-prompt`
- Server 推論：`--trim-with-vad`、`--turn-*`
- Client VAD/輸入：`--client-*`
- 音訊裝置：`--input-device`、`--output-device`、`--disable-playback`

## 5. 詳細參數對照

### 5.1 Server 啟動參數（`scripts/run_voicebot_ws.py`）

| 參數 | 型別 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--model-path` | `string` | `None` | 本地模型路徑；不帶時走預設 HF model id。 |
| `--bot-config` | `string` | `None` | JSON/TOML 設定檔，可提供啟動時 `system_prompt`。 |
| `--prompt-speaker` | `string` | `scarlett_johansson` | session 未指定 speaker 時的預設值。 |
| `--max-new-tokens` | `int` | `1000` | 每回合音訊生成 token 上限。 |
| `--max-text-new-tokens` | `int` | `64` | thinker 文字分支 token 上限。 |
| `--temperature` | `float` | `0.7` | 採樣溫度。 |
| `--top-p` | `float` | `0.9` | nucleus sampling 門檻。 |
| `--output-chunk-sec` | `float` | `0.24` | session 未指定時的預設 chunk 秒數。 |
| `--use-half-precision` | `flag` | `false` | CUDA 可用時請求 fp16。 |
| `--disable-text` | `flag` | `false` | 關閉 thinker 文字輸出事件。 |
| `--max-sessions` | `int` | `3` | engine 允許同時存在的 session 數。 |
| `--host` | `string` | `0.0.0.0` | WebSocket 綁定 host。 |
| `--port` | `int` | `8765` | WebSocket 綁定 port。 |
| `--max-message-size` | `int` | `8388608` | WS 單訊息大小上限（bytes）。 |
| `--warmup` | `flag` | `false` | 啟動時先跑一輪 warmup。 |
| `--server-asr-model` | `string` | `""` | 啟用 server-side ASR。 |
| `--server-asr-language` | `string` | `""` | ASR 語言提示（可選）。 |
| `--server-asr-device` | `auto\|cpu\|cuda` | `auto` | ASR 執行裝置。 |
| `--server-asr-timeout-sec` | `float` | `1.2` | 每次 commit 的 ASR timeout。 |

### 5.2 Mic Client 會送到 Session Config 的參數（`scripts/run_voicebot_ws_mic_client.py`）

| 參數 | Config 欄位 | 型別 | 預設值 | 說明 |
| --- | --- | --- | --- | --- |
| `--speaker` | `speaker` | `string` | `scarlett_johansson` | persona speaker。 |
| `--memory-turns` | `memory_turns` | `int` | `6` | `<0` 會被 server 正規化成 `0`。 |
| `--output-chunk-sec` | `output_chunk_sec` | `float` | `0.24` | `<=0` 會被回退到預設值。 |
| `--text-mode` | `text_mode` | `enum` | `sentence` | `none`、`sentence`、`final`。 |
| `--include-transcript-in-query` | `include_transcript_in_query` | `bool` | `false` | 是否把 transcript 直接放進同回合 query。 |
| `--system-prompt` | `system_prompt` | `string\|null` | `None` | 若有值會 trim，且不可空字串。 |
| `--trim-with-vad` | `trim_with_vad` | `bool` | `false` | 推論前是否做 VAD trim。 |
| `--turn-detection-mode` | `turn_detection.mode` | `enum` | `client_commit` | 決定 turn 邊界主導方。 |
| `--turn-threshold` | `turn_detection.threshold` | `float` | `0.5` | `server_vad` 模式使用。 |
| `--turn-min-speech-ms` | `turn_detection.min_speech_ms` | `int` | `250` | `server_vad` 模式使用。 |
| `--turn-min-silence-ms` | `turn_detection.min_silence_ms` | `int` | `500` | `server_vad` 模式使用。 |
| `--turn-speech-pad-ms` | `turn_detection.speech_pad_ms` | `int` | `200` | `server_vad` 模式使用。 |

### 5.3 Mic Client 本地行為參數（不會寫進 session config）

| 參數 | 型別 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `--client-chunk-ms` | `int` | `40` | 麥克風擷取 chunk 長度。 |
| `--client-pre-roll-ms` | `int` | `200` | client VAD 觸發前保留的 pre-roll。 |
| `--client-min-speech-ms` | `int` | `280` | `client_commit` 模式判定可 commit 的最短語音。 |
| `--client-pause-ms` | `int` | `500` | `client_commit` 模式下靜音 commit 門檻。 |
| `--client-vad-threshold` | `float` | `0.015` | `client_commit` 模式 RMS 門檻。 |
| `--input-device` | `string` | `None` | 輸入裝置名稱或 index。 |
| `--output-device` | `string` | `None` | 輸出裝置名稱或 index。 |
| `--disable-playback` | `flag` | `false` | 關閉本地播放。 |
| `--jitter-buffer-ms` | `int` | `300` | 播放 jitter buffer 大小。 |
| `--crossfade-ms` | `float` | `8.0` | chunk 接縫平滑（`0` 到 `10`）。 |
| `--no-device-prompt` | `flag` | `false` | 跳過互動式裝置選單。 |
| `--no-color` | `flag` | `false` | 關閉本地顏色標籤。 |
| `--log-level` | `string` | `INFO` | 本地 logger level。 |

## 6. 執行時指令
- 輸入 `/quit` 後 Enter，client 會送 `session.close` 並斷線。
