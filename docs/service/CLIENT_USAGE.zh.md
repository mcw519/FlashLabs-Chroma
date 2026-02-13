# Chroma WS Client 使用指南（V2）

## 1. 前置需求
- 已安裝專案依賴（建議 `uv sync`）
- server 可讀到 Chroma 模型（預設 `FlashLabs/Chroma-4B`）
- 麥克風 client 需要 `sounddevice` 可用音訊裝置

## 2. 最快開始（Mic Client）

### 2.1 開一個 Terminal 啟動 Server
```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --decode-mode full_turn \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

### 2.2 另一個 Terminal 啟動麥克風 Client
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

### 2.3 結束連線
- 在 client 輸入 `/quit` + Enter，會送 `session.close` 並斷線。

## 3. 常用啟動組合

### 3.1 低延遲語音串流（overlap_stream）
Server：
```bash
python scripts/run_voicebot_ws.py \
  --decode-mode overlap_stream \
  --overlap-frames 2 \
  --output-chunk-sec 0.20
```

Client：
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --output-chunk-sec 0.20
```

### 3.2 server 主導切 turn（server_vad）
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --turn-detection-mode server_vad \
  --turn-threshold 0.5 \
  --turn-min-speech-ms 250 \
  --turn-min-silence-ms 500 \
  --turn-speech-pad-ms 200
```

### 3.3 client 主導切 turn（client_commit）
```bash
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --turn-detection-mode client_commit \
  --client-vad-threshold 0.015 \
  --client-min-speech-ms 280 \
  --client-pause-ms 500
```

### 3.4 關閉文字輸出 / 關閉 session log
```bash
python scripts/run_voicebot_ws.py \
  --disable-text \
  --disable-session-log
```

### 3.5 啟用 server ASR（commit 時自動轉錄）
```bash
python scripts/run_voicebot_ws.py \
  --server-asr-model openai/whisper-small \
  --server-asr-language en \
  --server-asr-timeout-sec 1.2
```

## 4. 參數地圖（誰影響什麼）

### 4.1 Server（`scripts/run_voicebot_ws.py`）
核心：
- 推論：`--max-new-tokens`、`--temperature`、`--top-p`
- 串流：`--decode-mode`、`--overlap-frames`、`--output-chunk-sec`
- 服務：`--host`、`--port`、`--max-message-size`、`--max-sessions`
- 功能：`--disable-text`、`--warmup`、`--server-asr-*`、`--disable-session-log`

### 4.2 Mic Client 送進 session config 的參數
- `--speaker` -> `speaker`
- `--memory-turns` -> `memory_turns`
- `--output-chunk-sec` -> `output_chunk_sec`
- `--text-mode` -> `text_mode`
- `--include-transcript-in-query` -> `include_transcript_in_query`
- `--system-prompt` -> `system_prompt`
- `--trim-with-vad` -> `trim_with_vad`
- `--turn-detection-mode` / `--turn-*` -> `turn_detection.*`

Transcript 時序：
- `include_transcript_in_query=false`：當回合 query 不帶 transcript；transcript 只會加入短期 memory 供後續回合使用。
- `include_transcript_in_query=true`：當回合 query 會帶 transcript（`text + audio`）。

### 4.3 Mic Client 本地行為（不送 server）
- VAD/送流：`--client-*`
- 音訊裝置：`--input-device`、`--output-device`
- 播放平滑：`--jitter-buffer-ms`、`--crossfade-ms`
- 互動與顯示：`--no-device-prompt`、`--no-color`、`--log-level`

## 5. 純 WebSocket 使用範例（Python）

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

## 6. Barge-in（打斷模型說話）
- client 可在偵測到使用者開口後送：
```json
{"type":"response.cancel","session_id":"..."}
```
- 成功中斷時，事件流會回 `response.cancelled`。

## 7. Troubleshooting
- `invalid_base64`
  - `audio_b64` 非合法 base64，或編碼來源不是 PCM bytes。
- `audio_too_short`
  - commit 後有效語音太短（可能被 VAD trim 掉）。可提高錄音長度或調整 VAD 參數。
- 收到音訊但沒聲音
  - 確認 client 播放未 `--disable-playback`，且 `--output-device` 正確。
- `Max sessions limit reached`
  - 調大 server `--max-sessions` 或關閉舊 session。

## 8. 參考文件
- 服務協議：`docs/service/STREAMING_SERVICE_SPEC.zh.md`
- 模型串流演算：`docs/model/S2S_PIPELINE.zh.md`
