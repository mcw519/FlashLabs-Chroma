# Chroma S2S 推論鏈路（Streaming Runtime）

## 1. 模型架構摘要
- Reasoner：基於 Qwen2.5-Omni-3B 的 thinker
- Backbone：基於 Llama3 的因果語言骨幹
- Decoder：基於 Llama3 的 audio codebook decoder
- Codec：Mimi 24kHz waveform codec

## 2. 對齊基準
串流執行路徑對齊單次推論基準：
- `scripts/run_chroma.py`

對齊重點：
1. conversation 組裝（`system + user(audio)`，可選 transcript）
2. `processor(..., add_generation_prompt=True, tokenize=False, prompt_audio, prompt_text)`
3. tensor 的 device/dtype 搬移
4. `model.generate` 採樣流程
5. codec decode clamp（`min(2047, vocab_size-1)`）

## 3. 串流單回合流程
1. 透過 `input.audio.append` 累積 PCM16 16k mono 音訊
2. commit 後：
  - PCM16 -> float32
  - 可選 VAD trim（`trim_with_vad`）
  - 組裝含 memory context 的多模態輸入
3. 啟動 streamer：
  - frame tokens -> codec decode -> 24k PCM chunk
  - 連續送出 `response.audio.delta`
4. 可選文字分支：
  - thinker 產生文字
  - 送出 `response.text.delta`
5. 收尾：
  - 正常結束送 `response.done` + metrics
  - 中斷則送 `response.cancelled`

## 4. 指標
`response.done.metrics` 主要欄位：
- `raw_audio_sec`
- `trimmed_audio_sec`
- `ttfs_ms`
- `chunk_gap_ms`
- `audio_out_sec`
- `audio_tokens`
- `tokens_per_sec`
- `tokens_per_sec_post_ttfs`
- `cancel_to_stop_ms`
