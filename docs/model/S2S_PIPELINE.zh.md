# Chroma S2S 演算與串流設計（Streaming Runtime）

## 1. 目的與範圍
本文件對齊目前程式實作（`chroma/modeling_chroma.py`、`chroma/generation_chroma.py`、`chroma/engine/streaming_engine.py`），說明：
- 模型結構與各子模組責任
- 單步生成演算法（audio frame 如何被產生）
- 串流解碼設計（`full_turn` / `overlap_stream`）
- 服務端一個 turn 的完整執行流程與指標

## 2. 模型結構（Architecture）

### 2.1 四層結構
- Thinker：`Qwen2.5-Omni Thinker`，負責語意推理與文字 token 生成。
- Backbone：Llama 風格 Causal LM（預設 16 層，hidden size 2048），負責每步先產生 `codebook-0`。
- Decoder：Llama 風格深度解碼器（預設 4 層，hidden size 1024），補齊剩餘 audio codebooks。
- Codec：`Mimi`，把 audio codebook 序列還原為波形（服務輸出為 24kHz）。

### 2.2 關鍵 config（預設）
來自 `chroma/configuration_chroma.py`：
- `audio_num_codebooks = 8`
- `vocab_size = 2051`
- `codebook_eos_token_id = 0`
- `codebook_pad_token_id = 2050`
- `codec_config.frame_rate = 12.5`

### 2.3 每步的 token/frame 概念
在 Chroma 裡，模型一次 loop 的目標是產生「1 個完整 audio frame」：
1. Backbone 先出第 1 個 codebook（`codebook-0`）
2. Decoder 再補 `codebook-1..7`
3. 合併成 shape=`[num_codebooks]` 的 frame

## 3. 單步生成演算法（核心）
對齊 `ChromaGenerationMixin._sample()`：

1. `prepare_inputs_for_generation()`
- 首步：建立 prompt embeds（音訊+文字上下文）
- 後續步：用上一個 frame 的 audio token 回饋
- 若 `thinker_flag=True`，Thinker 會再推進 1 個 text token，並把 thinker hidden state + token embedding 注入 backbone 輸入

2. Backbone 前向
- 取最後一步 logits，對 `codebook-0` 做取樣（greedy 或 sampling）

3. Decoder 補齊 codebooks
- 以 `codebook-0` + backbone hidden state 作條件
- 強制生成 `audio_num_codebooks - 1` 個 token
- 得到完整 frame（預期長度必須等於 `audio_num_codebooks`）

4. Streamer 收 frame
- `streamer.put(next_tokens.cpu())`
- 同步檢查 EOS（所有 codebook 都是 eos token）

5. 停止條件
- `codebook_eos_token_id` 命中
- 或 stopping criteria（含 cancel criteria）
- 或達到 `max_new_tokens`

## 4. 串流解碼設計（Engine Streamer）
`_EngineAudioStreamer` 主要模式：

### 4.1 `full_turn`（預設）
- 全部 frame 先累積在記憶體
- turn 結束後一次 `codec_model.decode()`
- 通常只送出 1 個 `response.audio.delta`
- 優點：穩定、音質一致性高
- 代價：首包延遲（TTFS）通常較高

### 4.2 `overlap_stream`
- 累積到 `frames_per_chunk` 就丟到背景 thread 解碼
- 解碼時可保留 `overlap_frames` 歷史 frame 作重疊，避免邊界爆音
- 逐塊送出多個 `response.audio.delta`
- 優點：更低的串流延遲
- 代價：拼接與參數調校較敏感

### 4.3 codec 安全夾限
解碼前會做：
- `max_token_id = min(2047, vocab_size - 1)`
- token clamp 到 `[0, max_token_id]`

這和 `scripts/run_chroma.py` 的離線推論路徑一致。

## 5. 服務端單回合（turn）時序
對齊 `StreamingVoicebotEngine.commit_turn()`：

1. 取出並清空 session audio buffer
2. PCM16(16k) -> float32；可選 VAD trim（`trim_with_vad`）
3. `_prepare_inputs()` 組 conversation：
- system prompt（可疊 memory context）
- user audio（可選帶 transcript text）
- speaker 對應 prompt_audio/prompt_text
4. 當前回合 inputs 準備完成後，可選把 pending transcript 寫入 memory（`role=user`）
5. 啟動生成 thread，先送 `response.started`
6. 每段音訊 emit `response.audio.delta`
7. 文字輸出：
- 先嘗試解碼 streamer 收集到的 thinker token
- 若無，fallback 到 `_generate_text()`
- `text_mode=sentence` 會切第一句，`final` 回完整，`none` 關閉
8. 完成送 `response.done`（含 metrics）
9. 若取消則送 `response.cancelled`

## 6. 記憶體上下文（memory）
- `memory_turns` 控制保留回合數（user+assistant 成對）
- 實作上保留 `memory_turns * 2` 筆條目
- 寫入 system prompt 末端，形成短期對話記憶
- `include_transcript_in_query=false` 時，transcript 不會進當回合 query，只會透過 memory 影響後續回合

## 7. 取消機制（barge-in / cancel）
- `response.cancel` 會 set session 的 cancel event
- 生成 loop 綁定 `AbortOnCancelCriteria`
- streamer 在 `put()` 也會檢查 cancel flag
- 成功取消後回 `response.cancelled`，並附 `cancel_to_stop_ms`

## 8. 主要 metrics 定義
`response.done.metrics`：
- `raw_audio_sec`：commit 前原始音長
- `trimmed_audio_sec`：VAD trim 後音長
- `ttfs_ms`：從 commit 到第一個音訊 chunk 的時間
- `chunk_gap_ms`：chunk 間平均間隔
- `audio_out_sec`：輸出音訊長度
- `audio_tokens`：估算輸出 token 數
- `tokens_per_sec`：整體吞吐
- `tokens_per_sec_post_ttfs`：扣掉 TTFS 後吞吐
- `cancel_to_stop_ms`：取消請求到停止耗時

### 8.1 LLM 每步 generation 觀測（step profiler）
除了 `response.done.metrics` 的整輪統計外，也可開啟逐步 profiler 觀察 jitter：

- `CHROMA_GEN_STEP_PROFILE=1`
- `CHROMA_GEN_STEP_PROFILE_INTERVAL=10`
- `CHROMA_GEN_STEP_PROFILE_CUDA_SYNC=1`（GPU 建議保持開啟）

每 N 步會輸出：
- `step_ms`：單一步總耗時
- `forward_ms`：backbone forward
- `sample_ms`：sampling
- `decoder_ms`：depth decoder generate
- `put_ms`：streamer.put handoff
- `step_p50` / `step_p95`：目前累積步數的中位數 / 95 分位

結束時會輸出 `gen-step summary`（整輪 p50/p95 與各子階段 p50）。

## 9. 與離線推論對齊
離線腳本：`scripts/run_chroma.py`

串流/離線一致點：
- processor 打包方式（conversation + prompt_audio/prompt_text）
- device/dtype 搬移規則
- `model.generate` 取樣流程
- codec 解碼前 token clamp

差異點：
- 串流模式有 session state、memory、VAD、cancel、event 協議
- 離線模式為一次性輸入/輸出，不管理 session
