# Chroma S2S Runtime: Model Algorithm and Streaming Design

## 1. Scope
This document matches the current implementation in:
- `chroma/modeling_chroma.py`
- `chroma/generation_chroma.py`
- `chroma/engine/streaming_engine.py`

It describes:
- model structure and module responsibilities
- per-step generation algorithm
- streaming decode design (`full_turn` vs `overlap_stream`)
- end-to-end server turn lifecycle

## 2. Model Structure

### 2.1 Four major blocks
- Thinker: Qwen2.5-Omni thinker branch for semantic/text progression.
- Backbone: Llama-style causal LM (default 16 layers, hidden size 2048), predicts `codebook-0`.
- Decoder: Llama-style depth decoder (default 4 layers, hidden size 1024), predicts remaining codebooks.
- Codec: Mimi decoder from codebooks to waveform (service output is 24kHz PCM).

### 2.2 Key default config values
From `chroma/configuration_chroma.py`:
- `audio_num_codebooks = 8`
- `vocab_size = 2051`
- `codebook_eos_token_id = 0`
- `codebook_pad_token_id = 2050`
- `codec_config.frame_rate = 12.5`

### 2.3 What one generation step means
One loop iteration generates one complete audio frame:
1. Backbone samples `codebook-0`
2. Decoder generates `codebook-1..7`
3. Combined frame shape is `[num_codebooks]`

## 3. Core Generation Algorithm
Aligned with `ChromaGenerationMixin._sample()`:

1. `prepare_inputs_for_generation()`
- First step: builds prompt embeddings from multimodal inputs.
- Next steps: reuses previous audio frame tokens.
- If `thinker_flag=True`, thinker advances one text token and injects thinker hidden state + token embedding into backbone input.

2. Backbone forward
- Last-step logits are used to sample `codebook-0` (greedy or sampling).

3. Decoder completion
- Decoder receives `codebook-0` and backbone hidden state.
- It must generate exactly `audio_num_codebooks - 1` tokens.
- Result is a complete frame.

4. Streamer ingestion
- `streamer.put(next_tokens.cpu())`
- EOS check happens at frame level.

5. Stop criteria
- codebook EOS reached
- stopping criteria (including cancel criteria)
- max token limit

## 4. Streaming Decode Design
`_EngineAudioStreamer` supports two decode modes.

### 4.1 `full_turn` (default)
- Collect all frames in memory.
- Decode once at end of turn.
- Usually emits one `response.audio.delta`.
- Better stability, higher TTFS.

### 4.2 `overlap_stream`
- Decode in chunks as frames accumulate.
- Optional overlap window (`overlap_frames`) smooths chunk boundaries.
- Emits multiple incremental `response.audio.delta` events.
- Lower latency, more tuning sensitivity.

### 4.3 Codec safety clamp
Before codec decode:
- `max_token_id = min(2047, vocab_size - 1)`
- token clamp into `[0, max_token_id]`

This is consistent with offline path in `scripts/run_chroma.py`.

## 5. Server Turn Lifecycle
Aligned with `StreamingVoicebotEngine.commit_turn()`:

1. Snapshot and clear session audio buffer.
2. Decode PCM16(16k) to float32, optionally VAD-trim.
3. Build conversation inputs via `_prepare_inputs()`:
- system prompt (+ optional memory context)
- user audio (+ optional transcript text)
- speaker prompt audio/text
4. After current-turn inputs are prepared, optionally append pending transcript to memory (`role=user`).
5. Start generation thread and emit `response.started`.
6. Emit `response.audio.delta` chunks.
7. Text path:
- decode thinker tokens collected by streamer first
- fallback to `_generate_text()` when needed
- `text_mode=sentence`: first sentence, `final`: full text, `none`: disabled
8. Emit `response.done` with metrics.
9. If cancelled, emit `response.cancelled`.

## 6. Memory Context
- `memory_turns` controls retained turn history.
- Implementation keeps up to `memory_turns * 2` items (user + assistant).
- Memory is appended to system prompt as short-term context.
- If `include_transcript_in_query=false`, transcript is excluded from same-turn query and only affects later turns through memory.

## 7. Cancellation Path
- `response.cancel` sets session cancel flag.
- Generation loop uses `AbortOnCancelCriteria`.
- Streamer also checks cancel flag in `put()`.
- Successful cancellation ends with `response.cancelled` and `cancel_to_stop_ms` metric.

## 8. Metrics
`response.done.metrics` fields include:
- `raw_audio_sec`
- `trimmed_audio_sec`
- `ttfs_ms`
- `chunk_gap_ms`
- `audio_out_sec`
- `audio_tokens`
- `tokens_per_sec`
- `tokens_per_sec_post_ttfs`
- `cancel_to_stop_ms`

### 8.1 Per-Step LLM Generation Profiling
In addition to turn-level `response.done.metrics`, you can profile each generation step to inspect jitter:

- `CHROMA_GEN_STEP_PROFILE=1`
- `CHROMA_GEN_STEP_PROFILE_INTERVAL=10`
- `CHROMA_GEN_STEP_PROFILE_CUDA_SYNC=1` (recommended on GPU)

Every N steps, logs include:
- `step_ms`: total latency per generation step
- `forward_ms`: backbone forward latency
- `sample_ms`: sampling latency
- `decoder_ms`: depth decoder generation latency
- `put_ms`: `streamer.put` handoff latency
- `step_p50` / `step_p95`: running median / p95 of step latency

At turn end, a `gen-step summary` line is emitted with overall step p50/p95 and stage-level p50 values.

## 9. Offline vs Streaming Alignment
Offline baseline script: `scripts/run_chroma.py`

Shared behavior:
- processor packing (conversation + prompt_audio/prompt_text)
- tensor device/dtype transfer
- `model.generate` sampling path
- decode-time token clamp

Streaming-only behavior:
- session state
- VAD turn logic
- cancellation and runtime metrics
- WebSocket event emission
