# Chroma S2S Pipeline (Streaming Runtime)

## 1. Model Architecture Snapshot
- Reasoner: Qwen2.5-Omni-3B based thinker
- Backbone: Llama3-based causal backbone
- Decoder: Llama3-based audio codebook decoder
- Codec: Mimi 24kHz waveform codec

## 2. Reference Baseline
The streaming runtime is aligned with the single-shot baseline in:
- `scripts/run_chroma.py`

Alignment points:
1. Conversation packing (`system + user(audio)` plus optional transcript text)
2. `processor(..., add_generation_prompt=True, tokenize=False, prompt_audio, prompt_text)`
3. Tensor device/dtype move
4. `model.generate` sampling path
5. Codec decode clamp (`min(2047, vocab_size-1)`)

## 3. Streaming Turn Pipeline
1. Buffer PCM16 16k mono via `input.audio.append`
2. On commit:
  - decode PCM16 -> float32
  - optional VAD trim (`trim_with_vad`)
  - build multimodal inputs with memory context
3. Run generation with streamer:
  - frame tokens -> codec decode -> PCM16 24k chunks
  - emit `response.audio.delta`
4. Optional text branch:
  - thinker text generation
  - emit `response.text.delta`
5. Finalize:
  - emit `response.done` with metrics
  - or `response.cancelled` on cancel

## 4. Metrics
`response.done.metrics` includes:
- `raw_audio_sec`
- `trimmed_audio_sec`
- `ttfs_ms`
- `chunk_gap_ms`
- `audio_out_sec`
- `audio_tokens`
- `tokens_per_sec`
- `tokens_per_sec_post_ttfs`
- `cancel_to_stop_ms`
