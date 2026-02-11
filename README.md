<div align="center">

# FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning

</div>

<br>
<div align="center">
<img src="figures/logo.svg" alt="FlashLabs Chroma Logo" width="400px"/>

<h3>🚀 Get Started with <a href="https://www.flashlabs.ai/flashai-voice-agents">FlashAI Voice Agents</a>!</h3>

<p><strong>Production-ready voice AI solutions</strong> powered by Chroma | <strong>Open-source model</strong> for developers & researchers</p>

[![Deploy AI Voice Agents](https://img.shields.io/badge/🎯%20Voice%20Agents-blue?style=for-the-badge)](https://www.flashlabs.ai/flashai-voice-agents)
[![Download Model](https://img.shields.io/badge/🤗%20Download%20Model-orange?style=for-the-badge)](https://huggingface.co/FlashLabs/Chroma-4B)
[![Technical Report](https://img.shields.io/badge/📄%20Technical%20Report-red?style=for-the-badge)](https://arxiv.org/abs/2601.11141)
[![X (Twitter)](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+WDwvdGl0bGU+PHBhdGggZD0iTTE4LjkwMSAxLjE1M2gzLjY4bC04LjA0IDkuMTlMMjQgMjIuODQ2aC03LjQwNmwtNS44LTcuNTg0LTYuNjM4IDcuNTg0SC40NzRsOC42LTkuODNMMCAxLjE1NGg3LjU5NGw1LjI0MyA2LjkzMlpNMTcuNjEgMjAuNjQ0aDIuMDM5TDYuNDg2IDMuMjRINC4yOThaIiBmaWxsPSIjZmZmIi8+PC9zdmc+&logoColor=white)](https://x.com/flashlabsdotai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+TGlua2VkSW48L3RpdGxlPjxwYXRoIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjQgMi4wNjMtMi4wNjQgMS4xNDMgMCAyLjA2NC45MjYgMi4wNjQgMi4wNjQgMCAxLjEzOS0uOTI1IDIuMDY1LTIuMDY0IDIuMDY1em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiIGZpbGw9IiNmZmYiLz48L3N2Zz4=&logoColor=white)](https://www.linkedin.com/company/flashlabs-ai/)

</div>

## 🎬 Demo

<div align="center">

https://github.com/user-attachments/assets/3723c24d-d262-4c3e-88ee-34a16546359e

<p><i>Watch our model in action</i></p>
</div>


## Model Description

**Chroma 1.0** is an advanced multimodal model developed by **[FlashLabs](https://flashlabs.ai)**. It is designed to understand and generate content across multiple modalities, including text and audio. As a virtual human model, Chroma possesses the ability to process auditory inputs and respond with both text and synthesized speech, enabling natural voice interactions.

- **Model Type:** Multimodal Causal Language Model
- **Developed by:** FlashLabs
- **Language(s):** English
- **License:** Apache-2.0
- **Model Architecture:** 
  - **Reasoner:** Based on Qwen2.5-Omni-3B
  - **Backbone:** Based on Llama3 (16 layers, 2048 hidden size)
  - **Decoder:** Based on Llama3 (4 layers, 1024 hidden size)
  - **Codec:** Mimi (24kHz sampling rate)

## Model Architecture

<img src="figures/architecture.png" alt="Model Architecture" width="800" />


## Capabilities

Chroma 1.0 is capable of:
- **Speech Understanding:** Processing user audio input directly.
- **Multimodal Generation:** Generating coherent text and speech responses simultaneously.
- **Voice Cloning:** utilizing reference audio prompts to guide speech generation style.

## Usage

### Installation

#### Requirements
- Python 3.11 or higher
- CUDA 12.6 or compatible version (for GPU support)

#### Quick Install

```bash
# Clone the repository
git clone https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma.git
cd FlashLabs-Chroma

# Install uv (once)
pip install uv

# Optional: Create a new virtual environment with Python 3.11
uv venv -p 3.11
source .venv/bin/activate

# Install dependencies in the correct order
uv sync

```

### Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "FlashLabs/Chroma-4B" # Or local path

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

### Inference Example

Here is how to perform a simple conversation with audio input and audio output:

```python
import torch
from IPython.display import Audio



# Construct conversation history
system_prompt = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
conversation = [[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ],
    },
    {
        "role": "user",
        "content": [
            # Input audio file path
            {"type": "audio", "audio": "example/make_taco.wav"}, 
        ],
    },
]]

# Provide reference audio/text for style or context
def load_prompt(speaker_name):
    text_path = f"example/prompt_text/{speaker_name}.txt"
    audio_path = f"example/prompt_audio/{speaker_name}.wav"
    
    with open(text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    return [prompt_text], [audio_path]

prompt_text, prompt_audio = load_prompt("scarlett_johansson")

# Process inputs
inputs = processor(
    conversation,
    add_generation_prompt=True, 
    tokenize=False,
    prompt_audio=prompt_audio,
    prompt_text=prompt_text
)

# Move inputs to device
device = model.device
inputs = {k: v.to(device) for k, v in inputs.items()}

# 2. Generate
output = model.generate(
    **inputs, 
    max_new_tokens=1000, 
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    use_cache=True
)

# 3. Decode Audio
# The model outputs raw tokens; we decode the audio part using the codec
audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values

# Save or play audio (e.g., in Jupyter)
Audio(audio_values[0].cpu().detach().numpy(), rate=24_000)
```

### Realtime WebRTC Demo (FastRTC + Gradio)

This demo streams audio via WebRTC using FastRTC with Silero-VAD turn-taking and a Gradio UI.
Audio replies are emitted incrementally as the model generates; `--output-chunk-sec` controls chunk cadence.

Handler I/O schema (send-receive audio mode):
```python
def reply(
    audio: tuple[int, np.ndarray],  # (sample_rate, mono waveform) from mic
    webrtc_id: str,                 # session id
    speaker: str,                   # dropdown value
) -> Iterator[tuple[int, np.ndarray]]:  # yields (sample_rate, mono waveform)
    ...
```

```bash
# Install realtime demo dependencies (FastRTC + Gradio + Silero-VAD)
uv sync

# Launch the WebRTC UI
python scripts/run_realtime_webrtc.py \
  --use-half-precision \
  --prompt-speaker scarlett_johansson \
  --bot-config example/bot_configs/support_agent.toml
```

Then open http://localhost:7860 and speak into the mic to receive audio replies.

If running on a remote host/NAT, provide ICE servers (STUN/TURN) for WebRTC:
```bash
python scripts/run_realtime_webrtc.py \
  --ice-server stun:stun.l.google.com:19302
```

To disable text generation (audio only):
```bash
python scripts/run_realtime_webrtc.py --disable-text
```

### Streaming Voicebot WebSocket Server

The new voicebot engine provides a reusable streaming backend with:
- Session lifecycle (`session.start`, `session.update`, `session.end`)
- Audio streaming input (`audio.append`, base64 PCM16 @ 16kHz)
- Incremental response output (`response.audio.delta`, base64 PCM16 @ 24kHz)
- Turn control (`audio.commit`, `response.cancel`)

Run the server:

```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --bot-config example/bot_configs/support_agent.toml \
  --prompt-speaker scarlett_johansson \
  --host 0.0.0.0 \
  --port 8765
```

Bot config supports `.toml`/`.json` and can override the startup system prompt:

```toml
system_prompt = """
You are Chroma, a bilingual customer support voice assistant.
Keep answers concise and action-oriented.
"""
```

Enable server-side ASR (client does not need to send transcript):
```bash
python scripts/run_voicebot_ws.py \
  --use-half-precision \
  --server-asr-model openai/whisper-small \
  --server-asr-language zh \
  --server-asr-device auto
```

Minimal client event flow:
```json
{"type":"session.start","session_id":"demo-1","config":{"include_transcript_in_query":false}}
{"type":"audio.append","session_id":"demo-1","audio_b64":"..."}
{"type":"audio.commit","session_id":"demo-1","transcript":"optional user text"}
```

Per-session persona override:
```json
{"type":"session.start","session_id":"demo-2","config":{"system_prompt":"You are a concise travel concierge."}}
{"type":"session.update","session_id":"demo-2","config":{"system_prompt":"You are a strict interview coach."}}
{"type":"session.update","session_id":"demo-2","config":{"system_prompt":null}}
```

`include_transcript_in_query` defaults to `false` (current behavior): transcript is stored in memory only.
Set it to `true` to include transcript in the same turn's user query (`text + audio`).
When `--server-asr-model` is enabled on server, `audio.commit.transcript` from client is ignored.

Microphone streaming client (local mic -> WebSocket):
```bash
python -m pip install sounddevice websockets
python scripts/run_voicebot_ws_mic_client.py \
  --url ws://127.0.0.1:8765 \
  --speaker scarlett_johansson
```

The mic client auto-detects speech and sends `audio.commit` on pause, and will send
`response.cancel` automatically when barge-in is detected during model playback.
At startup it prompts for microphone/speaker device selection, and you can type `/quit`
to disconnect gracefully.

Detailed guide: `docs/VOICEBOT_WS_MIC_CLIENT.md`
Streaming runtime internals: `docs/CHROMA_STREAMING_RUNTIME.md`

## Troubleshooting

### Common Issues

#### TypeError: argument of type 'NoneType' is not iterable

**Problem:** This error occurs when loading the processor if `torchvision` is not properly detected during `transformers` module initialization.

**Solution:**
1. Ensure you install packages in the correct order (PyTorch/torchvision before transformers)
2. If you already installed them, reinstall in the correct order:
   ```bash
   uv pip uninstall transformers torchvision torch -y
   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
   uv pip install transformers==5.0.0rc0
   ```
3. Restart your Python kernel/session after reinstalling

#### CUDA out of memory

**Solution:** Use `device_map="auto"` when loading the model, or specify GPU device explicitly:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16  # Use bfloat16 to reduce memory usage
)
```

## Citation

If you use Chroma in your research, please cite:

```bibtex
@misc{chen2026flashlabschroma10realtime,
      title={FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning}, 
      author={Tanyu Chen and Tairan Chen and Kai Shen and Zhenghua Bao and Zhihui Zhang and Man Yuan and Yi Shi},
      year={2026},
      eprint={2601.11141},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.11141}, 
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 FlashLabs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contact

For questions or issues, please contact: chroma@flashlabs.ai
