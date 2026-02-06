# Avoid Conda-related library conflicts by unsetting relevant environment variables
# run: env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run python scripts/run_realtime_webrtc.py

import argparse
import logging
import queue
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    ReplyOnPause,
    SileroVadOptions,
    Stream,
)
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers.generation.streamers import BaseStreamer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_chroma import load_chroma_model  # noqa: E402

PROMPT_SPEAKERS = [
    "scarlett_johansson",
    "ariana_grande",
    "donald_trump",
    "lebron_james",
]
SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_ICE_SERVER = "stun:stun.l.google.com:19302"


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio[0]
        return audio.mean(axis=0)
    return audio


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32, copy=False)


def _resample_audio(
    audio: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    audio = _normalize_audio(audio)
    if sample_rate == target_sample_rate:
        return audio
    audio_tensor = torch.from_numpy(audio)
    resampled = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return resampled.cpu().numpy()


@lru_cache(maxsize=1)
def _load_vad():
    return load_silero_vad()


def _trim_with_vad(audio_16k: np.ndarray) -> np.ndarray:
    vad_model = _load_vad()
    audio_tensor = torch.from_numpy(audio_16k)
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=INPUT_SAMPLE_RATE
    )
    if not speech_timestamps:
        return audio_16k
    start = speech_timestamps[0]["start"]
    end = speech_timestamps[-1]["end"]
    trimmed = audio_tensor[start:end]
    if trimmed.numel() == 0:
        return audio_16k
    return trimmed.cpu().numpy()


class ChromaRealtimeEngine:
    def __init__(
        self,
        model_path: str | None,
        use_half_precision: bool,
        max_new_tokens: int,
        max_text_new_tokens: int,
        temperature: float,
        top_p: float,
        output_chunk_sec: float,
        default_speaker: str,
        enable_text: bool,
    ) -> None:
        self.model, self.processor = load_chroma_model(
            from_local_path=model_path, use_half_precision=use_half_precision
        )
        self.model.eval()
        self.device = self.model.device
        self.model_dtype = self.model.dtype
        self.max_new_tokens = max_new_tokens
        self.max_text_new_tokens = max_text_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.output_chunk_sec = output_chunk_sec
        self.default_speaker = default_speaker
        self.enable_text = enable_text
        self._prompt_cache: dict[str, tuple[list[str], list[str]]] = {}

    def _load_prompt(self, speaker: str) -> tuple[list[str], list[str]]:
        if speaker in self._prompt_cache:
            return self._prompt_cache[speaker]
        text_path = REPO_ROOT / "example" / "prompt_text" / f"{speaker}.txt"
        audio_path = REPO_ROOT / "example" / "prompt_audio" / f"{speaker}.wav"
        prompt_text = text_path.read_text(encoding="utf-8")
        prompt_audio = str(audio_path)
        payload = ([prompt_text], [prompt_audio])
        self._prompt_cache[speaker] = payload
        return payload

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(device=self.device, dtype=self.model_dtype)
            return value.to(device=self.device)
        return value

    def _prepare_inputs(self, audio_np: np.ndarray, speaker: str):
        prompt_text, prompt_audio = self._load_prompt(speaker)
        conversation = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {"role": "user", "content": [{"type": "audio", "audio": audio_np}]},
            ]
        ]
        inputs = self.processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )
        return {k: self._move_to_device(v) for k, v in inputs.items()}

    @torch.no_grad()
    def _decode_audio(self, output: torch.Tensor) -> np.ndarray:
        audio_values = self.model.codec_model.decode(
            output.permute(0, 2, 1)
        ).audio_values
        audio_np = audio_values[0].detach().float().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        return audio_np

    @torch.no_grad()
    def _generate_text(self, inputs: dict[str, torch.Tensor]) -> str | None:
        if not self.enable_text:
            return None
        thinker_input_ids = inputs.get("thinker_input_ids")
        if thinker_input_ids is None:
            return None
        output_ids = self.model.thinker.generate(
            input_ids=thinker_input_ids,
            attention_mask=inputs.get("thinker_attention_mask"),
            input_features=inputs.get("thinker_input_features"),
            feature_attention_mask=inputs.get("thinker_feature_attention_mask"),
            max_new_tokens=self.max_text_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            use_cache=True,
        )
        prompt_len = thinker_input_ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        if generated_ids.numel() == 0:
            return None
        text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return text or None

    def _run_generation(
        self, inputs: dict[str, torch.Tensor], streamer: "_ChromaAudioStreamer"
    ) -> None:
        try:
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    use_cache=True,
                    streamer=streamer,
                )
        except Exception as exc:
            logging.exception("Audio generation failed")
            streamer.close_with_error(exc)

    def reply(
        self,
        audio: Tuple[int, np.ndarray] | object,
        webrtc_id: str,
        speaker: str,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        if hasattr(audio, "audio"):
            audio = getattr(audio, "audio")
        if speaker not in PROMPT_SPEAKERS:
            logging.warning(
                "Invalid speaker '%s' received; falling back to '%s'",
                speaker,
                self.default_speaker,
            )
            speaker = self.default_speaker
        sample_rate, audio_array = audio  # type: ignore[misc]
        logging.info(
            "Received audio webrtc_id=%s speaker=%s sr=%s samples=%s",
            webrtc_id,
            speaker,
            sample_rate,
            audio_array.shape[-1],
        )
        audio_mono = _ensure_mono(audio_array)
        audio_16k = _resample_audio(audio_mono, sample_rate, INPUT_SAMPLE_RATE)
        raw_seconds = audio_16k.shape[-1] / INPUT_SAMPLE_RATE
        audio_16k = _trim_with_vad(audio_16k)
        trimmed_seconds = audio_16k.shape[-1] / INPUT_SAMPLE_RATE
        logging.info(
            "Audio after VAD trim: %.2fs -> %.2fs", raw_seconds, trimmed_seconds
        )
        if audio_16k.size < INPUT_SAMPLE_RATE // 10:
            logging.info("Skipped short audio segment (%.2fs)", trimmed_seconds)
            return
        inputs = self._prepare_inputs(audio_16k, speaker)
        logging.info("Running text+audio generation...")
        text_output = self._generate_text(inputs)
        frame_rate = getattr(self.model.config.codec_config, "frame_rate", 12.5)
        frames_per_chunk = max(1, int(round(self.output_chunk_sec * frame_rate)))
        logging.info(
            "Streaming audio with %.2ffps, %s frames/chunk; text=%s",
            frame_rate,
            frames_per_chunk,
            "yes" if text_output else "no",
        )
        audio_queue: queue.SimpleQueue = queue.SimpleQueue()
        streamer = _ChromaAudioStreamer(
            model=self.model,
            frames_per_chunk=frames_per_chunk,
            audio_queue=audio_queue,
        )
        thread = threading.Thread(
            target=self._run_generation, args=(inputs, streamer), daemon=True
        )
        thread.start()
        return _AudioStreamIterator(audio_queue=audio_queue, text_output=text_output)


class _AudioStreamIterator:
    def __init__(
        self,
        audio_queue: queue.SimpleQueue,
        text_output: str | None,
    ) -> None:
        self.audio_queue = audio_queue
        self.text_output = text_output
        self._text_sent = False

    def __iter__(self) -> "_AudioStreamIterator":
        return self

    def __next__(self):
        item = self.audio_queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        audio_chunk = item
        if self.text_output and not self._text_sent:
            self._text_sent = True
            return audio_chunk, AdditionalOutputs(self.text_output)
        return audio_chunk


class _ChromaAudioStreamer(BaseStreamer):
    def __init__(
        self,
        model,
        frames_per_chunk: int,
        audio_queue: queue.SimpleQueue,
    ) -> None:
        self.model = model
        self.frames_per_chunk = frames_per_chunk
        self.audio_queue = audio_queue
        self.eos_token_id = model.config.codebook_eos_token_id
        self.num_codebooks = model.config.decoder_config.audio_num_codebooks
        self._buffer: list[torch.Tensor] = []
        self._closed = False

    def put(self, value) -> None:
        if self._closed:
            return
        tokens = torch.as_tensor(value)
        if tokens.ndim == 2:
            if tokens.shape[0] == 1 and tokens.shape[1] == self.num_codebooks:
                tokens = tokens[0]
            elif tokens.shape[1] == self.num_codebooks:
                for row in tokens:
                    self._put_frame(row)
                return
            else:
                return
        if tokens.ndim != 1:
            return
        if tokens.numel() != self.num_codebooks:
            return
        self._put_frame(tokens)

    def _put_frame(self, tokens: torch.Tensor) -> None:
        if (tokens == self.eos_token_id).all():
            self.end()
            return
        self._buffer.append(tokens)
        if len(self._buffer) >= self.frames_per_chunk:
            self._emit_frames(self.frames_per_chunk)

    def end(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._buffer:
            self._emit_frames(len(self._buffer))
            self._buffer.clear()
        self.audio_queue.put(None)

    def close_with_error(self, exc: Exception) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_queue.put(exc)
        self.audio_queue.put(None)

    def _emit_frames(self, count: int) -> None:
        frames = self._buffer[:count]
        del self._buffer[:count]
        audio_np = self._decode_frames(frames)
        if audio_np.size == 0:
            return
        self.audio_queue.put((OUTPUT_SAMPLE_RATE, audio_np))

    @torch.no_grad()
    def _decode_frames(self, frames: list[torch.Tensor]) -> np.ndarray:
        audio_codes = torch.stack(frames).to(self.model.device)
        audio_values = self.model.codec_model.decode(
            audio_codes.transpose(0, 1).unsqueeze(0)
        ).audio_values
        audio_np = audio_values[0].detach().float().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        return audio_np


def _build_rtc_configuration(ice_servers: list[str]) -> dict[str, Any]:
    return {"iceServers": [{"urls": ice_servers}]}


def _update_text_output(_current_text: str, new_text: str) -> str:
    return new_text


def build_stream(
    engine: ChromaRealtimeEngine,
    default_speaker: str,
    rtc_configuration: dict[str, Any] | None,
) -> Stream:
    speaker_dropdown = gr.Dropdown(
        choices=PROMPT_SPEAKERS,
        value=default_speaker,
        label="Prompt Speaker",
    )
    text_output = gr.Textbox(
        label="Chroma Text",
        lines=4,
        interactive=False,
    )
    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.2,
        speech_threshold=0.2,
    )
    model_options = SileroVadOptions(
        threshold=0.6,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
    )
    handler = ReplyOnPause(
        engine.reply,
        algo_options=algo_options,
        model_options=model_options,
        input_sample_rate=INPUT_SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
        expected_layout="mono",
        can_interrupt=False,
    )
    return Stream(
        handler=handler,
        modality="audio",
        mode="send-receive",
        additional_inputs=[speaker_dropdown],
        additional_outputs=[text_output],
        additional_outputs_handler=_update_text_output,
        rtc_configuration=rtc_configuration,
        server_rtc_configuration=rtc_configuration,
        ui_args={"title": "Chroma Realtime WebRTC"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chroma realtime WebRTC demo")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt-speaker", type=str, default="scarlett_johansson")
    parser.add_argument(
        "--max-new-tokens", type=int, default=200
    )  # output audio length control * 0.08 s
    parser.add_argument("--max-text-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)  # default: 0.7
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-chunk-sec", type=float, default=0.5)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ice-server", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    if args.prompt_speaker not in PROMPT_SPEAKERS:
        raise ValueError(f"Invalid prompt speaker. Choose from: {PROMPT_SPEAKERS}")
    logging.info("Loading Chroma model and processor...")
    engine = ChromaRealtimeEngine(
        model_path=args.model_path,
        use_half_precision=args.use_half_precision,
        max_new_tokens=args.max_new_tokens,
        max_text_new_tokens=args.max_text_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_chunk_sec=args.output_chunk_sec,
        default_speaker=args.prompt_speaker,
        enable_text=not args.disable_text,
    )
    ice_servers = args.ice_server or [DEFAULT_ICE_SERVER]
    rtc_configuration = _build_rtc_configuration(ice_servers)
    stream = build_stream(engine, args.prompt_speaker, rtc_configuration)
    stream.ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
