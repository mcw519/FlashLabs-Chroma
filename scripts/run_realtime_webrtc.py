# Avoid Conda-related library conflicts by unsetting relevant environment variables
# run: env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run python scripts/run_realtime_webrtc.py

from __future__ import annotations

import argparse
import base64
import logging
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

from chroma.engine import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    PROMPT_SPEAKERS,
    SessionConfig,
    StreamingVoicebotEngine,
    float32_to_pcm16le_bytes,
    pcm16le_bytes_to_float32_mono,
)

SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
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


def _resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    audio = _normalize_audio(audio)
    if sample_rate == target_sample_rate:
        return audio
    audio_tensor = torch.from_numpy(audio)
    resampled = torchaudio.functional.resample(
        audio_tensor,
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
    )
    return resampled.cpu().numpy()


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
        self.enable_text = enable_text
        self.output_chunk_sec = output_chunk_sec
        self.default_speaker = default_speaker
        self.engine = StreamingVoicebotEngine(
            model_path=model_path,
            use_half_precision=use_half_precision,
            max_new_tokens=max_new_tokens,
            max_text_new_tokens=max_text_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_chunk_sec=output_chunk_sec,
            default_speaker=default_speaker,
            enable_text=enable_text,
            max_sessions=3,
        )

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
                "Invalid speaker '%s' received; fallback to '%s'",
                speaker,
                self.default_speaker,
            )
            speaker = self.default_speaker

        sample_rate, audio_array = audio  # type: ignore[misc]
        audio_mono = _ensure_mono(audio_array)
        audio_16k = _resample_audio(audio_mono, sample_rate, INPUT_SAMPLE_RATE)
        pcm_bytes = float32_to_pcm16le_bytes(audio_16k)

        self.engine.create_session(
            webrtc_id,
            SessionConfig(
                speaker=speaker,
                memory_turns=6,
                output_chunk_sec=self.output_chunk_sec,
                text_mode="sentence" if self.enable_text else "none",
            ),
        )
        self.engine.append_audio(webrtc_id, pcm_bytes)

        pending_text: str | None = None
        text_sent = False

        for event in self.engine.commit_turn_sync(webrtc_id):
            event_type = getattr(event, "type", None)
            if event_type == "response.text.delta":
                pending_text = event.text
                continue
            if event_type == "response.audio.delta":
                audio_chunk = pcm16le_bytes_to_float32_mono(
                    base64.b64decode(event.audio_b64)
                )[None, :]
                if pending_text and not text_sent:
                    text_sent = True
                    yield (OUTPUT_SAMPLE_RATE, audio_chunk), AdditionalOutputs(pending_text)
                else:
                    yield OUTPUT_SAMPLE_RATE, audio_chunk
                continue
            if event_type == "response.done":
                if pending_text and not text_sent:
                    text_sent = True
                    tiny_silence = np.zeros((1, 1), dtype=np.float32)
                    yield (OUTPUT_SAMPLE_RATE, tiny_silence), AdditionalOutputs(pending_text)
                continue
            if event_type == "error":
                logging.error("Engine error: %s - %s", event.code, event.message)
                continue


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
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-text-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
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
