# Avoid Conda-related library conflicts by unsetting relevant environment variables
# run: env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run python scripts/run_chroma.py

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import colorlog
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoProcessor


def _configure_logging() -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s%(reset)s | %(message)s")
    )
    root_logger = colorlog.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_configure_logging()


def _resolve_local_model_path(local_path: str) -> str:
    path = Path(local_path)
    if (path / "config.json").is_file():
        return str(path)

    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        ref_path = path / "refs" / "main"
        if ref_path.is_file():
            snapshot = snapshots_dir / ref_path.read_text().strip()
            if (snapshot / "config.json").is_file():
                return str(snapshot)

        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if len(snapshot_dirs) == 1 and (snapshot_dirs[0] / "config.json").is_file():
            return str(snapshot_dirs[0])

        raise ValueError(
            "Local model path looks like a Hugging Face cache; pass the snapshot "
            "directory (e.g. .../snapshots/<hash>)."
        )

    return str(path)


def load_chroma_model(
    from_local_path: Optional[str] = None, *, use_half_precision: bool = True
):
    model_id = (
        _resolve_local_model_path(from_local_path)
        if from_local_path
        else "FlashLabs/Chroma-4B"
    )

    cache_dir = None
    if not from_local_path:
        repo_root = Path(__file__).resolve().parents[1]
        cache_dir = repo_root / "pretrained_models"
        cache_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float32
    if use_half_precision:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            logging.warning(
                "Half precision requested but CUDA is unavailable; using fp32."
            )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=str(cache_dir) if cache_dir else None,
        torch_dtype=torch_dtype,
    ).eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    return model, processor


@torch.no_grad()
def chroma_inference(
    model_path: str,
    input_audio: str,
    output_path: str,
    use_half_precision: bool,
    prompt_speaker: str = "scarlett_johansson",
):
    prompt_speaker_list = [
        "scarlett_johansson",
        "ariana_grande",
        "donald_trump",
        "lebron_james",
    ]
    if prompt_speaker not in prompt_speaker_list:
        raise ValueError(f"Invalid prompt speaker. Choose from: {prompt_speaker_list}")

    model, processor = load_chroma_model(
        from_local_path=model_path, use_half_precision=use_half_precision
    )
    logging.info("Model and processor loaded successfully.")

    # Construct conversation history
    system_prompt = (
        "You are Chroma, an advanced virtual human created by the FlashLabs. "
        "You possess the ability to understand auditory inputs and generate both text and speech."
    )
    conversation = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    # Input audio file path
                    {"type": "audio", "audio": input_audio},
                ],
            },
        ]
    ]

    # Provide reference audio/text for style or context
    def load_prompt(speaker_name):
        text_path = f"example/prompt_text/{speaker_name}.txt"
        audio_path = f"example/prompt_audio/{speaker_name}.wav"

        with open(text_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        return [prompt_text], [audio_path]

    prompt_text, prompt_audio = load_prompt(prompt_speaker)

    # Process inputs
    inputs = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
    )

    # Move inputs to device and match model dtype for floating tensors
    device = model.device
    model_dtype = model.dtype

    def _move_to_device(value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(device=device, dtype=model_dtype)
            return value.to(device=device)
        return value

    inputs = {k: _move_to_device(v) for k, v in inputs.items()}

    # 2. Generate (measure inference time)
    start_time = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    elapsed = time.perf_counter() - start_time
    logging.info(f"Model inference time: {elapsed:.3f}s")

    # 3. Decode Audio
    # The model outputs raw tokens; we decode the audio part using the codec
    audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values
    logging.info("Audio generation completed.")

    # Save audio output
    torchaudio.save(output_path, audio_values[0].float().cpu(), 24_000)
    logging.info(f"Generated audio saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chroma Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to local Chroma model (optional, will use Hugging Face if not provided)",
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to input audio file for the user query",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated audio output",
    )
    parser.add_argument(
        "--half_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable half-precision inference (default: enabled)",
    )
    parser.add_argument(
        "--prompt_speaker",
        type=str,
        default="scarlett_johansson",
        help="Reference speaker for style (default: scarlett_johansson)",
    )
    args = parser.parse_args()

    chroma_inference(
        args.model_path,
        args.input_audio,
        args.output_path,
        args.half_precision,
        args.prompt_speaker,
    )
