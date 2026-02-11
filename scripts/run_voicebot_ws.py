# Avoid Conda-related library conflicts by unsetting relevant environment variables
# run: env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run python scripts/run_voicebot_ws.py

from __future__ import annotations

import argparse
import asyncio

from chroma.engine import StreamingVoicebotEngine
from chroma.obs_logging import configure_root_logger
from chroma.transport import VoicebotWebSocketServer
from chroma.transport.server_asr import build_server_asr_transcriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chroma streaming voicebot WebSocket server"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--bot-config",
        type=str,
        default=None,
        help="Path to bot config (.json/.toml) with system_prompt",
    )
    parser.add_argument("--prompt-speaker", type=str, default="scarlett_johansson")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--max-text-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-chunk-sec", type=float, default=0.24)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-message-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--server-asr-model",
        type=str,
        default="",
        help="Optional server-side ASR model id/path (e.g. openai/whisper-small)",
    )
    parser.add_argument(
        "--server-asr-language",
        type=str,
        default="",
        help="Optional ASR language hint (e.g. zh, en)",
    )
    parser.add_argument(
        "--server-asr-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for server-side ASR",
    )
    parser.add_argument(
        "--server-asr-timeout-sec",
        type=float,
        default=1.2,
        help="Timeout for server-side ASR per committed turn",
    )
    return parser.parse_args()


def main() -> None:
    configure_root_logger(component="ws_server", default_level="INFO")
    args = parse_args()

    engine = StreamingVoicebotEngine(
        model_path=args.model_path,
        use_half_precision=args.use_half_precision,
        max_new_tokens=args.max_new_tokens,
        max_text_new_tokens=args.max_text_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_chunk_sec=args.output_chunk_sec,
        default_speaker=args.prompt_speaker,
        enable_text=not args.disable_text,
        max_sessions=args.max_sessions,
        warmup=args.warmup,
        bot_config_path=args.bot_config,
    )

    asr_transcriber = build_server_asr_transcriber(
        model_id=args.server_asr_model,
        language=args.server_asr_language,
        device=args.server_asr_device,
    )

    server = VoicebotWebSocketServer(
        engine=engine,
        host=args.host,
        port=args.port,
        max_message_size=args.max_message_size,
        asr_transcriber=asr_transcriber,
        server_asr_timeout_sec=args.server_asr_timeout_sec,
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
