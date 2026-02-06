# Avoid Conda-related library conflicts by unsetting relevant environment variables
# run: env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run python scripts/run_voicebot_ws.py

from __future__ import annotations

import argparse
import asyncio
import logging

from chroma.engine import StreamingVoicebotEngine
from chroma.transport import VoicebotWebSocketServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chroma streaming voicebot WebSocket server"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt-speaker", type=str, default="scarlett_johansson")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-text-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-chunk-sec", type=float, default=0.5)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-message-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
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
    )

    server = VoicebotWebSocketServer(
        engine=engine,
        host=args.host,
        port=args.port,
        max_message_size=args.max_message_size,
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
