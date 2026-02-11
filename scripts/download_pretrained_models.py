from __future__ import annotations

import argparse
import logging

from chroma.pretrained import (
    DEFAULT_CHROMA_MODEL_ID,
    DEFAULT_WHISPER_MODEL_ID,
    get_pretrained_cache_dir,
    download_model_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all pretrained models into a unified local cache directory"
    )
    parser.add_argument(
        "--chroma-model",
        type=str,
        default=DEFAULT_CHROMA_MODEL_ID,
        help=f"Chroma model ID (default: {DEFAULT_CHROMA_MODEL_ID})",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL_ID,
        help=f"Whisper model ID (default: {DEFAULT_WHISPER_MODEL_ID})",
    )
    parser.add_argument(
        "--skip-chroma",
        action="store_true",
        help="Skip downloading the Chroma model",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip downloading the Whisper ASR model",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cache_dir = get_pretrained_cache_dir()
    logging.info("Unified pretrained cache: %s", cache_dir)

    if not args.skip_chroma:
        logging.info("Downloading Chroma model: %s", args.chroma_model)
        local_path = download_model_snapshot(args.chroma_model)
        logging.info("Chroma model ready at: %s", local_path)

    if not args.skip_whisper and args.whisper_model.strip():
        whisper_model = args.whisper_model.strip()
        logging.info("Downloading Whisper model: %s", whisper_model)
        local_path = download_model_snapshot(whisper_model)
        logging.info("Whisper model ready at: %s", local_path)

    if args.skip_chroma and args.skip_whisper:
        logging.warning("Nothing to download; both --skip-chroma and --skip-whisper were set.")


if __name__ == "__main__":
    main()
