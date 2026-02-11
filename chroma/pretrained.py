from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_CHROMA_MODEL_ID = "FlashLabs/Chroma-4B"
DEFAULT_WHISPER_MODEL_ID = "openai/whisper-small"
PRETRAINED_DIR_NAME = "pretrained_models"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_pretrained_cache_dir() -> Path:
    cache_dir = get_repo_root() / PRETRAINED_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_local_model_path(local_path: str) -> str:
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


def resolve_model_id_and_cache_dir(
    from_local_path: str | None,
    *,
    default_model_id: str,
) -> tuple[str, str | None]:
    if from_local_path:
        return resolve_local_model_path(from_local_path), None
    return default_model_id, str(get_pretrained_cache_dir())


def download_model_snapshot(model_id: str) -> str:
    cache_dir = get_pretrained_cache_dir()
    return snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
