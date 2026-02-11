from __future__ import annotations

from pathlib import Path

from chroma import pretrained


def test_get_pretrained_cache_dir_uses_unified_repo_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pretrained, "get_repo_root", lambda: tmp_path)

    cache_dir = pretrained.get_pretrained_cache_dir()

    assert cache_dir == tmp_path / pretrained.PRETRAINED_DIR_NAME
    assert cache_dir.is_dir()


def test_resolve_model_id_and_cache_dir_for_remote_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pretrained, "get_repo_root", lambda: tmp_path)

    model_id, cache_dir = pretrained.resolve_model_id_and_cache_dir(
        None,
        default_model_id=pretrained.DEFAULT_CHROMA_MODEL_ID,
    )

    assert model_id == pretrained.DEFAULT_CHROMA_MODEL_ID
    assert cache_dir == str(tmp_path / pretrained.PRETRAINED_DIR_NAME)


def test_resolve_model_id_and_cache_dir_for_local_snapshot(tmp_path) -> None:
    local_model = tmp_path / "local_model"
    local_model.mkdir(parents=True, exist_ok=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")

    model_id, cache_dir = pretrained.resolve_model_id_and_cache_dir(
        str(local_model),
        default_model_id=pretrained.DEFAULT_CHROMA_MODEL_ID,
    )

    assert model_id == str(local_model)
    assert cache_dir is None


def test_resolve_local_model_path_from_hf_cache_ref(tmp_path) -> None:
    cache_root = tmp_path / "models--FlashLabs--Chroma-4B"
    snapshot = cache_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    refs = cache_root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text("abc123", encoding="utf-8")

    resolved = pretrained.resolve_local_model_path(str(cache_root))

    assert resolved == str(snapshot)


def test_download_model_snapshot_uses_unified_cache(tmp_path, monkeypatch) -> None:
    called: dict[str, str] = {}

    def _fake_snapshot_download(*, repo_id: str, cache_dir: str) -> str:
        called["repo_id"] = repo_id
        called["cache_dir"] = cache_dir
        return str(Path(cache_dir) / "models--fake")

    monkeypatch.setattr(pretrained, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(pretrained, "snapshot_download", _fake_snapshot_download)

    output = pretrained.download_model_snapshot("openai/whisper-small")

    assert output.endswith("models--fake")
    assert called["repo_id"] == "openai/whisper-small"
    assert called["cache_dir"] == str(tmp_path / pretrained.PRETRAINED_DIR_NAME)
