from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from chroma.transport import server_asr


def test_asr_pipeline_uses_repo_pretrained_cache(monkeypatch) -> None:
    called: dict[str, object] = {}

    class _FakePipeline:
        def __call__(self, *args, **kwargs):
            return {"text": "ok"}

    def _fake_hf_pipeline(**kwargs):
        called.update(kwargs)
        return _FakePipeline()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(pipeline=_fake_hf_pipeline),
    )

    transcriber = server_asr.TransformersASRTranscriber(
        server_asr.ServerASRConfig(model_id="openai/whisper-small", device="cpu")
    )

    assert transcriber is not None
    expected_cache_dir = Path(server_asr.__file__).resolve().parents[2] / "pretrained_models"
    assert called["task"] == "automatic-speech-recognition"
    assert called["model"] == "openai/whisper-small"
    assert called["model_kwargs"] == {"cache_dir": str(expected_cache_dir)}
    assert expected_cache_dir.is_dir()
