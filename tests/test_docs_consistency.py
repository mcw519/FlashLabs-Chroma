from __future__ import annotations

from pathlib import Path

from chroma.session_schema import SessionConfigV2


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_service_docs_include_turn_modes() -> None:
    en = _read("docs/service/STREAMING_SERVICE_SPEC.en.md")
    zh = _read("docs/service/STREAMING_SERVICE_SPEC.zh.md")
    assert "client_commit" in en
    assert "server_vad" in en
    assert "client_commit" in zh
    assert "server_vad" in zh


def test_service_docs_include_current_defaults() -> None:
    default = SessionConfigV2()
    en = _read("docs/service/STREAMING_SERVICE_SPEC.en.md")
    assert "output_chunk_sec" in en
    assert str(default.output_chunk_sec) in en
    assert "threshold" in en
    assert str(default.turn_detection.threshold) in en
