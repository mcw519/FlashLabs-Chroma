from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

TextMode = Literal["none", "sentence", "final"]
TurnDetectionMode = Literal["client_commit", "server_vad"]


@dataclass(slots=True)
class TurnDetectionConfig:
    mode: TurnDetectionMode = "server_vad"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    speech_pad_ms: int = 200


@dataclass(slots=True)
class SessionConfigV2:
    speaker: str = "scarlett_johansson"
    memory_turns: int = 6
    output_chunk_sec: float = 0.24
    text_mode: TextMode = "final" # "sentence"
    include_transcript_in_query: bool = False
    system_prompt: str | None = None
    trim_with_vad: bool = False
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)


def session_config_to_dict(config: SessionConfigV2) -> dict[str, Any]:
    return asdict(config)


def session_config_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    base: SessionConfigV2 | None = None,
) -> SessionConfigV2:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("Field 'config' must be an object")

    base_config = base or SessionConfigV2()

    td_payload = payload.get("turn_detection", {})
    if td_payload is None:
        td_payload = {}
    if not isinstance(td_payload, Mapping):
        raise ValueError("Field 'config.turn_detection' must be an object")

    if "system_prompt" in payload:
        system_prompt = payload.get("system_prompt")
    else:
        system_prompt = base_config.system_prompt

    return SessionConfigV2(
        speaker=payload.get("speaker", base_config.speaker),
        memory_turns=payload.get("memory_turns", base_config.memory_turns),
        output_chunk_sec=payload.get("output_chunk_sec", base_config.output_chunk_sec),
        text_mode=payload.get("text_mode", base_config.text_mode),
        include_transcript_in_query=payload.get(
            "include_transcript_in_query",
            base_config.include_transcript_in_query,
        ),
        system_prompt=system_prompt,
        trim_with_vad=payload.get("trim_with_vad", base_config.trim_with_vad),
        turn_detection=TurnDetectionConfig(
            mode=td_payload.get("mode", base_config.turn_detection.mode),
            threshold=td_payload.get("threshold", base_config.turn_detection.threshold),
            min_speech_ms=td_payload.get(
                "min_speech_ms",
                base_config.turn_detection.min_speech_ms,
            ),
            min_silence_ms=td_payload.get(
                "min_silence_ms",
                base_config.turn_detection.min_silence_ms,
            ),
            speech_pad_ms=td_payload.get(
                "speech_pad_ms",
                base_config.turn_detection.speech_pad_ms,
            ),
        ),
    )


__all__ = [
    "SessionConfigV2",
    "TextMode",
    "TurnDetectionConfig",
    "TurnDetectionMode",
    "session_config_from_payload",
    "session_config_to_dict",
]
