from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TextMode = Literal["none", "sentence", "final"]


@dataclass(slots=True)
class SessionConfig:
    speaker: str = "scarlett_johansson"
    memory_turns: int = 6
    output_chunk_sec: float = 0.24
    text_mode: TextMode = "sentence"


@dataclass(slots=True)
class ResponseStartedEvent:
    session_id: str
    turn_id: str
    type: Literal["response.started"] = "response.started"


@dataclass(slots=True)
class ResponseAudioDeltaEvent:
    session_id: str
    turn_id: str
    audio_b64: str
    sample_rate: int
    channels: int = 1
    mime_type: Literal["audio/pcm;rate=24000;encoding=s16le"] = (
        "audio/pcm;rate=24000;encoding=s16le"
    )
    type: Literal["response.audio.delta"] = "response.audio.delta"


@dataclass(slots=True)
class ResponseTextDeltaEvent:
    session_id: str
    turn_id: str
    text: str
    type: Literal["response.text.delta"] = "response.text.delta"


@dataclass(slots=True)
class ResponseDoneEvent:
    session_id: str
    turn_id: str
    metrics: dict[str, float]
    text: str | None = None
    type: Literal["response.done"] = "response.done"


@dataclass(slots=True)
class ResponseCancelledEvent:
    session_id: str
    turn_id: str
    reason: str = "barge_in"
    type: Literal["response.cancelled"] = "response.cancelled"


@dataclass(slots=True)
class ErrorEvent:
    session_id: str | None
    code: str
    message: str
    details: dict[str, Any] | None = None
    type: Literal["error"] = "error"


EngineEvent = (
    ResponseStartedEvent
    | ResponseAudioDeltaEvent
    | ResponseTextDeltaEvent
    | ResponseDoneEvent
    | ResponseCancelledEvent
    | ErrorEvent
)


def event_to_dict(event: EngineEvent) -> dict[str, Any]:
    return asdict(event)


__all__ = [
    "EngineEvent",
    "ErrorEvent",
    "ResponseAudioDeltaEvent",
    "ResponseCancelledEvent",
    "ResponseDoneEvent",
    "ResponseStartedEvent",
    "ResponseTextDeltaEvent",
    "SessionConfig",
    "TextMode",
    "event_to_dict",
]
