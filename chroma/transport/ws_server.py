from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import websockets
from websockets.exceptions import ConnectionClosed

from chroma.engine.streaming_engine import StreamingVoicebotEngine
from chroma.engine.types import ErrorEvent, event_to_dict
from chroma.obs_logging import configure_component_logger
from chroma.session_schema import session_config_from_payload, session_config_to_dict

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False


def _clip_text(text: str | None, max_chars: int = 220) -> str:
    if not text:
        return ""
    clipped = text.strip().replace("\n", " ")
    if len(clipped) <= max_chars:
        return clipped
    return clipped[: max_chars - 3] + "..."


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _today_folder() -> str:
    return datetime.now().astimezone().date().isoformat()


def _safe_session_dirname(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", session_id).strip("._")
    return cleaned or "session"


def _write_pcm16_wav(path: Path, *, sample_rate: int, audio_bytes: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


class _SessionLogWriter:
    def __init__(self, root_dir: Path, session_id: str):
        self._root_dir = root_dir
        self._session_id = session_id
        self._safe_session_id = _safe_session_dirname(session_id)
        self._session_dir = self._root_dir / _today_folder() / self._safe_session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._conversation_path = self._session_dir / "conversation_log.json"
        self._conversation: dict[str, Any] = self._load_or_init()
        self._turn_index = self._resolve_next_turn_index()
        if self._safe_session_id != self._session_id:
            logger.warning(
                "session log path sanitized session_id=%s safe_session_id=%s",
                self._session_id,
                self._safe_session_id,
            )

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def record_turn(
        self,
        *,
        source: str,
        turn_id: str | None,
        user_audio_bytes: bytes,
        user_text: str | None,
        user_text_source: str,
        assistant_audio_bytes: bytes,
        assistant_text: str | None,
        assistant_event: str,
        metrics: dict[str, float],
        error: dict[str, Any] | None = None,
    ) -> None:
        self._turn_index += 1
        turn_index = self._turn_index
        user_audio_name = f"user_{turn_index:04d}.wav"
        user_audio_path = self._session_dir / user_audio_name
        _write_pcm16_wav(user_audio_path, sample_rate=16000, audio_bytes=user_audio_bytes)

        assistant_audio_name: str | None = None
        if assistant_audio_bytes:
            assistant_audio_name = f"bot_{turn_index:04d}.wav"
            assistant_audio_path = self._session_dir / assistant_audio_name
            _write_pcm16_wav(
                assistant_audio_path,
                sample_rate=24000,
                audio_bytes=assistant_audio_bytes,
            )

        entry: dict[str, Any] = {
            "turn_index": turn_index,
            "turn_id": turn_id,
            "source": source,
            "created_at": _now_iso(),
            "user": {
                "text": user_text,
                "text_source": user_text_source,
                "audio_file": user_audio_name,
                "audio_bytes": len(user_audio_bytes),
            },
            "assistant": {
                "event": assistant_event,
                "text": assistant_text,
                "audio_file": assistant_audio_name,
                "audio_bytes": len(assistant_audio_bytes),
            },
            "metrics": metrics,
        }
        if error is not None:
            entry["error"] = error

        turns = self._conversation.setdefault("turns", [])
        if not isinstance(turns, list):
            turns = []
            self._conversation["turns"] = turns
        turns.append(entry)
        self._conversation["updated_at"] = _now_iso()
        self.flush()

    def flush(self) -> None:
        self._conversation_path.write_text(
            json.dumps(self._conversation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_or_init(self) -> dict[str, Any]:
        if self._conversation_path.exists():
            try:
                loaded = json.loads(self._conversation_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(
                    "session log read failed path=%s", self._conversation_path
                )
            else:
                if isinstance(loaded, dict):
                    return loaded
        now = _now_iso()
        initial = {
            "session_id": self._session_id,
            "safe_session_id": self._safe_session_id,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        self._conversation_path.write_text(
            json.dumps(initial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return initial

    def _resolve_next_turn_index(self) -> int:
        turns = self._conversation.get("turns", [])
        if not isinstance(turns, list) or not turns:
            return 0
        max_index = 0
        for item in turns:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("turn_index")
            if isinstance(raw_index, int):
                max_index = max(max_index, raw_index)
        return max_index


class RequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AudioTranscriber(Protocol):
    def transcribe_pcm16_16k(self, pcm16_16k: bytes) -> str | None: ...


def _configure_transport_logging() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    configure_component_logger(
        logger,
        component="ws_server",
        env_level_key="CHROMA_SERVER_LOG_LEVEL",
        default_level="INFO",
    )
    _LOGGER_CONFIGURED = True


class VoicebotWebSocketServer:
    def __init__(
        self,
        engine: StreamingVoicebotEngine,
        *,
        host: str,
        port: int,
        max_message_size: int = 8 * 1024 * 1024,
        asr_transcriber: AudioTranscriber | None = None,
        server_asr_timeout_sec: float = 1.2,
        session_log_root: str | Path | None = None,
        session_log_enabled: bool = True,
    ) -> None:
        _configure_transport_logging()
        self.engine = engine
        self.host = host
        self.port = port
        self.max_message_size = max_message_size
        self.asr_transcriber = asr_transcriber
        self.server_asr_timeout_sec = max(0.05, float(server_asr_timeout_sec))
        self.session_log_enabled = bool(session_log_enabled)
        repo_root = Path(__file__).resolve().parents[2]
        if session_log_root is None:
            self.session_log_root = repo_root / "logs"
        else:
            self.session_log_root = Path(session_log_root)
        if self.session_log_enabled:
            self.session_log_root.mkdir(parents=True, exist_ok=True)
            logger.info("session logging enabled root=%s", self.session_log_root)
        else:
            logger.info("session logging disabled")

    async def serve_forever(self) -> None:
        logger.info("server start url=ws://%s:%s", self.host, self.port)
        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_size,
        ):
            await asyncio.Future()

    async def _handle_connection(self, websocket) -> None:
        owned_sessions: set[str] = set()
        stream_tasks: dict[str, asyncio.Task] = {}
        auto_commit_tasks: dict[str, asyncio.Task] = {}
        session_logs: dict[str, _SessionLogWriter] = {}
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> bool:
            async with send_lock:
                try:
                    await websocket.send(json.dumps(payload, ensure_ascii=True))
                except ConnectionClosed:
                    return False
            return True

        async def _start_commit(request: dict[str, Any], source: str) -> None:
            session_id = self._require_session_id(request)
            task = stream_tasks.get(session_id)
            if task is not None and not task.done():
                logger.info(
                    "commit skipped session_id=%s source=%s reason=active_turn",
                    session_id,
                    source,
                )
                return
            if not await send_json(
                {"type": "response.stream.opened", "session_id": session_id}
            ):
                return
            logger.info("commit start session_id=%s source=%s", session_id, source)
            stream_task = asyncio.create_task(
                self._handle_audio_commit(
                    request=request,
                    send_json=send_json,
                    source=source,
                    session_logs=session_logs,
                )
            )
            stream_tasks[session_id] = stream_task
            stream_task.add_done_callback(lambda _: stream_tasks.pop(session_id, None))

        async def _auto_commit_loop(session_id: str) -> None:
            try:
                while True:
                    await asyncio.sleep(0.05)
                    if session_id not in owned_sessions:
                        return
                    try:
                        self.engine.get_session_config(session_id)
                    except ValueError:
                        return
                    task = stream_tasks.get(session_id)
                    if task is not None and not task.done():
                        continue
                    ready, reason, trailing_ms, buffered_sec = self.engine.should_auto_commit(
                        session_id
                    )
                    if not ready:
                        continue
                    logger.info(
                        "auto commit trigger session_id=%s reason=%s buffered_sec=%.3f trailing_silence_ms=%.1f",
                        session_id,
                        reason,
                        buffered_sec,
                        trailing_ms,
                    )
                    await _start_commit(
                        {"type": "input.turn.commit", "session_id": session_id},
                        f"auto:{reason}",
                    )
            except asyncio.CancelledError:
                return

        try:
            async for raw_message in websocket:
                request = self._parse_json(raw_message)
                if request is None:
                    await self._send_error(
                        ErrorEvent(
                            session_id=None,
                            code="bad_json",
                            message="Invalid JSON payload",
                        ),
                        send_json=send_json,
                    )
                    continue

                event_type = request.get("type")
                if not isinstance(event_type, str):
                    await self._send_error(
                        ErrorEvent(
                            session_id=None,
                            code="missing_type",
                            message="Field 'type' is required",
                        ),
                        send_json=send_json,
                    )
                    continue

                try:
                    if event_type == "session.open":
                        response = self._handle_session_open(request, owned_sessions)
                        if not await send_json(response):
                            break
                        session_id = response.get("session_id")
                        if (
                            self.session_log_enabled
                            and isinstance(session_id, str)
                            and session_id not in session_logs
                        ):
                            try:
                                session_logs[session_id] = _SessionLogWriter(
                                    self.session_log_root, session_id
                                )
                            except Exception:
                                logger.exception(
                                    "session log init failed session_id=%s", session_id
                                )
                        if isinstance(session_id, str) and session_id not in auto_commit_tasks:
                            auto_commit_tasks[session_id] = asyncio.create_task(
                                _auto_commit_loop(session_id)
                            )
                    elif event_type == "session.update":
                        response = self._handle_session_update(request)
                        if not await send_json(response):
                            break
                    elif event_type == "input.audio.append":
                        response = self._handle_audio_append(request)
                        if not await send_json(response):
                            break
                    elif event_type == "input.turn.commit":
                        await _start_commit(request, "client")
                    elif event_type == "response.cancel":
                        self._handle_response_cancel(request)
                    elif event_type == "session.close":
                        response = self._handle_session_close(request, owned_sessions)
                        if not await send_json(response):
                            break
                        session_id = response.get("session_id")
                        task = auto_commit_tasks.pop(session_id, None)
                        if task is not None:
                            task.cancel()
                        if isinstance(session_id, str):
                            session_logs.pop(session_id, None)
                    else:
                        await self._send_error(
                            ErrorEvent(
                                session_id=request.get("session_id"),
                                code="unknown_type",
                                message=f"Unsupported event type: {event_type}",
                            ),
                            send_json=send_json,
                        )
                except ConnectionClosed:
                    logger.info("client disconnected while handling event=%s", event_type)
                    break
                except RequestError as exc:
                    await self._send_error(
                        ErrorEvent(
                            session_id=request.get("session_id"),
                            code=exc.code,
                            message=exc.message,
                        ),
                        send_json=send_json,
                    )
                except Exception as exc:
                    logger.exception("request failed event=%s", event_type)
                    await self._send_error(
                        ErrorEvent(
                            session_id=request.get("session_id"),
                            code="request_failed",
                            message=str(exc),
                        ),
                        send_json=send_json,
                    )
        except ConnectionClosed:
            logger.info("websocket closed")
        finally:
            for task in stream_tasks.values():
                task.cancel()
            if stream_tasks:
                await asyncio.gather(*stream_tasks.values(), return_exceptions=True)
            for task in auto_commit_tasks.values():
                task.cancel()
            if auto_commit_tasks:
                await asyncio.gather(*auto_commit_tasks.values(), return_exceptions=True)
            for session_id in owned_sessions:
                try:
                    self.engine.close_session(session_id)
                except Exception:
                    logger.exception("session close failed session_id=%s", session_id)

    def _handle_session_open(
        self,
        request: dict[str, Any],
        owned_sessions: set[str],
    ) -> dict[str, Any]:
        session_id = request.get("session_id") or str(uuid.uuid4())
        config = session_config_from_payload(request.get("config"))
        self.engine.create_session(session_id, config)
        owned_sessions.add(session_id)
        normalized = self.engine.get_session_config(session_id)
        return {
            "type": "session.opened",
            "session_id": session_id,
            "config": session_config_to_dict(normalized),
        }

    def _handle_session_update(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        base = self.engine.get_session_config(session_id)
        config = session_config_from_payload(request.get("config"), base=base)
        self.engine.update_session(session_id, config)
        normalized = self.engine.get_session_config(session_id)
        return {
            "type": "session.updated",
            "session_id": session_id,
            "config": session_config_to_dict(normalized),
        }

    def _handle_audio_append(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        audio_b64 = request.get("audio_b64")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise RequestError("invalid_audio_payload", "Field 'audio_b64' is required")
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise RequestError("invalid_base64", "Invalid base64 in field 'audio_b64'") from exc

        self.engine.append_audio(session_id, audio_bytes)
        return {
            "type": "input.audio.accepted",
            "session_id": session_id,
            "num_bytes": len(audio_bytes),
        }

    async def _handle_audio_commit(
        self,
        request: dict[str, Any],
        send_json: Callable[[dict[str, Any]], Awaitable[bool]],
        source: str,
        session_logs: dict[str, _SessionLogWriter],
    ) -> None:
        session_id = request.get("session_id")
        turn_id: str | None = None
        user_audio_bytes = b""
        user_text: str | None = None
        user_text_source = "none"
        assistant_audio_bytes = bytearray()
        assistant_text: str | None = None
        assistant_event = "stream_incomplete"
        metrics: dict[str, float] = {}
        turn_error: dict[str, Any] | None = None
        session_logger: _SessionLogWriter | None = None
        try:
            session_id = self._require_session_id(request)
            if self.session_log_enabled:
                session_logger = session_logs.get(session_id)
                if session_logger is None:
                    try:
                        session_logger = _SessionLogWriter(self.session_log_root, session_id)
                    except Exception:
                        logger.exception("session log init failed session_id=%s", session_id)
                    else:
                        session_logs[session_id] = session_logger

            if session_logger is not None:
                user_audio_bytes = self.engine.get_buffered_audio(session_id)
            transcript = request.get("transcript")
            if transcript is not None and not isinstance(transcript, str):
                raise ValueError("Field 'transcript' must be string")
            server_transcript = await self._transcribe_for_commit(session_id)
            if server_transcript:
                self.engine.set_pending_user_text(session_id, server_transcript)
                user_text = server_transcript
                user_text_source = "server_asr"
                logger.info(
                    "server_asr transcript accepted session_id=%s length=%s",
                    session_id,
                    len(server_transcript),
                )
                logger.info(
                    "user.input session_id=%s source=server_asr text=%s",
                    session_id,
                    _clip_text(server_transcript),
                )
            elif self.asr_transcriber is None and transcript:
                self.engine.set_pending_user_text(session_id, transcript)
                user_text = transcript
                user_text_source = "client_transcript"
                logger.info(
                    "user.input session_id=%s source=client text=%s",
                    session_id,
                    _clip_text(transcript),
                )
            elif self.asr_transcriber is not None and transcript:
                logger.debug(
                    "client transcript ignored because server ASR is enabled session_id=%s",
                    session_id,
                )

            async for event in self.engine.commit_turn(session_id):
                event_type = getattr(event, "type", None)
                event_turn_id = getattr(event, "turn_id", None)
                if isinstance(event_turn_id, str):
                    turn_id = event_turn_id

                if session_logger is None:
                    pass
                elif event_type == "response.audio.delta":
                    audio_b64 = getattr(event, "audio_b64", None)
                    if isinstance(audio_b64, str) and audio_b64:
                        try:
                            chunk = base64.b64decode(audio_b64, validate=True)
                        except Exception:
                            logger.warning(
                                "session log skipped invalid response audio chunk session_id=%s turn_id=%s",
                                session_id,
                                turn_id,
                            )
                        else:
                            assistant_audio_bytes.extend(chunk)
                elif event_type == "response.text.delta":
                    text_delta = getattr(event, "text", None)
                    if isinstance(text_delta, str) and text_delta:
                        assistant_text = text_delta
                elif event_type == "response.done":
                    assistant_event = "response.done"
                    final_text = getattr(event, "text", None)
                    if isinstance(final_text, str) and final_text:
                        assistant_text = final_text
                    raw_metrics = getattr(event, "metrics", None)
                    if isinstance(raw_metrics, dict):
                        normalized_metrics: dict[str, float] = {}
                        for key, value in raw_metrics.items():
                            if isinstance(value, bool):
                                continue
                            if isinstance(value, (int, float)):
                                normalized_metrics[str(key)] = float(value)
                        metrics = normalized_metrics
                if event_type == "response.cancelled":
                    assistant_event = "response.cancelled"
                elif event_type == "error":
                    assistant_event = "error"
                    turn_error = {
                        "code": getattr(event, "code", "engine_error"),
                        "message": getattr(event, "message", "Engine returned error"),
                        "details": getattr(event, "details", None),
                    }

                if event_type in {"response.done", "response.cancelled"}:
                    logger.info(
                        "turn completed session_id=%s turn_id=%s event_type=%s",
                        session_id,
                        getattr(event, "turn_id", None),
                        event_type,
                    )
                if not await send_json(event_to_dict(event)):
                    return
        except asyncio.CancelledError:
            assistant_event = "response.cancelled"
            if isinstance(session_id, str):
                self.engine.cancel_response(session_id)
            raise
        except ConnectionClosed:
            return
        except Exception as exc:
            assistant_event = "error"
            turn_error = {"code": "audio_commit_failed", "message": str(exc)}
            logger.exception("commit failed session_id=%s", request.get("session_id"))
            await send_json(
                event_to_dict(
                    ErrorEvent(
                        session_id=request.get("session_id"),
                        code="audio_commit_failed",
                        message=str(exc),
                    )
                )
            )
        finally:
            if session_logger is not None:
                try:
                    session_logger.record_turn(
                        source=source,
                        turn_id=turn_id,
                        user_audio_bytes=user_audio_bytes,
                        user_text=user_text,
                        user_text_source=user_text_source,
                        assistant_audio_bytes=bytes(assistant_audio_bytes),
                        assistant_text=assistant_text,
                        assistant_event=assistant_event,
                        metrics=metrics,
                        error=turn_error,
                    )
                except Exception:
                    logger.exception(
                        "session log write failed session_id=%s turn_id=%s",
                        session_id,
                        turn_id,
                    )

    async def _transcribe_for_commit(self, session_id: str) -> str | None:
        if self.asr_transcriber is None:
            return None
        audio_bytes = self.engine.get_buffered_audio(session_id)
        if not audio_bytes:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.asr_transcriber.transcribe_pcm16_16k, audio_bytes),
                timeout=self.server_asr_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "server_asr timeout session_id=%s timeout_sec=%.2f",
                session_id,
                self.server_asr_timeout_sec,
            )
            return None
        except Exception:
            logger.exception("server_asr failure session_id=%s", session_id)
            return None

    def _handle_response_cancel(self, request: dict[str, Any]) -> None:
        session_id = self._require_session_id(request)
        self.engine.cancel_response(session_id)

    def _handle_session_close(
        self,
        request: dict[str, Any],
        owned_sessions: set[str],
    ) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        self.engine.close_session(session_id)
        owned_sessions.discard(session_id)
        return {"type": "session.closed", "session_id": session_id}

    @staticmethod
    def _require_session_id(request: dict[str, Any]) -> str:
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Field 'session_id' is required")
        return session_id

    @staticmethod
    def _parse_json(raw_message: Any) -> dict[str, Any] | None:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8", errors="replace")
        if not isinstance(raw_message, str):
            return None
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    async def _send_error(
        self,
        error_event: ErrorEvent,
        send_json: Callable[[dict[str, Any]], Awaitable[bool]],
    ) -> None:
        await send_json(event_to_dict(error_event))


__all__ = ["VoicebotWebSocketServer"]
