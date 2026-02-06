from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

import websockets

from chroma.engine.streaming_engine import StreamingVoicebotEngine
from chroma.engine.types import ErrorEvent, SessionConfig, event_to_dict

logger = logging.getLogger(__name__)


class VoicebotWebSocketServer:
    def __init__(
        self,
        engine: StreamingVoicebotEngine,
        *,
        host: str,
        port: int,
        max_message_size: int = 8 * 1024 * 1024,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.max_message_size = max_message_size

    async def serve_forever(self) -> None:
        logger.info("Starting WebSocket server on ws://%s:%s", self.host, self.port)
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
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send(json.dumps(payload, ensure_ascii=True))

        try:
            async for raw_message in websocket:
                request = self._parse_json(raw_message)
                if request is None:
                    await self._send_error(
                        websocket,
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
                        websocket,
                        ErrorEvent(
                            session_id=None,
                            code="missing_type",
                            message="Field 'type' is required",
                        ),
                        send_json=send_json,
                    )
                    continue

                try:
                    if event_type == "session.start":
                        response = self._handle_session_start(request, owned_sessions)
                        await send_json(response)
                    elif event_type == "session.update":
                        response = self._handle_session_update(request)
                        await send_json(response)
                    elif event_type == "audio.append":
                        response = self._handle_audio_append(request)
                        await send_json(response)
                    elif event_type == "audio.commit":
                        session_id = self._require_session_id(request)
                        task = stream_tasks.get(session_id)
                        if task is not None and not task.done():
                            task.cancel()
                        stream_tasks[session_id] = asyncio.create_task(
                            self._handle_audio_commit(
                                websocket=websocket,
                                request=request,
                                send_json=send_json,
                            )
                        )
                        await send_json(
                            {
                                "type": "response.stream.started",
                                "session_id": session_id,
                            }
                        )
                    elif event_type == "response.cancel":
                        response = self._handle_response_cancel(request)
                        await send_json(response)
                    elif event_type == "session.end":
                        response = self._handle_session_end(request, owned_sessions)
                        await send_json(response)
                    else:
                        await self._send_error(
                            websocket,
                            ErrorEvent(
                                session_id=request.get("session_id"),
                                code="unknown_type",
                                message=f"Unsupported event type: {event_type}",
                            ),
                            send_json=send_json,
                        )
                except Exception as exc:
                    logger.exception("Failed to handle event: %s", event_type)
                    await self._send_error(
                        websocket,
                        ErrorEvent(
                            session_id=request.get("session_id"),
                            code="request_failed",
                            message=str(exc),
                        ),
                        send_json=send_json,
                    )
        finally:
            for task in stream_tasks.values():
                task.cancel()
            for session_id in owned_sessions:
                try:
                    self.engine.close_session(session_id)
                except Exception:
                    logger.exception("Failed to close session: %s", session_id)

    def _handle_session_start(self, request: dict[str, Any], owned_sessions: set[str]) -> dict[str, Any]:
        session_id = request.get("session_id") or str(uuid.uuid4())
        config_payload = request.get("config") or {}
        config = SessionConfig(
            speaker=config_payload.get("speaker", "scarlett_johansson"),
            memory_turns=config_payload.get("memory_turns", 6),
            output_chunk_sec=config_payload.get("output_chunk_sec", 0.24),
            text_mode=config_payload.get("text_mode", "sentence"),
        )
        self.engine.create_session(session_id, config)
        owned_sessions.add(session_id)
        return {
            "type": "session.started",
            "session_id": session_id,
            "config": {
                "speaker": config.speaker,
                "memory_turns": config.memory_turns,
                "output_chunk_sec": config.output_chunk_sec,
                "text_mode": config.text_mode,
            },
        }

    def _handle_session_update(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        updates = request.get("config") or {}
        self.engine.update_session(
            session_id,
            speaker=updates.get("speaker"),
            memory_turns=updates.get("memory_turns"),
            output_chunk_sec=updates.get("output_chunk_sec"),
            text_mode=updates.get("text_mode"),
        )
        return {
            "type": "session.updated",
            "session_id": session_id,
            "config": updates,
        }

    def _handle_audio_append(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        audio_b64 = request.get("audio_b64")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ValueError("Field 'audio_b64' is required")
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as exc:
            raise ValueError("Invalid base64 in field 'audio_b64'") from exc

        self.engine.append_audio(session_id, audio_bytes)
        return {
            "type": "audio.appended",
            "session_id": session_id,
            "num_bytes": len(audio_bytes),
        }

    async def _handle_audio_commit(self, websocket, request: dict[str, Any], send_json) -> None:
        session_id = request.get("session_id")
        try:
            session_id = self._require_session_id(request)
            transcript = request.get("transcript")
            if transcript is not None and not isinstance(transcript, str):
                raise ValueError("Field 'transcript' must be string")
            if transcript:
                self.engine.set_pending_user_text(session_id, transcript)

            async for event in self.engine.commit_turn(session_id):
                await send_json(event_to_dict(event))
        except asyncio.CancelledError:
            if isinstance(session_id, str):
                self.engine.cancel_response(session_id)
            raise
        except Exception as exc:
            logger.exception("audio.commit failed for session %s", request.get("session_id"))
            await send_json(
                event_to_dict(
                    ErrorEvent(
                        session_id=request.get("session_id"),
                        code="audio_commit_failed",
                        message=str(exc),
                    )
                )
            )

    def _handle_response_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        self.engine.cancel_response(session_id)
        return {
            "type": "response.cancelled.requested",
            "session_id": session_id,
        }

    def _handle_session_end(self, request: dict[str, Any], owned_sessions: set[str]) -> dict[str, Any]:
        session_id = self._require_session_id(request)
        self.engine.close_session(session_id)
        owned_sessions.discard(session_id)
        return {"type": "session.ended", "session_id": session_id}

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

    async def _send_error(self, websocket, error_event: ErrorEvent, send_json=None) -> None:
        payload = event_to_dict(error_event)
        if send_json is not None:
            await send_json(payload)
            return
        await websocket.send(json.dumps(payload, ensure_ascii=True))


__all__ = ["VoicebotWebSocketServer"]
