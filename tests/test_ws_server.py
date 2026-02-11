from __future__ import annotations

import asyncio
import base64
import json
import socket

import websockets

from chroma.engine.types import (
    ResponseAudioDeltaEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
    ResponseTextDeltaEvent,
    SessionConfig,
)
from chroma.transport.ws_server import VoicebotWebSocketServer


class FakeEngine:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.cancel_count: dict[str, int] = {}
        self.pending_text: dict[str, str | None] = {}

    @staticmethod
    def _normalize_system_prompt(prompt):
        if prompt is None:
            return None
        if not isinstance(prompt, str):
            raise ValueError("system_prompt must be string or null")
        normalized = prompt.strip()
        if not normalized:
            raise ValueError("system_prompt must not be empty")
        if len(normalized) > 4000:
            raise ValueError("system_prompt exceeds 4000 characters")
        return normalized

    def create_session(self, session_id, config) -> None:
        normalized_prompt = self._normalize_system_prompt(config.system_prompt)
        normalized = SessionConfig(
            speaker=config.speaker,
            memory_turns=config.memory_turns,
            output_chunk_sec=config.output_chunk_sec,
            text_mode=config.text_mode,
            include_transcript_in_query=config.include_transcript_in_query,
            system_prompt=normalized_prompt,
            auto_commit=config.auto_commit,
            vad_threshold=config.vad_threshold,
            vad_min_speech_ms=config.vad_min_speech_ms,
            vad_min_silence_ms=config.vad_min_silence_ms,
            vad_speech_pad_ms=config.vad_speech_pad_ms,
            trim_with_vad=config.trim_with_vad,
        )
        self.sessions[session_id] = {"config": normalized, "audio": bytearray()}

    def update_session(self, session_id, **kwargs) -> None:
        if session_id not in self.sessions:
            raise ValueError("missing session")
        config = self.sessions[session_id]["config"]
        if "system_prompt" in kwargs:
            system_prompt = self._normalize_system_prompt(kwargs.get("system_prompt"))
        else:
            system_prompt = config.system_prompt
        self.sessions[session_id]["config"] = SessionConfig(
            speaker=config.speaker if kwargs.get("speaker") is None else kwargs.get("speaker"),
            memory_turns=(
                config.memory_turns
                if kwargs.get("memory_turns") is None
                else kwargs.get("memory_turns")
            ),
            output_chunk_sec=(
                config.output_chunk_sec
                if kwargs.get("output_chunk_sec") is None
                else kwargs.get("output_chunk_sec")
            ),
            text_mode=config.text_mode if kwargs.get("text_mode") is None else kwargs.get("text_mode"),
            include_transcript_in_query=(
                config.include_transcript_in_query
                if kwargs.get("include_transcript_in_query") is None
                else kwargs.get("include_transcript_in_query")
            ),
            system_prompt=system_prompt,
            auto_commit=(
                config.auto_commit if kwargs.get("auto_commit") is None else kwargs.get("auto_commit")
            ),
            vad_threshold=(
                config.vad_threshold
                if kwargs.get("vad_threshold") is None
                else kwargs.get("vad_threshold")
            ),
            vad_min_speech_ms=(
                config.vad_min_speech_ms
                if kwargs.get("vad_min_speech_ms") is None
                else kwargs.get("vad_min_speech_ms")
            ),
            vad_min_silence_ms=(
                config.vad_min_silence_ms
                if kwargs.get("vad_min_silence_ms") is None
                else kwargs.get("vad_min_silence_ms")
            ),
            vad_speech_pad_ms=(
                config.vad_speech_pad_ms
                if kwargs.get("vad_speech_pad_ms") is None
                else kwargs.get("vad_speech_pad_ms")
            ),
            trim_with_vad=(
                config.trim_with_vad
                if kwargs.get("trim_with_vad") is None
                else kwargs.get("trim_with_vad")
            ),
        )
        self.sessions[session_id]["updates"] = kwargs

    def set_pending_user_text(self, session_id, text) -> None:
        self.pending_text[session_id] = text

    def append_audio(self, session_id, pcm16_16k: bytes) -> None:
        self.sessions[session_id]["audio"].extend(pcm16_16k)

    def cancel_response(self, session_id) -> None:
        self.cancel_count[session_id] = self.cancel_count.get(session_id, 0) + 1

    def close_session(self, session_id) -> None:
        self.sessions.pop(session_id, None)

    def get_session_config(self, session_id):
        return self.sessions[session_id]["config"]

    def should_auto_commit(self, session_id):
        return False, "disabled", 0.0, 0.0

    async def commit_turn(self, session_id):
        turn_id = "1"
        yield ResponseStartedEvent(session_id=session_id, turn_id=turn_id)
        yield ResponseAudioDeltaEvent(
            session_id=session_id,
            turn_id=turn_id,
            audio_b64=base64.b64encode(b"\x01\x02").decode("ascii"),
            sample_rate=24000,
        )
        pending = self.pending_text.get(session_id)
        if pending:
            yield ResponseTextDeltaEvent(
                session_id=session_id,
                turn_id=turn_id,
                text=pending,
            )
        yield ResponseDoneEvent(
            session_id=session_id,
            turn_id=turn_id,
            text=pending,
            metrics={"ttfs_ms": 100.0},
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_server():
    port = _free_port()
    engine = FakeEngine()
    server = VoicebotWebSocketServer(engine=engine, host="127.0.0.1", port=port)
    task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)
    return engine, task, f"ws://127.0.0.1:{port}"


async def _stop_server(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_ws_audio_commit_flow() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"session.start","session_id":"s1"}')
                started = await ws.recv()
                assert "session.started" in started

                audio_b64 = base64.b64encode(b"\x00\x00\x01\x00").decode("ascii")
                await ws.send(
                    '{"type":"audio.append","session_id":"s1","audio_b64":"'
                    + audio_b64
                    + '"}'
                )
                appended = await ws.recv()
                assert "audio.appended" in appended

                await ws.send(
                    '{"type":"audio.commit","session_id":"s1","transcript":"hello"}'
                )
                stream_started = await ws.recv()
                event1 = await ws.recv()
                event2 = await ws.recv()
                event3 = await ws.recv()
                event4 = await ws.recv()

                assert "response.stream.started" in stream_started
                assert "response.started" in event1
                assert "response.audio.delta" in event2
                assert "response.text.delta" in event3
                assert "response.done" in event4
                assert engine.pending_text["s1"] == "hello"
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_cancel_idempotent() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"session.start","session_id":"s2"}')
                await ws.recv()

                await ws.send('{"type":"response.cancel","session_id":"s2"}')
                resp1 = await ws.recv()
                await ws.send('{"type":"response.cancel","session_id":"s2"}')
                resp2 = await ws.recv()

                assert "response.cancelled.requested" in resp1
                assert "response.cancelled.requested" in resp2
                assert engine.cancel_count["s2"] == 2
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_invalid_payload_returns_error() -> None:
    async def _run() -> None:
        _, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send("{")
                error = await ws.recv()
                assert '"type": "error"' in error
                assert "bad_json" in error
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_session_start_with_system_prompt() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "session_id": "s3",
                            "config": {"system_prompt": "  You are a tutor.  "},
                        }
                    )
                )
                started = json.loads(await ws.recv())
                assert started["type"] == "session.started"
                assert started["config"]["system_prompt"] == "You are a tutor."
                assert engine.sessions["s3"]["config"].system_prompt == "You are a tutor."
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_session_update_set_and_clear_system_prompt() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "session.start", "session_id": "s4"}))
                await ws.recv()

                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session_id": "s4",
                            "config": {"system_prompt": "Persona A"},
                        }
                    )
                )
                updated = json.loads(await ws.recv())
                assert updated["type"] == "session.updated"
                assert updated["config"]["system_prompt"] == "Persona A"
                assert engine.sessions["s4"]["config"].system_prompt == "Persona A"

                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session_id": "s4",
                            "config": {"speaker": "ariana_grande"},
                        }
                    )
                )
                retained = json.loads(await ws.recv())
                assert retained["type"] == "session.updated"
                assert retained["config"]["system_prompt"] == "Persona A"
                assert engine.sessions["s4"]["config"].system_prompt == "Persona A"

                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session_id": "s4",
                            "config": {"system_prompt": None},
                        }
                    )
                )
                cleared = json.loads(await ws.recv())
                assert cleared["type"] == "session.updated"
                assert cleared["config"]["system_prompt"] is None
                assert engine.sessions["s4"]["config"].system_prompt is None
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_invalid_system_prompt_returns_request_failed() -> None:
    async def _run() -> None:
        _, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "session_id": "s5",
                            "config": {"system_prompt": "   "},
                        }
                    )
                )
                error = json.loads(await ws.recv())
                assert error["type"] == "error"
                assert error["code"] == "request_failed"
                assert "system_prompt" in error["message"]
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_system_prompt_too_long_returns_request_failed() -> None:
    async def _run() -> None:
        _, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "session_id": "s6",
                            "config": {"system_prompt": "x" * 4001},
                        }
                    )
                )
                error = json.loads(await ws.recv())
                assert error["type"] == "error"
                assert error["code"] == "request_failed"
                assert "4000" in error["message"]
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_audio_append_invalid_base64_returns_invalid_base64_code() -> None:
    async def _run() -> None:
        _, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "session.start", "session_id": "s7"}))
                await ws.recv()
                await ws.send(
                    json.dumps(
                        {"type": "audio.append", "session_id": "s7", "audio_b64": "###"}
                    )
                )
                error = json.loads(await ws.recv())
                assert error["type"] == "error"
                assert error["code"] == "invalid_base64"
        finally:
            await _stop_server(task)

    asyncio.run(_run())
