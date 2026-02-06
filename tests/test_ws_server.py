from __future__ import annotations

import asyncio
import base64
import socket

import websockets

from chroma.engine.types import (
    ResponseAudioDeltaEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
    ResponseTextDeltaEvent,
)
from chroma.transport.ws_server import VoicebotWebSocketServer


class FakeEngine:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.cancel_count: dict[str, int] = {}
        self.pending_text: dict[str, str | None] = {}

    def create_session(self, session_id, config) -> None:
        self.sessions[session_id] = {"config": config, "audio": bytearray()}

    def update_session(self, session_id, **kwargs) -> None:
        if session_id not in self.sessions:
            raise ValueError("missing session")
        self.sessions[session_id]["updates"] = kwargs

    def set_pending_user_text(self, session_id, text) -> None:
        self.pending_text[session_id] = text

    def append_audio(self, session_id, pcm16_16k: bytes) -> None:
        self.sessions[session_id]["audio"].extend(pcm16_16k)

    def cancel_response(self, session_id) -> None:
        self.cancel_count[session_id] = self.cancel_count.get(session_id, 0) + 1

    def close_session(self, session_id) -> None:
        self.sessions.pop(session_id, None)

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
