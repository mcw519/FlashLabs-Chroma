from __future__ import annotations

import asyncio
import base64
import json
import socket
import tempfile
import wave
from pathlib import Path

import websockets

from chroma.engine.types import (
    ResponseAudioDeltaEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
    ResponseTextDeltaEvent,
    SessionConfig,
    TurnDetectionConfig,
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
            trim_with_vad=config.trim_with_vad,
            turn_detection=TurnDetectionConfig(
                mode=config.turn_detection.mode,
                threshold=config.turn_detection.threshold,
                min_speech_ms=config.turn_detection.min_speech_ms,
                min_silence_ms=config.turn_detection.min_silence_ms,
                speech_pad_ms=config.turn_detection.speech_pad_ms,
            ),
        )
        self.sessions[session_id] = {"config": normalized, "audio": bytearray()}

    def update_session(self, session_id, config) -> None:
        if session_id not in self.sessions:
            raise ValueError("missing session")
        self.create_session(session_id, config)

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
        config = self.sessions[session_id]["config"]
        if config.turn_detection.mode == "server_vad" and self.sessions[session_id]["audio"]:
            return True, "silence", 600.0, 0.6
        return False, "mode_client_commit", 0.0, 0.0

    def get_buffered_audio(self, session_id) -> bytes:
        return bytes(self.sessions[session_id]["audio"])

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


async def _start_server(
    log_root: Path | None = None,
    disable_session_log: bool = False,
):
    port = _free_port()
    engine = FakeEngine()
    server = VoicebotWebSocketServer(
        engine=engine,
        host="127.0.0.1",
        port=port,
        session_log_root=log_root,
        session_log_enabled=not disable_session_log,
    )
    task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)
    return engine, task, f"ws://127.0.0.1:{port}"


async def _stop_server(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_ws_input_turn_commit_flow() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"session.open","session_id":"s1"}')
                opened = await ws.recv()
                assert "session.opened" in opened

                audio_b64 = base64.b64encode(b"\x00\x00\x01\x00").decode("ascii")
                await ws.send(
                    '{"type":"input.audio.append","session_id":"s1","audio_b64":"'
                    + audio_b64
                    + '"}'
                )
                accepted = await ws.recv()
                assert "input.audio.accepted" in accepted

                await ws.send(
                    '{"type":"input.turn.commit","session_id":"s1","transcript":"hello"}'
                )
                stream_opened = await ws.recv()
                event1 = await ws.recv()
                event2 = await ws.recv()
                event3 = await ws.recv()
                event4 = await ws.recv()

                assert "response.stream.opened" in stream_opened
                assert "response.started" in event1
                assert "response.audio.delta" in event2
                assert "response.text.delta" in event3
                assert "response.done" in event4
                assert engine.pending_text["s1"] == "hello"
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_commit_persists_session_artifacts() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir) / "logs"
            engine, task, url = await _start_server(log_root=log_root)
            try:
                async with websockets.connect(url) as ws:
                    session_id = "persist_s1"
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.open",
                                "session_id": session_id,
                            }
                        )
                    )
                    await ws.recv()

                    audio_payload = b"\x00\x00\x01\x00"
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.audio.append",
                                "session_id": session_id,
                                "audio_b64": base64.b64encode(audio_payload).decode(
                                    "ascii"
                                ),
                            }
                        )
                    )
                    await ws.recv()

                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.turn.commit",
                                "session_id": session_id,
                                "transcript": "hello",
                            }
                        )
                    )

                    stream_opened = await ws.recv()
                    event1 = await ws.recv()
                    event2 = await ws.recv()
                    event3 = await ws.recv()
                    event4 = await ws.recv()
                    assert "response.stream.opened" in stream_opened
                    assert "response.started" in event1
                    assert "response.audio.delta" in event2
                    assert "response.text.delta" in event3
                    assert "response.done" in event4
                    assert engine.pending_text[session_id] == "hello"

                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.close",
                                "session_id": session_id,
                            }
                        )
                    )
                    await ws.recv()
            finally:
                await _stop_server(task)

            date_dirs = [p for p in log_root.iterdir() if p.is_dir()]
            assert len(date_dirs) == 1
            session_dir = date_dirs[0] / "persist_s1"
            assert session_dir.exists()

            user_audio_path = session_dir / "user_0001.wav"
            bot_audio_path = session_dir / "bot_0001.wav"
            conversation_path = session_dir / "conversation_log.json"
            assert user_audio_path.exists()
            assert bot_audio_path.exists()
            assert conversation_path.exists()

            with wave.open(str(user_audio_path), "rb") as user_wav:
                assert user_wav.getframerate() == 16000
                assert user_wav.getnchannels() == 1
                assert user_wav.getsampwidth() == 2
                assert user_wav.getnframes() > 0

            with wave.open(str(bot_audio_path), "rb") as bot_wav:
                assert bot_wav.getframerate() == 24000
                assert bot_wav.getnchannels() == 1
                assert bot_wav.getsampwidth() == 2
                assert bot_wav.getnframes() > 0

            payload = json.loads(conversation_path.read_text(encoding="utf-8"))
            assert payload["session_id"] == "persist_s1"
            assert len(payload["turns"]) == 1
            turn = payload["turns"][0]
            assert turn["user"]["text"] == "hello"
            assert turn["user"]["text_source"] == "client_transcript"
            assert turn["user"]["audio_file"] == "user_0001.wav"
            assert turn["assistant"]["text"] == "hello"
            assert turn["assistant"]["event"] == "response.done"
            assert turn["assistant"]["audio_file"] == "bot_0001.wav"
            assert turn["metrics"]["ttfs_ms"] == 100.0

    asyncio.run(_run())


def test_ws_commit_disable_session_log() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir) / "logs_disabled"
            engine, task, url = await _start_server(
                log_root=log_root,
                disable_session_log=True,
            )
            try:
                async with websockets.connect(url) as ws:
                    session_id = "no_log_s1"
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.open",
                                "session_id": session_id,
                            }
                        )
                    )
                    await ws.recv()

                    audio_payload = b"\x00\x00\x01\x00"
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.audio.append",
                                "session_id": session_id,
                                "audio_b64": base64.b64encode(audio_payload).decode(
                                    "ascii"
                                ),
                            }
                        )
                    )
                    await ws.recv()

                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.turn.commit",
                                "session_id": session_id,
                                "transcript": "hello",
                            }
                        )
                    )
                    await ws.recv()
                    await ws.recv()
                    await ws.recv()
                    await ws.recv()
                    await ws.recv()
                    assert engine.pending_text[session_id] == "hello"
            finally:
                await _stop_server(task)

            assert not log_root.exists()

    asyncio.run(_run())


def test_ws_cancel_has_no_ack() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"session.open","session_id":"s2"}')
                await ws.recv()

                await ws.send('{"type":"response.cancel","session_id":"s2"}')
                await asyncio.sleep(0.05)
                assert engine.cancel_count["s2"] == 1

                try:
                    payload = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    payload = ""
                assert payload == ""
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


def test_ws_session_open_with_system_prompt() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.open",
                            "session_id": "s3",
                            "config": {"system_prompt": "  You are a tutor.  "},
                        }
                    )
                )
                opened = json.loads(await ws.recv())
                assert opened["type"] == "session.opened"
                assert opened["config"]["system_prompt"] == "You are a tutor."
                assert engine.sessions["s3"]["config"].system_prompt == "You are a tutor."
        finally:
            await _stop_server(task)

    asyncio.run(_run())


def test_ws_session_update_set_and_clear_system_prompt() -> None:
    async def _run() -> None:
        engine, task, url = await _start_server()
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "session.open", "session_id": "s4"}))
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
                            "type": "session.open",
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
                            "type": "session.open",
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
                await ws.send(json.dumps({"type": "session.open", "session_id": "s7"}))
                await ws.recv()
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.audio.append",
                            "session_id": "s7",
                            "audio_b64": "###",
                        }
                    )
                )
                error = json.loads(await ws.recv())
                assert error["type"] == "error"
                assert error["code"] == "invalid_base64"
        finally:
            await _stop_server(task)

    asyncio.run(_run())
