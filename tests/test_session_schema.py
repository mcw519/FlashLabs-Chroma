from __future__ import annotations

import pytest

from chroma.session_schema import SessionConfigV2, session_config_from_payload


def test_session_config_defaults() -> None:
    config = session_config_from_payload(None)
    assert config.speaker == "scarlett_johansson"
    assert config.turn_detection.mode == "server_vad"
    assert config.turn_detection.threshold == 0.5


def test_session_config_update_preserves_missing_fields() -> None:
    base = SessionConfigV2(system_prompt="Persona A")
    config = session_config_from_payload({"speaker": "ariana_grande"}, base=base)
    assert config.speaker == "ariana_grande"
    assert config.system_prompt == "Persona A"


def test_session_config_update_can_clear_system_prompt() -> None:
    base = SessionConfigV2(system_prompt="Persona A")
    config = session_config_from_payload({"system_prompt": None}, base=base)
    assert config.system_prompt is None


def test_session_config_invalid_turn_detection_payload() -> None:
    with pytest.raises(ValueError, match="turn_detection"):
        session_config_from_payload({"turn_detection": []})
