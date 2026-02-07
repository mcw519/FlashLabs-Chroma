from __future__ import annotations

import json

import pytest

from chroma.engine.bot_config import load_bot_config


def test_load_bot_config_toml_top_level(tmp_path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text('system_prompt = "You are a concise travel assistant."', encoding="utf-8")

    config = load_bot_config(path)

    assert config.system_prompt == "You are a concise travel assistant."


def test_load_bot_config_json_nested_bot_section(tmp_path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps({"bot": {"system_prompt": "You are a patient coding mentor."}}),
        encoding="utf-8",
    )

    config = load_bot_config(path)

    assert config.system_prompt == "You are a patient coding mentor."


def test_load_bot_config_requires_system_prompt(tmp_path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text('name = "demo-agent"', encoding="utf-8")

    with pytest.raises(ValueError, match="system_prompt"):
        load_bot_config(path)
