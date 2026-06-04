"""Tests for AgentConfig enhancements."""

import os
import json
import pytest
from pathlib import Path


class TestConfigEnhancements:
    """New config features: from_env, to_dict, validation."""

    def test_from_env_prefixed(self, monkeypatch):
        from openclaw_colab_agent.config import AgentConfig
        monkeypatch.setenv("OPENCLAW_LLM_MODEL", "gpt-4")
        monkeypatch.setenv("OPENCLAW_HEARTBEAT_INTERVAL_SECONDS", "30")
        cfg = AgentConfig.from_env()
        assert cfg.llm_model == "gpt-4"
        assert cfg.heartbeat_interval_seconds == 30

    def test_to_dict_converts_paths(self):
        from openclaw_colab_agent.config import AgentConfig
        cfg = AgentConfig()
        d = cfg.to_dict()
        assert isinstance(d["workspace_root"], str)

    def test_validation_catches_bad_temperature(self):
        from openclaw_colab_agent.config import AgentConfig
        cfg = AgentConfig()
        cfg.llm_temperature = 5.0
        errors = cfg.validate()
        assert any("temperature" in e for e in errors)

    def test_validation_catches_low_heartbeat(self):
        from openclaw_colab_agent.config import AgentConfig
        cfg = AgentConfig()
        cfg.heartbeat_interval_seconds = 1
        errors = cfg.validate()
        assert any("heartbeat" in e for e in errors)

    def test_resolve_env_fills_keys(self, monkeypatch):
        from openclaw_colab_agent.config import AgentConfig
        monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
        monkeypatch.setenv("HF_TOKEN", "hf-token")
        cfg = AgentConfig()
        cfg.resolve_env()
        assert cfg.openai_api_key == "sk-key"
        assert cfg.huggingface_token == "hf-token"
