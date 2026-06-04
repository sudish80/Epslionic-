"""Tests for core modules: config, domain, memory, gateway, heartbeat, brain."""

import os
import json
import pytest


class TestConfig:
    """AgentConfig loading and environment resolution."""

    def test_default_config(self):
        from epsionic.config import AgentConfig
        cfg = AgentConfig()
        assert cfg.workspace_root is not None
        assert cfg.openai_api_key is None
        assert cfg.huggingface_token is None
        assert cfg.llm_model == "gpt-4o-mini"
        assert cfg.heartbeat_interval_seconds == 60

    def test_from_json_file(self, tmp_workspace):
        from epsionic.config import AgentConfig
        path = tmp_workspace / "config.json"
        path.write_text(json.dumps({"llm_model": "gpt-4", "heartbeat_interval_seconds": 120}))
        cfg = AgentConfig.from_file(str(path))
        assert cfg.llm_model == "gpt-4"
        assert cfg.heartbeat_interval_seconds == 120

    def test_resolve_env(self, monkeypatch):
        from epsionic.config import AgentConfig
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")
        cfg = AgentConfig()
        cfg.resolve_env()
        assert cfg.openai_api_key == "sk-test-key"
        assert cfg.huggingface_token == "hf-test-token"

    def test_default_training_config(self):
        from epsionic.config import AgentConfig
        cfg = AgentConfig()
        tc = cfg.default_training_config
        assert tc["model_name"] == "unsloth/mistral-7b-bnb-4bit"
        assert tc["batch_size"] == 2
        assert tc["learning_rate"] == 2e-4


class TestDomain:
    """Domain selection and knowledge base."""

    def test_domains_have_required_keys(self):
        from epsionic.core.domain import DOMAINS
        assert len(DOMAINS) >= 9
        for key, d in DOMAINS.items():
            assert "name" in d, f"{key} missing name"
            assert "emoji" in d, f"{key} missing emoji"
            assert "datasets" in d, f"{key} missing datasets"
            assert "models" in d, f"{key} missing models"
            assert "training_hints" in d or "hints" in d, f"{key} missing hints"

    def test_math_domain(self):
        from epsionic.core.domain import DOMAINS
        math = DOMAINS["math"]
        assert math["name"] == "Mathematical Reasoning"
        assert "gsm8k" in math["datasets"]

    def test_code_domain(self):
        from epsionic.core.domain import DOMAINS
        code = DOMAINS["code"]
        assert code["name"] == "Code Generation"
        assert "code_alpaca" in code["datasets"]

    def test_domain_selector_fuzzy_match(self):
        from epsionic.core.domain import DomainSelector
        sel = DomainSelector()
        assert sel._fuzzy_match("math") == "math"
        assert sel._fuzzy_match("code") == "code"
        assert sel._fuzzy_match("medical") == "medical"

    def test_auto_select_returns_valid_domain(self):
        from epsionic.core.domain import DomainSelector
        sel = DomainSelector()
        env = {"gpu": False, "vram_gb": 0, "internet": True, "api_key": False, "hf_token": False}
        key = sel._auto_pick(env)
        assert key in ["general", "math", "chat"]

    def test_build_objective(self):
        from epsionic.core.domain import DomainSelector, DOMAINS
        sel = DomainSelector()
        obj = sel.build_objective("code", DOMAINS["code"])
        assert "Code Generation" in obj
        assert "code_alpaca" in obj


class TestExceptions:
    """Exception hierarchy."""

    def test_error_codes(self):
        from epsionic.exceptions import (
            EpslionicError, ConfigurationError, DomainError, GPUError,
            TrainingError, AuthenticationError, error_code,
        )
        assert error_code(ConfigurationError("bad config")) == "CONFIG_ERROR"
        assert error_code(DomainError("bad domain")) == "DOMAIN_ERROR"
        assert error_code(GPUError("OOM")) == "GPU_ERROR"
        assert error_code(TrainingError("fail")) == "TRAINING_ERROR"
        assert error_code(AuthenticationError("no key")) == "AUTH_ERROR"
        assert error_code(ValueError("other")) == "UNKNOWN_ERROR"
        assert issubclass(GPUError, EpslionicError)
