"""Tests for Gateway orchestrator and domain pipeline."""

import os
import pytest


class TestGateway:
    """Gateway construction and state management."""

    def test_gateway_initialization(self, gateway):
        assert gateway.config is not None
        assert gateway.memory is not None
        assert gateway.session_manager is not None
        assert gateway.brain is not None
        assert gateway.heartbeat is not None
        assert gateway.state.status == "idle"

    def test_register_tool(self, gateway):
        gateway.register_tool("test_tool", "A test tool", lambda **kw: "ok")
        assert len(gateway._tool_registry) >= 1

    def test_has_tools_after_setup(self, gateway_with_tools):
        assert len(gateway_with_tools._tool_registry) >= 4

    def test_status_md(self, gateway):
        md = gateway.get_status_markdown()
        assert len(md) > 0
        assert "Gateway" in md or "Status" in md or "status" in md.lower()

    def test_run_domain_pipeline(self, gateway_with_tools):
        from openclaw_colab_agent.core.domain import DOMAINS
        result = gateway_with_tools.run_domain("math", DOMAINS["math"])
        assert "objective" in result
        assert "domain" in result
        assert "experiment_id" in result
        assert "steps" in result
        assert len(result["steps"]) >= 4

    def test_run_domain_creates_experiment(self, gateway_with_tools):
        from openclaw_colab_agent.core.domain import DOMAINS
        before = len(gateway_with_tools.memory.list_experiments())
        gateway_with_tools.run_domain("math", DOMAINS["math"])
        after = len(gateway_with_tools.memory.list_experiments())
        assert after > before

    def test_heartbeat_starts(self, gateway):
        gateway.start_heartbeat()
        assert gateway.heartbeat._running is True
        gateway.heartbeat.stop()
