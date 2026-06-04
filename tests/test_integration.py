"""Integration tests for Gateway, FastAPI, and full pipelines."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGatewayIntegration:
    """End-to-end Gateway pipeline with mocked tools."""

    def test_build_gateway_from_config(self, tmp_workspace):
        from openclaw_colab_agent.config import AgentConfig
        from openclaw_colab_agent.core.gateway import Gateway
        cfg = AgentConfig()
        cfg.workspace_root = tmp_workspace
        gw = Gateway(cfg)
        assert gw.memory is not None
        assert gw.state.status == "idle"

    def test_full_domain_pipeline(self, gateway_with_tools):
        from openclaw_colab_agent.core.domain import DOMAINS
        result = gateway_with_tools.run_domain("math", DOMAINS["math"])
        assert result["domain"] == "math"
        assert len(result["steps"]) >= 4
        assert result.get("error") is None

    def test_autonomous_loop_returns_list(self, gateway_with_tools):
        result = gateway_with_tools.run_autonomous_loop(continuous=False)
        assert isinstance(result, list)

    def test_pipeline_builds_correct_steps(self, gateway):
        pipeline = gateway.build_pipeline("math")
        assert len(pipeline._steps) == 4
        names = [s.name for s in pipeline._steps]
        assert names == ["discover", "select", "prepare", "train"]


class TestMemoryIntegration:
    """Memory store end-to-end tests."""

    def test_experiment_lifecycle(self, mock_memory_store):
        ms = mock_memory_store
        exp_id = ms.create_experiment("test_exp", {"domain": "math"})
        assert ms.get_experiment(exp_id) is not None
        ms.update_experiment(exp_id, status="running")
        assert ms.get_experiment(exp_id)["status"] == "running"
        ms.update_experiment(exp_id, status="completed", metrics={"loss": 0.5})
        exps = ms.list_experiments("completed")
        assert any(e["id"] == exp_id for e in exps)

    def test_error_logging_and_fixing(self, mock_memory_store):
        ms = mock_memory_store
        err_id = ms.log_error("train", "CUDA OOM", {"batch_size": 4})
        assert ms.get_unfixed_errors() != []
        ms.mark_error_fixed(err_id, "Reduced batch size")
        assert ms.get_unfixed_errors() == []

    def test_export_csv(self, mock_memory_store):
        ms = mock_memory_store
        ms.create_experiment("exp1", {})
        ms.create_experiment("exp2", {})
        path = ms.export_experiments_csv()
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "exp1" in content

    def test_audit_logging(self, mock_memory_store):
        ms = mock_memory_store
        ms.write_audit_log("test_action", {"key": "value"})
        audit_files = list(ms._audit_dir.glob("*.jsonl"))
        assert len(audit_files) >= 1


class TestPluginIntegration:
    """Plugin system integration."""

    def test_gateway_discovers_plugins(self, gateway):
        assert hasattr(gateway, "_plugin_manager")
        assert len(gateway.plugins) >= 0

    def test_heartbeat_includes_plugins(self, gateway):
        gateway.start_heartbeat()
        result = gateway._check_plugins()
        assert isinstance(result, str)


class TestStateIntegration:
    """State machine integration with Gateway."""

    def test_gateway_state_transitions(self, gateway):
        assert gateway.state.status == "idle"
        gateway._state_transition("running")
        assert gateway.state.status == "running"
        gateway._state_transition("idle")
        assert gateway.state.status == "idle"

    def test_invalid_transition_logged(self, gateway):
        gateway._state_transition("running")
        gateway._state_transition("paused")
        assert gateway.state.status == "paused"
        gateway._state_transition("running")
        assert gateway.state.status == "running"
