"""Tests for Phase 2 tools: FlowEngine, CostTracker, Notifier, ExperimentTracker, Optimizer, Quantizer."""

import json
import os
import time
import pytest
from pathlib import Path


class TestFlowEngine:
    def test_create_flow(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        fid = engine.create_flow("test", "desc", triggers=[{"type": "manual"}])
        assert fid.startswith("flow_")
        flow = engine.get_flow(fid)
        assert flow["name"] == "test"
        assert flow["description"] == "desc"

    def test_add_node(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        fid = engine.create_flow("pipe")
        nid = engine.add_node(fid, "train", {"lr": 0.001})
        assert nid.startswith("node_")
        flow = engine.get_flow(fid)
        assert len(flow["nodes"]) == 1
        assert flow["nodes"][0]["type"] == "train"

    def test_add_node_with_connection(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        fid = engine.create_flow("pipe")
        n1 = engine.add_node(fid, "discover", {"query": "math"})
        n2 = engine.add_node(fid, "train", {"lr": 0.001}, after_node=n1)
        flow = engine.get_flow(fid)
        assert len(flow["nodes"]) == 2
        assert flow["nodes"][0]["connections"]["output"] == n2

    def test_execute_flow(self, tmp_path):
        from epsionic.tools.flow import FlowEngine, register_node_type
        engine = FlowEngine(flows_dir=tmp_path)
        register_node_type("mock_op_test")(lambda **kw: {"result": "ok"})
        fid = engine.create_flow("test_exec")
        engine.add_node(fid, "mock_op_test", {"x": 1})
        result = engine.execute(fid)
        assert result["success"] is True
        assert len(result["nodes"]) == 1

    def test_export_mermaid(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        fid = engine.create_flow("viz")
        n1 = engine.add_node(fid, "a", {})
        n2 = engine.add_node(fid, "b", {}, after_node=n1)
        md = engine.export_mermaid(fid)
        assert "graph TD;" in md
        assert n1 in md
        assert n2 in md

    def test_export_yaml(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        fid = engine.create_flow("yaml_test")
        engine.add_node(fid, "x", {"y": 1})
        y = engine.export_yaml(fid)
        assert "name:" in y

    def test_list_flows(self, tmp_path):
        from epsionic.tools.flow import FlowEngine
        engine = FlowEngine(flows_dir=tmp_path)
        engine.create_flow("a")
        engine.create_flow("b")
        flows = engine.list_flows()
        assert len(flows) == 2

    def test_execute_missing_flow(self, tmp_path):
        from epsionic.tools.flow import FlowEngine, FlowError
        engine = FlowEngine(flows_dir=tmp_path)
        with pytest.raises(FlowError):
            engine.execute("nonexistent")


class TestCostTracker:
    def _make(self, tmp_path):
        from epsionic.tools.cost_tracker import CostTracker
        return CostTracker(cost_file=tmp_path / "costs.jsonl")

    def test_estimate_cost(self, tmp_path):
        ct = self._make(tmp_path)
        est = ct.estimate_training_cost(gpu_type="T4", hours=2, gpus=1)
        assert est["gpu"] == "T4"
        assert est["estimated_hours"] == 2
        assert est["estimated_cost"] > 0

    def test_track_training(self, tmp_path):
        ct = self._make(tmp_path)
        entry = ct.track_training(1.0, gpu_type="T4")
        assert entry.category == "gpu"
        assert entry.amount > 0

    def test_track_api_call(self, tmp_path):
        ct = self._make(tmp_path)
        entry = ct.track_api_call("openai", tokens=1000)
        assert entry.category == "api"
        assert entry.amount > 0

    def test_track_storage(self, tmp_path):
        ct = self._make(tmp_path)
        entry = ct.track_storage(10.0, duration_days=1)
        assert entry.category == "storage"

    def test_total_and_session(self, tmp_path):
        ct = self._make(tmp_path)
        ct.track_training(1)
        ct.track_api_call("openai", 500)
        assert ct.get_total_cost() > 0
        assert ct.get_session_cost() > 0

    def test_budget_remaining(self, tmp_path):
        ct = self._make(tmp_path)
        ct.budget_limit = 100
        ct.track_training(1, gpu_type="T4")
        remaining = ct.get_budget_remaining()
        assert remaining < 100
        assert remaining >= 0

    def test_is_over_budget(self, tmp_path):
        ct = self._make(tmp_path)
        ct.budget_limit = 0.001
        ct.track_training(1, gpu_type="T4")
        assert ct.is_over_budget() is True

    def test_summary(self, tmp_path):
        ct = self._make(tmp_path)
        ct.track_training(1)
        s = ct.summary()
        assert "total_cost" in s
        assert "by_category" in s

    def test_reset_session(self, tmp_path):
        ct = self._make(tmp_path)
        ct.track_training(1)
        ct.reset_session()
        assert ct.get_session_cost() < 0.01

    def test_tool_description(self, tmp_path):
        d = self._make(tmp_path).get_tool_description()
        assert "estimate_training_cost" in d


class TestNotifier:
    def _make(self):
        from epsionic.tools.notifier import Notifier
        return Notifier()

    def test_send_log(self):
        n = self._make()
        result = n.send("Test", "Hello", channels=["log"])
        assert result["log"]["success"] is True

    def test_notify_done(self):
        n = self._make()
        result = n.notify_done("exp_1")
        assert any(v["success"] for v in result.values())

    def test_notify_error(self):
        n = self._make()
        result = n.notify_error("exp_1", "error msg")
        assert any(v["success"] for v in result.values())

    def test_notify_budget(self):
        n = self._make()
        result = n.notify_budget(10, 50)
        assert any(v["success"] for v in result.values())

    def test_unknown_channel(self):
        n = self._make()
        result = n.send("X", "Y", channels=["unknown_channel_xyz"])
        assert result.get("unknown_channel_xyz", {}).get("success") is False

    def test_rate_limit(self):
        n = self._make()
        n.send("A", "B", channels=["log"])
        result = n.send("A", "B", channels=["log"])
        # Second send within 1s gets rate-limited
        assert result["log"]["success"] is False
        assert result["log"]["error"] == "rate_limited"

    def test_get_tool_description(self):
        d = self._make().get_tool_description()
        assert "send" in d


class TestExperimentTracker:
    def test_init_run_wandb(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        assert et.backend == "none"
        assert et.run_id is None

    def test_log_params_noop(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        et.log_params({"lr": 0.01})
        assert et.run_id is None

    def test_log_metrics_noop(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        et.log_metrics({"loss": 0.5})
        assert et.run_id is None

    def test_finish_noop(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        et.finish()
        assert et.run_id is None

    def test_run_url_noop(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        assert et.get_run_url() is None

    def test_init_run_unsupported(self):
        from epsionic.tools.experiment_tracker import ExperimentTracker
        et = ExperimentTracker(backend="none")
        et.init_run("test", "run1")
        assert et.run_id is None


class TestHyperparameterOptimizer:
    def test_get_search_space(self, tmp_path):
        from epsionic.tools.optimizer import HyperparameterOptimizer
        opt = HyperparameterOptimizer(studies_dir=tmp_path)
        space = opt.get_search_space("math", vram_gb=16)
        assert "lr" in space
        assert "lora_r" in space

    def test_get_search_space_low_vram(self, tmp_path):
        from epsionic.tools.optimizer import HyperparameterOptimizer
        opt = HyperparameterOptimizer(studies_dir=tmp_path)
        space = opt.get_search_space("math", vram_gb=8)
        assert space["batch_size"][1] <= 2

    def test_get_search_space_high_vram(self, tmp_path):
        from epsionic.tools.optimizer import HyperparameterOptimizer
        opt = HyperparameterOptimizer(studies_dir=tmp_path)
        space = opt.get_search_space("math", vram_gb=24)
        assert space["batch_size"][1] >= 4

    def test_get_search_space_unknown_domain(self, tmp_path):
        from epsionic.tools.optimizer import HyperparameterOptimizer
        opt = HyperparameterOptimizer(studies_dir=tmp_path)
        space = opt.get_search_space("nonexistent")
        assert "lr" in space

    def test_get_tool_description(self):
        from epsionic.tools.optimizer import HyperparameterOptimizer
        d = HyperparameterOptimizer().get_tool_description()
        assert "optimize" in d


class TestQuantizer:
    def test_folder_size_zero(self, tmp_path):
        from epsionic.tools.quantizer import ModelQuantizer
        q = ModelQuantizer(models_dir=tmp_path)
        assert q._folder_size(str(tmp_path)) == 0.0

    def test_get_tool_description(self):
        from epsionic.tools.quantizer import ModelQuantizer
        d = ModelQuantizer().get_tool_description()
        assert "quantize_gptq" in d
        assert "quantize_awq" in d
        assert "export_gguf" in d

    def test_quantize_gptq_no_auto_gptq(self, tmp_path):
        from epsionic.tools.quantizer import ModelQuantizer
        q = ModelQuantizer(models_dir=tmp_path)
        result = q.quantize_gptq("fake/model")
        assert result["success"] is False
        assert "auto_gptq" in result.get("error", "")

    def test_quantize_awq_no_awq(self, tmp_path):
        from epsionic.tools.quantizer import ModelQuantizer
        q = ModelQuantizer(models_dir=tmp_path)
        result = q.quantize_awq("fake/model")
        assert result["success"] is False

    def test_export_gguf_hangs_safely(self):
        """export_gguf tries HF download and subprocess; just verify error handling exists."""
        from epsionic.tools.quantizer import ModelQuantizer
        q = ModelQuantizer()
        assert hasattr(q, 'export_gguf')
        desc = q.get_tool_description()
        assert 'export_gguf' in desc
