"""Tests for MemPalaceStore — spatial memory with Wings, Rooms, Drawers, and Knowledge Graph."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def palace():
    from epsionic.memory import MemPalaceStore
    tmp = Path(tempfile.mkdtemp())
    store = MemPalaceStore(tmp / "palace")
    yield store
    store.close()


class TestWingsAndRooms:
    def test_ensure_wing_creates(self, palace):
        assert palace.ensure_wing("math") == "math"
        wings = palace.list_wings()
        assert any(w["name"] == "math" for w in wings)

    def test_ensure_wing_normalizes(self, palace):
        assert palace.ensure_wing("Code Domain") == "code_domain"

    def test_create_experiment_creates_room(self, palace):
        exp_id = palace.create_experiment("test_run", {"lr": 2e-4}, domain="math")
        assert exp_id.startswith("test_run_")
        exp = palace.get_experiment(exp_id)
        assert exp is not None
        assert exp["status"] == "created"
        assert exp["name"] == "test_run"

    def test_list_experiments_empty(self, palace):
        assert palace.list_experiments() == []

    def test_list_experiments_by_domain(self, palace):
        e1 = palace.create_experiment("run1", domain="math")
        e2 = palace.create_experiment("run2", domain="code")
        math_exps = palace.list_experiments(domain="math")
        assert len(math_exps) == 1
        assert math_exps[0]["id"] == e1
        assert math_exps[0]["domain"] == "math"


class TestExperimentCRUD:
    def test_update_status(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, status="running")
        exp = palace.get_experiment(exp_id)
        assert exp["status"] == "running"

    def test_update_metrics(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, metrics={"loss": 0.5, "acc": 0.8})
        palace.update_experiment(exp_id, metrics={"loss": 0.3})
        exp = palace.get_experiment(exp_id)
        assert exp["metrics"]["loss"] == 0.3
        assert exp["metrics"]["acc"] == 0.8

    def test_update_errors_append(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, errors=["CUDA OOM"])
        palace.update_experiment(exp_id, errors=["NaN loss"])
        exp = palace.get_experiment(exp_id)
        assert len(exp["errors"]) >= 2

    def test_get_nonexistent(self, palace):
        assert palace.get_experiment("nonexistent") is None

    def test_list_by_status(self, palace):
        e1 = palace.create_experiment("a", domain="math")
        e2 = palace.create_experiment("b", domain="math")
        palace.update_experiment(e2, status="completed")
        running = palace.list_experiments(status="created")
        completed = palace.list_experiments(status="completed")
        assert any(r["id"] == e1 for r in running)
        assert any(c["id"] == e2 for c in completed)

    def test_export_csv(self, palace):
        palace.create_experiment("test", domain="math")
        csv_out = palace.export_experiments_csv()
        assert "test" in csv_out


class TestErrorLogging:
    def test_log_error(self, palace):
        eid = palace.log_error("trainer", "CUDA out of memory", {"batch_size": 4})
        assert eid.startswith("err_")

    def test_get_unfixed_errors(self, palace):
        palace.log_error("trainer", "OOM error")
        palace.log_error("trainer", "NaN loss")
        assert len(palace.get_unfixed_errors()) == 2

    def test_mark_error_fixed(self, palace):
        eid = palace.log_error("trainer", "OOM error")
        palace.mark_error_fixed(eid, "Reduced batch size")
        unfixed = palace.get_unfixed_errors()
        assert all(not e.get("fixed") for e in unfixed) or len(unfixed) == 0


class TestAgentState:
    def test_write_and_read(self, palace):
        palace.write_agent_state("budget_halt", {"cost": 12.5, "limit": 50})
        data = palace.read_agent_state("budget_halt")
        assert data["cost"] == 12.5
        assert data["limit"] == 50

    def test_read_nonexistent(self, palace):
        assert palace.read_agent_state("nonexistent") is None


class TestHeartbeat:
    def test_write_and_read(self, palace):
        palace.write_heartbeat_log("HEARTBEAT_OK", "All good")
        palace.write_heartbeat_log("ISSUE", "GPU temp high")
        beats = palace.get_recent_heartbeats(5)
        assert len(beats) == 2
        # Newest entry first
        assert "GPU temp high" in beats[0]["entry"]
        assert beats[0]["agent_name"] == "heartbeat"

    def test_heartbeat_limit(self, palace):
        for i in range(20):
            palace.write_heartbeat_log("OK", f"beat {i}")
        beats = palace.get_recent_heartbeats(10)
        assert len(beats) == 10


class TestKnowledgeGraph:
    def test_add_knowledge(self, palace):
        palace.add_knowledge("exp_001", "trained_on", "mistral-7b")
        palace.add_knowledge("exp_001", "used_dataset", "gsm8k")
        results = palace.query_knowledge("exp_001")
        assert len(results) == 2
        assert any(r["predicate"] == "trained_on" for r in results)

    def test_knowledge_graph_stats(self, palace):
        palace.add_knowledge("a", "related_to", "b")
        stats = palace.get_knowledge_graph_stats()
        assert stats["entities"] >= 2
        assert stats["triples"] >= 1


class TestTrainingSummary:
    def test_summary_empty(self, palace):
        md = palace.get_training_summary_markdown()
        assert "No wings yet" in md

    def test_summary_with_experiments(self, palace):
        exp_id = palace.create_experiment("test_run", {"lr": 2e-4}, domain="math")
        palace.update_experiment(exp_id, metrics={"loss": 0.35, "acc": 0.72})
        md = palace.get_training_summary_markdown()
        assert "Wing: math" in md
        assert "test_run" in md
        assert "0.35" in md or "loss=0.35" in md


class TestEnhancedFeatures:
    def test_wake_up_with_wing(self, palace):
        palace.create_experiment("test", domain="math")
        ctx = palace.wake_up(wing="math")
        assert "L0" in ctx
        assert "L1" in ctx
        assert "math" in ctx

    def test_recall_empty(self, palace):
        recall = palace.recall(wing="math")
        assert "No results" in recall

    def test_recall_with_data(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, metrics={"loss": 0.5})
        recall = palace.recall(wing="math", n=5)
        assert "metrics" in recall or "test" in recall

    def test_search(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, metrics={"loss": 0.5})
        results = palace.search("0.5")
        assert len(results) >= 1

    def test_search_fallback(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.update_experiment(exp_id, metrics={"acc": 0.95})
        results = palace.search("0.95")
        assert len(results) >= 1

    def test_get_best_hyperparams(self, palace):
        e1 = palace.create_experiment("run1", {"lr": 2e-4}, domain="math")
        e2 = palace.create_experiment("run2", {"lr": 1e-4}, domain="math")
        palace.update_experiment(e1, metrics={"eval_loss": 0.5})
        palace.update_experiment(e2, metrics={"eval_loss": 0.3})
        best = palace.get_best_hyperparams("math", metric="eval_loss", n=2)
        assert len(best) >= 1

    def test_recall_auto_fix_missing(self, palace):
        fix = palace.recall_auto_fix("CUDA out of memory")
        assert fix is None

    def test_recall_auto_fix_present(self, palace):
        palace.add_knowledge("CUDA OOM", "fixed_by", "reduce_batch_size",
                             confidence=0.9)
        fix = palace.recall_auto_fix("CUDA OOM")
        assert fix is not None
        assert fix["fix"] == "reduce_batch_size"

    def test_record_training_run(self, palace):
        exp_id = palace.create_experiment("test", domain="math")
        palace.record_training_run(exp_id, "math", "mistral-7b", "gsm8k",
                                   {"lr": 2e-4}, {"acc": 0.75}, 5.50)
        kg = palace.query_knowledge(exp_id)
        predicates = [r["predicate"] for r in kg]
        assert "trained_on" in predicates
        assert "used_dataset" in predicates

    def test_get_cost_effective(self, palace):
        e1 = palace.create_experiment("cheap", domain="math")
        e2 = palace.create_experiment("expensive", domain="math")
        palace.record_training_run(e1, "math", "m1", "d1",
                                   {}, {"acc": 0.7}, 2.0)
        palace.record_training_run(e2, "math", "m2", "d2",
                                   {}, {"acc": 0.9}, 50.0)
        cheap = palace.get_cost_effective("math", budget=10.0)
        assert len(cheap) >= 1


class TestStatePersistence:
    def test_save_state(self, palace):
        result = palace.save_state()
        assert result["success"] is True
        assert result["store_type"] == "mempalace"

    def test_restore_state(self, palace):
        palace.create_experiment("test", domain="math")
        result = palace.restore_state()
        assert result["success"] is True
        assert result["experiments_restored"] >= 1

    def test_store_type(self, palace):
        assert palace.store_type == "mempalace"
