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
        from epsionic.core.domain import DOMAINS
        result = gateway_with_tools.run_domain("math", DOMAINS["math"])
        assert "objective" in result
        assert "domain" in result
        assert "experiment_id" in result
        assert "steps" in result
        assert len(result["steps"]) >= 4

    def test_run_domain_creates_experiment(self, gateway_with_tools):
        from epsionic.core.domain import DOMAINS
        before = len(gateway_with_tools.memory.list_experiments())
        gateway_with_tools.run_domain("math", DOMAINS["math"])
        after = len(gateway_with_tools.memory.list_experiments())
        assert after > before

    def test_heartbeat_starts(self, gateway):
        gateway.start_heartbeat()
        assert gateway.heartbeat._running is True
        gateway.heartbeat.stop()

    # ── Budget halt tests ─────────────────────────────────────────

    def test_check_budget_under_limit(self, gateway):
        """check_budget returns True when cost is under budget."""
        assert gateway.check_budget() is True

    def test_check_budget_over_limit(self, gateway):
        """check_budget returns False when cost exceeds budget."""
        from epsionic.core.gateway import _BUDGET_HALT
        # Reset global flag
        import epsionic.core.gateway as gw_mod
        gw_mod._BUDGET_HALT = False

        # Set up cost tracker with low limit
        from epsionic.tools.cost_tracker import CostTracker
        gateway._cost_tracker = CostTracker(memory_store=gateway.memory, budget_limit=1.0)

        # Track enough cost to exceed $1 budget (T4 = $0.35/hr, need ~3 hrs)
        gateway._cost_tracker.track_training(duration_hours=3.0)

        assert gateway.check_budget() is False
        assert gw_mod._BUDGET_HALT is True

    def test_budget_halt_persists(self, gateway):
        """Once _BUDGET_HALT is set, check_budget always returns False."""
        import epsionic.core.gateway as gw_mod
        gw_mod._BUDGET_HALT = True
        assert gateway.check_budget() is False

    def test_budget_halt_skips_training_step(self, gateway_with_tools):
        """_run_chain_step returns halted error for train step."""
        import epsionic.core.gateway as gw_mod
        gw_mod._BUDGET_HALT = True
        from epsionic.core.domain import DOMAINS

        ctx = {"domain": "math", "domain_config": DOMAINS["math"]}
        result = gateway_with_tools._run_chain_step("train", ctx)
        assert result.get("halted") is True
        assert "Budget" in result.get("error", "")

    def test_budget_halt_skips_expensive_tools_in_run_objective(self, gateway):
        """run_objective skips train/prepare/evaluate steps when budget exceeded."""
        import epsionic.core.gateway as gw_mod
        gw_mod._BUDGET_HALT = True

        # Register a train tool so the brain can find it
        gateway.register_tool("train", "mock train", lambda **kw: {"status": "ok"})
        gateway.register_tool("discover", "mock discover", lambda **kw: {"datasets": ["test"]})

        result = gateway.run_objective("test halt", max_steps=3)
        for step in result.get("steps", []):
            if step.get("tool") in ("train", "prepare", "evaluate"):
                assert step.get("status") == "skipped"
                assert "Budget" in step.get("error", "")

    def test_budget_halt_writes_agent_state(self, gateway):
        """Budget halt writes agent state record."""
        import epsionic.core.gateway as gw_mod
        gw_mod._BUDGET_HALT = False

        from epsionic.tools.cost_tracker import CostTracker
        gateway._cost_tracker = CostTracker(memory_store=gateway.memory, budget_limit=0.5)
        gateway._cost_tracker.track_training(duration_hours=2.0)

        gateway.check_budget()

        # Check agent state was written
        data = gateway.memory.read_agent_state("budget_halt")
        assert data is not None
        assert "cost" in data
        assert "limit" in data
        assert data["limit"] == 0.5
