"""Tests for new modules: state machine, chain, plugin, device."""

import pytest
import tempfile
from pathlib import Path


class TestStateMachine:
    """State machine transitions (Home-Assistant pattern)."""

    def test_initial_state(self):
        from epsionic.core.state import StateMachine
        sm = StateMachine("idle")
        assert sm.state == "idle"

    @pytest.mark.parametrize("start,target,expected", [
        ("idle", "running", "running"),
        ("running", "idle", "idle"),
        ("running", "paused", "paused"),
        ("paused", "running", "running"),
        ("paused", "stopped", "stopped"),
        ("stopped", "idle", "idle"),
        ("error", "idle", "idle"),
    ])
    def test_valid_transitions(self, start, target, expected):
        from epsionic.core.state import StateMachine
        sm = StateMachine(start)
        sm.transition(target)
        assert sm.state == expected

    @pytest.mark.parametrize("start,invalid", [
        ("idle", "paused"),
        ("stopped", "paused"),
    ])
    def test_invalid_transitions(self, start, invalid):
        from epsionic.core.state import StateMachine, StateTransitionError
        sm = StateMachine(start)
        with pytest.raises(StateTransitionError):
            sm.transition(invalid)

    def test_history(self):
        from epsionic.core.state import StateMachine
        sm = StateMachine("idle")
        sm.transition("running")
        sm.transition("idle")
        assert sm.history == ["idle", "running", "idle"]


class TestChain:
    """Chain composition (LangChain pattern)."""

    def test_single_chain(self):
        from epsionic.core.chain import Chain
        c = Chain("double", lambda x: {"value": x["num"] * 2})
        result = c.invoke({"num": 5})
        assert result["value"] == 10

    def test_pipe_operator(self):
        from epsionic.core.chain import Chain
        add1 = Chain("add1", lambda x: {"v": x["v"] + 1})
        mul2 = Chain("mul2", lambda x: {"v": x["v"] * 2})
        add1 >> mul2
        result = add1.invoke({"v": 3})
        assert result["v"] == 8

    def test_chain_builder(self):
        from epsionic.core.chain import ChainBuilder
        b = ChainBuilder()
        b.add_step("step1", lambda ctx: {"a": ctx["x"] + 1})
        b.add_step("step2", lambda ctx: {"b": ctx["a"] * 2})
        result = b.run({"x": 5})
        assert result.get("b") == 12

    def test_pipeline(self):
        from epsionic.core.chain import Pipeline
        p = Pipeline()
        p.add("init", lambda ctx: {"value": ctx["start"]})
        p.add("add", lambda ctx: {"value": ctx["value"] + 10})
        result = p.execute({"start": 5})
        assert result.get("value") == 15


class TestPlugin:
    """Plugin system (AutoGPT pattern)."""

    def test_plugin_manager_creation(self):
        from epsionic.core.plugin import PluginManager
        with tempfile.TemporaryDirectory() as td:
            pm = PluginManager(Path(td))
            assert len(pm.plugins) == 0

    def test_create_example(self):
        from epsionic.core.plugin import PluginManager
        with tempfile.TemporaryDirectory() as td:
            pm = PluginManager(Path(td))
            pm.create_example()
            assert (Path(td) / "example.py").exists()

    def test_discover_empty(self):
        from epsionic.core.plugin import PluginManager
        with tempfile.TemporaryDirectory() as td:
            pm = PluginManager(Path(td))
            found = pm.discover()
            assert found == []


class TestDevice:
    """Device manager (PyTorch pattern)."""

    def test_device_creation(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        assert dm.backend in ("cpu", "cuda", "mps")
        assert dm.name is not None

    def test_recommended_batch(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        batch = dm.recommended_batch_size()
        assert isinstance(batch, int)
        assert batch >= 1

    def test_summary(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        summary = dm.summary()
        assert "Device" in summary
        assert "Backend" in summary
