"""Tests for tool modules: dataset_discovery, auto_fixer, trainer, dashboard."""

import json
import pytest


class TestAutoFixer:
    """Error auto-fix patterns and resolution."""

    def test_cuda_oom_fix(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        result = fx.fix("CUDA out of memory. Tried to allocate 2.00 GiB.", {"batch_size": 4})
        assert result["fixed"] is True
        assert result["conf"] >= 0.8
        assert result["desc"] is not None

    def test_import_error_fix(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        result = fx.fix("No module named 'bitsandbytes'", {"batch_size": 2})
        assert result["fixed"] is True

    def test_nan_loss_fix(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        result = fx.fix("Loss is NaN at step 47", {"learning_rate": 2e-4})
        assert result["fixed"] is True
        assert result["conf"] >= 0.7

    def test_unknown_error_returns_not_fixed(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        result = fx.fix("Some weird non-matching error message", {})
        assert result["fixed"] is False

    def test_fix_logs_to_memory(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        fx.fix("CUDA out of memory", {"batch_size": 4})
        unfixed = memory.get_unfixed_errors()
        assert len(unfixed) == 0  # fixed errors should not be unfixed

    def test_dtype_mismatch_fix(self):
        from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool
        from openclaw_colab_agent.core.memory import MemoryStore
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        fx = AutoFixerTool(memory)
        result = fx.fix("expected scalar type Half but found Float", {"torch_dtype": "float16"})
        assert result["fixed"] is True


class TestDatasetDiscovery:
    """Dataset discovery and loading."""

    def test_search_returns_list(self):
        from openclaw_colab_agent.tools.dataset_discovery import DatasetDiscoveryTool
        from openclaw_colab_agent.core.memory import MemoryStore
        from pathlib import Path
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        tool = DatasetDiscoveryTool(memory, Path(tempfile.mkdtemp()), "")
        results = tool.search("math reasoning", max_results=5)
        assert isinstance(results, list)

    def test_mock_fallback(self, monkeypatch):
        from openclaw_colab_agent.tools.dataset_discovery import DatasetDiscoveryTool
        from openclaw_colab_agent.core.memory import MemoryStore
        from pathlib import Path
        import tempfile
        memory = MemoryStore(tempfile.mkdtemp())
        tool = DatasetDiscoveryTool(memory, Path(tempfile.mkdtemp()), "")
        monkeypatch.setattr(tool, "_mock_search", lambda q: (_ for _ in ()).throw(Exception("HF down")))
        results = tool.search("math", max_results=5)
        assert isinstance(results, list)
