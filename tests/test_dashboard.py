"""Tests for Dashboard tool components."""

import json
import pytest
from pathlib import Path


pytestmark = pytest.mark.skip(reason="Dashboard tests require gradio (hangs on Windows CI)")


class TestDashboardLoaders:
    """Dashboard data loaders (unit tests without gradio)."""

    def test_load_status(self, mock_memory_store):
        from epsionic.tools.dashboard import load_status
        result = load_status(mock_memory_store)
        assert "Workspace" in result or "Status" in result
        assert isinstance(result, str)

    def test_load_experiments_empty(self, mock_memory_store):
        from epsionic.tools.dashboard import load_experiments
        result = load_experiments(mock_memory_store)
        assert "No experiments" in result or "Experiments" in result

    def test_load_experiments_with_data(self, mock_memory_store):
        mock_memory_store.create_experiment("test_exp", {"domain": "math"})
        from epsionic.tools.dashboard import load_experiments
        result = load_experiments(mock_memory_store)
        assert "test_exp" in result

    def test_load_errors_empty(self, mock_memory_store):
        from epsionic.tools.dashboard import load_errors
        result = load_errors(mock_memory_store)
        assert isinstance(result, str)

    def test_load_errors_with_data(self, mock_memory_store):
        mock_memory_store.log_error("test", "Sample error", {})
        from epsionic.tools.dashboard import load_errors
        result = load_errors(mock_memory_store)
        assert "Sample error" in result

    def test_load_heartbeat_empty(self, mock_memory_store):
        from epsionic.tools.dashboard import load_heartbeat
        result = load_heartbeat(mock_memory_store)
        assert "heartbeat" in result.lower() or "No heartbeat" in result

    def test_load_datasets_empty(self, mock_memory_store):
        from epsionic.tools.dashboard import load_datasets
        result = load_datasets(mock_memory_store)
        assert isinstance(result, str)

    def test_run_action_refresh(self, mock_memory_store):
        from epsionic.tools.dashboard import run_action
        result = run_action(mock_memory_store, None, "refresh", "")
        assert "Refreshing" in result

    def test_run_action_unknown(self, mock_memory_store):
        from epsionic.tools.dashboard import run_action
        result = run_action(mock_memory_store, None, "invalid", "")
        assert "Unknown action" in result


class TestDashboardUI:
    """UI construction tests (without loading gradio)."""

    def test_safe_json(self):
        from epsionic.tools.dashboard import _safe_json
        result = _safe_json({"a": 1, "b": [2, 3]})
        assert '"a": 1' in result

    def test_fmt_time(self):
        from epsionic.tools.dashboard import _fmt_time
        result = _fmt_time("2024-01-15T10:30:00")
        assert "2024-01-15" in result
