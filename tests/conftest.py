"""
Pytest configuration with shared fixtures.
Patterns from huggingface/transformers conftest.py (120k+ stars).
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures: Temporary workspace
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def tmp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_memory_store(tmp_workspace: Path):
    """Create a MemoryStore-like object backed by temp JSON files."""
    from openclaw_colab_agent.core.memory import MemoryStore
    store = MemoryStore(tmp_workspace / "memory")
    return store


@pytest.fixture(scope="function")
def gateway(tmp_workspace: Path):
    """Create a fully wired Gateway for testing."""
    from openclaw_colab_agent.config import AgentConfig
    from openclaw_colab_agent.core.gateway import Gateway

    cfg = AgentConfig()
    cfg.workspace_root = tmp_workspace
    cfg.memory_dir = tmp_workspace / "memory"
    cfg.skills_dir = tmp_workspace / "skills"
    cfg.models_dir = tmp_workspace / "models"
    cfg.datasets_dir = tmp_workspace / "datasets"
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    cfg.huggingface_token = os.environ.get("HF_TOKEN", "")

    gw = Gateway(cfg)
    return gw


@pytest.fixture(scope="function")
def gateway_with_tools(gateway):
    """Gateway with all tools registered (mock trainer)."""
    from openclaw_colab_agent.tools.dataset_discovery import DatasetDiscoveryTool
    from openclaw_colab_agent.tools.trainer import TrainerTool
    from openclaw_colab_agent.tools.auto_fixer import AutoFixerTool

    ds_dir = Path(gateway.config.datasets_dir) if not isinstance(gateway.config.datasets_dir, Path) else gateway.config.datasets_dir
    ds_tool = DatasetDiscoveryTool(gateway.memory, ds_dir, gateway.config.huggingface_token)
    tr_tool = TrainerTool(gateway.memory, gateway.config.models_dir)
    fx_tool = AutoFixerTool(gateway.memory)

    gateway.register_tool("discover", "Search datasets", lambda **kw: ds_tool.search(keywords=kw.get("query", "")))
    gateway.register_tool("train", "Train LLM", lambda **kw: {"status": "queued", "note": "mock"})
    gateway.register_tool("prepare", "Prepare env", lambda **kw: {"status": "ready", "gpu": "mock"})
    gateway.register_tool("auto_fix", "Fix errors", lambda **kw: fx_tool.fix(kw.get("error", ""), kw.get("context", {})))

    return gateway


@pytest.fixture(scope="function")
def sample_domain_config() -> Dict[str, Any]:
    """Sample domain configuration for testing."""
    return {
        "name": "Test Domain",
        "emoji": "[T]",
        "desc": "Test domain for unit tests",
        "search": ["test"],
        "datasets": ["test_dataset"],
        "models": ["unsloth/mistral-7b-bnb-4bit"],
        "hints": {"learning_rate": 2e-4, "lora_r": 8, "max_seq_length": 512},
        "format": "### Instruction\n{q}\n### Response\n{a}",
    }


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')")
    config.addinivalue_line("markers", "network: mark test as requiring network (deselect with '-m \"not network\"')")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU and network tests when not available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
