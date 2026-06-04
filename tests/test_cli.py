"""Tests for CLI entry point and argument parsing."""

import os
import sys
import json
import pytest
from pathlib import Path


class TestParseArgs:
    """CLI argument parsing."""

    def test_default_args(self):
        from openclaw_colab_agent.cli import parse_args
        args = parse_args([])
        assert args.autonomous is False
        assert args.interactive is False
        assert args.dashboard is False
        assert args.serve is False
        assert args.domain is None

    def test_autonomous_flag(self):
        from openclaw_colab_agent.cli import parse_args
        args = parse_args(["-a"])
        assert args.autonomous is True

    def test_domain_flag(self):
        from openclaw_colab_agent.cli import parse_args
        args = parse_args(["-d", "math"])
        assert args.domain == "math"

    def test_verbose_flag(self):
        from openclaw_colab_agent.cli import parse_args
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_continuous_requires_autonomous(self):
        from openclaw_colab_agent.cli import parse_args
        args = parse_args(["-a", "--continuous"])
        assert args.autonomous and args.continuous is True


class TestMain:
    """End-to-end CLI main function."""

    def test_setup_only_exits_early(self, monkeypatch):
        from openclaw_colab_agent.cli import main
        monkeypatch.setattr("sys.argv", ["openclaw", "--setup-only"])
        monkeypatch.setattr("openclaw_colab_agent.cli._auto_install", lambda **kw: None)
        rc = main()
        assert rc == 0

    def test_unknown_domain_returns_error(self, monkeypatch):
        from openclaw_colab_agent.cli import main
        monkeypatch.setattr("openclaw_colab_agent.cli._auto_install", lambda **kw: None)
        mock_gw = type("MockGW", (), {"state": type("MockState", (), {"tools_loaded": 0})(), "device": type("MockDev", (), {"name": "CPU", "backend": "cpu", "vram_gb": 0})(), "plugins": {}})()
        monkeypatch.setattr("openclaw_colab_agent.cli.build_gateway", lambda cfg: mock_gw)
        rc = main(["-d", "nonexistent_domain_xyz"])
        assert rc == 1
