"""End-to-end tests for the FastAPI server using TestClient."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class MockBrain:
    def generate(self, prompt, max_tokens=512):
        return f"Mock response to: {prompt[:50]}"


class TestAPIServer:
    """Integration tests using TestClient against a mock FastAPI server."""

    @pytest.fixture
    def gw(self):
        gw = MagicMock()
        gw._tool_registry = {}
        gw.state.status = "running"
        gw.state.started_at = "2025-01-01T00:00:00"
        gw.state.sessions_created = 5
        gw.state.errors_encountered = 0
        gw.state.domain = "math"
        gw.state.tools_loaded = 3
        gw._budget_halt = False
        gw.api_keys = set()
        gw.memory = MagicMock()
        gw.memory.backup.return_value = None
        gw.memory._get_schema_version = 3
        gw.brain = MockBrain()
        gw.config.workspace_root = Path(__file__).parent / "tmp_test_ws"
        return gw

    def test_ready(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.get("/ready")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"

    def test_version(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.get("/version")
            assert resp.status_code == 200
            data = resp.json()
            assert "version" in data
            assert data["api_version"] == "v1"

    def test_metrics(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.get("/metrics")
            assert resp.status_code == 200
            assert "epsionic_build_info" in resp.text

    def test_openai_models(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            assert "data" in resp.json()

    def test_openai_chat(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert "choices" in data

    def test_health_auth_block(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        with patch.dict("os.environ", {"EPSIONIC_API_KEY": "secret"}):
            app = make_app(gw)
            with TestClient(app) as client:
                resp = client.get("/ready", headers={"Authorization": "Bearer wrong"})
                assert resp.status_code == 403

    def test_cors_headers(self, gw):
        from fastapi.testclient import TestClient
        from epsionic.cli import make_app
        app = make_app(gw)
        with TestClient(app) as client:
            resp = client.get("/ready", headers={"Origin": "http://example.com"})
            assert resp.headers.get("access-control-allow-origin") == "*"
