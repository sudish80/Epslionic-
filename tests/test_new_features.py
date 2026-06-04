"""Tests for all new 100k-star features: Recipes, Marketplace, Benchmarks, Prompts, Webhooks, Workspaces, Colab Deploy."""

import json
import os
import tempfile
from pathlib import Path

import pytest


# ── Recipe Manager ────────────────────────────────────────────────

class TestRecipeManager:
    @pytest.fixture
    def rm(self):
        from epsionic.tools.recipe_manager import RecipeManager
        tmp = Path(tempfile.mkdtemp())
        # Create a test recipe
        recipe = {"name": "Test Recipe", "model": "test-model", "dataset": "test-dataset",
                  "domain": "math", "training": {"learning_rate": 2e-4, "lora_r": 16}}
        (tmp / "test_recipe.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in recipe.items()).replace("'", ""), encoding="utf-8")
        return RecipeManager(tmp)

    def test_list_recipes(self, rm):
        recipes = rm.list_recipes()
        assert len(recipes) >= 1

    def test_list_recipes_by_domain(self, rm):
        recipes = rm.list_recipes(domain="math")
        assert len(recipes) >= 1
        assert recipes[0]["domain"] == "math"

    def test_get_recipe(self, rm):
        recipe = rm.get_recipe("Test Recipe")
        assert recipe is not None
        assert recipe["name"] == "Test Recipe"
        assert recipe["model"] == "test-model"

    def test_get_recipe_nonexistent(self, rm):
        assert rm.get_recipe("nonexistent") is None

    def test_apply_recipe(self, rm):
        config = rm.apply_recipe("Test Recipe", {"model": "override-model"})
        assert config["model"] == "test-model"  # recipe wins
        assert config["dataset"] == "test-dataset"
        assert config["recipe_name"] == "Test Recipe"

    def test_apply_recipe_merges_training(self, rm):
        config = rm.apply_recipe("Test Recipe", {"training": {"learning_rate": 1e-4, "num_epochs": 5}})
        assert config["training"]["learning_rate"] == 2e-4  # recipe wins
        assert config["training"]["num_epochs"] == 5  # domain_config fills gaps

    def test_get_tool_description(self, rm):
        desc = rm.get_tool_description()
        assert desc["name"] == "recipe_manager"
        assert "parameters" in desc


# ── Colab Deploy ──────────────────────────────────────────────────

class TestColabDeploy:
    def test_generate_notebook(self):
        from epsionic.tools.colab_deploy import generate_colab_notebook
        notebook = generate_colab_notebook(api_key="test-key")
        assert "test-key" in notebook
        assert "ngrok" in notebook
        assert "Gateway" in notebook or "gateway" in notebook.lower()

    def test_generate_notebook_writes_file(self):
        from epsionic.tools.colab_deploy import generate_colab_notebook
        tmp = Path(tempfile.mkdtemp()) / "deploy.py"
        notebook = generate_colab_notebook(output_path=str(tmp))
        assert tmp.exists()
        assert tmp.read_text(encoding="utf-8") == notebook


# ── Marketplace ───────────────────────────────────────────────────

class TestMarketplace:
    @pytest.fixture
    def mp(self):
        from epsionic.tools.marketplace import Marketplace
        return Marketplace(tempfile.mkdtemp())

    def test_init_creates_dirs(self, mp):
        assert mp.recipes_dir.exists()
        assert mp.plugins_dir.exists()

    def test_list_available_empty(self, mp):
        # Without network, list returns empty gracefully
        items = mp.list_available("recipes")
        assert isinstance(items, list)

    def test_search_empty(self, mp):
        results = mp.search("nonexistent")
        assert isinstance(results, list)


# ── Benchmark Runner ──────────────────────────────────────────────

class TestBenchmarkRunner:
    def test_list_benchmarks(self):
        from epsionic.tools.benchmarks import BenchmarkRunner
        benches = BenchmarkRunner.list_benchmarks()
        assert "mmlu" in benches
        assert "gsm8k" in benches
        assert "humaneval" in benches

    def test_run_missing_model(self):
        from epsionic.tools.benchmarks import BenchmarkRunner
        br = BenchmarkRunner()
        with pytest.raises(ValueError, match="model_path"):
            br.run("mmlu")

    def test_run_invalid_benchmark(self):
        from epsionic.tools.benchmarks import BenchmarkRunner
        br = BenchmarkRunner(model_path="/tmp/test")
        with pytest.raises(ValueError, match="Unknown benchmark"):
            br.run("nonexistent")

    def test_summary_empty(self):
        from epsionic.tools.benchmarks import BenchmarkRunner
        br = BenchmarkRunner()
        summary = br.summary()
        assert "Benchmark Results" in summary


# ── Prompt Manager ────────────────────────────────────────────────

class TestPromptManager:
    @pytest.fixture
    def pm(self):
        from epsionic.tools.prompt_manager import PromptManager
        tmp = Path(tempfile.mkdtemp()) / "prompts.db"
        pm = PromptManager(tmp)
        yield pm
        pm.close()

    def test_create_template(self, pm):
        result = pm.create_template("test-prompt", "Hello {name}!", variables=["name"])
        assert result["name"] == "test-prompt"
        assert result["version"] == 1

    def test_get_template(self, pm):
        created = pm.create_template("test", "template text")
        tmpl = pm.get_template(created["id"])
        assert tmpl is not None
        assert tmpl["name"] == "test"
        assert tmpl["template"] == "template text"

    def test_list_templates(self, pm):
        pm.create_template("a", "text a", tags="math")
        pm.create_template("b", "text b")
        templates = pm.list_templates()
        assert len(templates) >= 2

    def test_list_by_tag(self, pm):
        pm.create_template("c", "text c", tags="code")
        math_templates = pm.list_templates(tag="math")
        code_templates = pm.list_templates(tag="code")
        assert any(t["name"] == "c" for t in code_templates)

    def test_update_template(self, pm):
        created = pm.create_template("updatable", "v1")
        updated = pm.update_template(created["id"], "v2", changelog="Updated to v2")
        assert updated["version"] == 2
        assert updated["template"] == "v2"

    def test_version_history(self, pm):
        created = pm.create_template("historic", "v1")
        pm.update_template(created["id"], "v2")
        pm.update_template(created["id"], "v3")
        history = pm.get_version_history(created["id"])
        assert len(history) == 3
        assert history[0]["version"] == 3  # newest first

    def test_render(self, pm):
        created = pm.create_template("render-test", "Hello {name}, you are {age} years old!",
                                     variables=["name", "age"])
        result = pm.render(created["id"], {"name": "Alice", "age": "30"})
        assert result == "Hello Alice, you are 30 years old!"

    def test_render_missing_template(self, pm):
        assert pm.render("nonexistent") is None

    def test_create_ab_test(self, pm):
        a = pm.create_template("variant_a", "Prompt A")
        b = pm.create_template("variant_b", "Prompt B")
        ab = pm.create_ab_test("Test A vs B", a["id"], b["id"])
        assert ab["name"] == "Test A vs B"

    def test_record_ab_result(self, pm):
        a = pm.create_template("va", "pa")
        b = pm.create_template("vb", "pb")
        ab = pm.create_ab_test("test", a["id"], b["id"])
        pm.record_ab_result(ab["id"], "a", "pa", "input1", "output1", latency_ms=100, token_count=50, scored_value=0.8)
        pm.record_ab_result(ab["id"], "b", "pb", "input1", "output2", latency_ms=200, token_count=100, scored_value=0.6)
        summary = pm.get_ab_summary(ab["id"])
        assert summary["total_results"] == 2
        assert "a" in summary["variants"]
        assert "b" in summary["variants"]


# ── Webhook Manager ───────────────────────────────────────────────

class TestWebhookManager:
    @pytest.fixture
    def wh(self):
        from epsionic.core.webhooks import WebhookManager
        return WebhookManager()

    def test_register(self, wh):
        hook = wh.register("https://httpbin.org/post", ["train_complete", "error"], secret="mysecret")
        assert hook["url"] == "https://httpbin.org/post"
        assert "train_complete" in hook["events"]
        assert hook["enabled"] is True

    def test_list(self, wh):
        wh.register("https://example.com/hook1", ["event_a"])
        wh.register("https://example.com/hook2", ["event_b"])
        hooks = wh.list()
        assert len(hooks) == 2

    def test_remove(self, wh):
        hook = wh.register("https://example.com/hook", ["event"])
        assert wh.remove(hook["id"]) is True
        assert wh.remove("nonexistent") is False

    def test_remove_reduces_list(self, wh):
        wh.register("https://a.com", ["e1"])
        h2 = wh.register("https://b.com", ["e2"])
        wh.remove(h2["id"])
        assert len(wh.list()) == 1


# ── Workspace Manager ─────────────────────────────────────────────

class TestWorkspaceManager:
    @pytest.fixture
    def wm(self):
        from epsionic.core.workspace import WorkspaceManager
        tmp = Path(tempfile.mkdtemp()) / "tenants.db"
        wm = WorkspaceManager(tmp)
        yield wm
        wm.close()

    def test_create_tenant(self, wm):
        tenant = wm.create_tenant("test-user", tempfile.mkdtemp())
        assert tenant["tenant_id"].startswith("tenant_")
        assert tenant["api_key"].startswith("epsk_")

    def test_get_tenant_by_key(self, wm):
        tenant = wm.create_tenant("key-user", tempfile.mkdtemp())
        found = wm.get_tenant_by_key(tenant["api_key"])
        assert found is not None
        assert found["name"] == "key-user"

    def test_get_tenant_invalid_key(self, wm):
        assert wm.get_tenant_by_key("invalid-key") is None

    def test_generate_api_key(self, wm):
        tenant = wm.create_tenant("multi-key", tempfile.mkdtemp())
        new_key = wm.generate_api_key(tenant["tenant_id"], "backup")
        assert new_key.startswith("epsk_")
        assert new_key != tenant["api_key"]

    def test_revoke_api_key(self, wm):
        tenant = wm.create_tenant("revoke-test", tempfile.mkdtemp())
        assert wm.revoke_api_key(tenant["api_key"]) is True
        assert wm.revoke_api_key("nonexistent") is False

    def test_list_tenants(self, wm):
        wm.create_tenant("user1", tempfile.mkdtemp())
        wm.create_tenant("user2", tempfile.mkdtemp())
        tenants = wm.list_tenants()
        assert len(tenants) >= 2

    def test_audit_log(self, wm):
        tenant = wm.create_tenant("audit-user", tempfile.mkdtemp())
        wm.log_audit(tenant["tenant_id"], "train_started", {"model": "test"})
        wm.log_audit(tenant["tenant_id"], "train_completed", {"accuracy": 0.95})
        logs = wm.get_audit_log(tenant["tenant_id"])
        assert len(logs) >= 2
        assert logs[0]["action"] == "train_completed"  # newest first
