"""RecipeManager — load, list, search, and apply training recipes."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("epsionic.recipes")

_RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"
_REGISTRY_URL = "https://raw.githubusercontent.com/sudish80/Epslionic-/main/recipes/"


class RecipeManager:
    """Manage training recipes — pre-configured model+dataset+hyperparameter combos."""

    def __init__(self, recipes_dir: str | Path = None):
        self.recipes_dir = Path(recipes_dir or _RECIPES_DIR)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, dict] = {}

    def list_recipes(self, domain: str = None) -> List[dict]:
        """List all available recipes, optionally filtered by domain."""
        results = []
        for f in sorted(self.recipes_dir.glob("*.yaml")):
            try:
                import yaml
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if not data or not isinstance(data, dict):
                    continue
                if domain and data.get("domain") != domain:
                    continue
                data["_file"] = f.name
                results.append(data)
            except Exception as e:
                logger.warning("Failed to load recipe %s: %s", f.name, e)
        return results

    def get_recipe(self, name: str) -> Optional[dict]:
        """Get a single recipe by name (case-insensitive substring match)."""
        name_lower = name.lower()
        for f in self.recipes_dir.glob("*.yaml"):
            try:
                import yaml
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if data and name_lower in data.get("name", "").lower():
                    data["_file"] = f.name
                    return data
            except Exception:
                continue
        return None

    def apply_recipe(self, recipe_name: str, domain_config: dict = None) -> dict:
        """Merge a recipe into a domain config for training."""
        recipe = self.get_recipe(recipe_name)
        if not recipe:
            raise ValueError(f"Recipe not found: {recipe_name}")
        config = dict(domain_config or {})
        config["model"] = recipe.get("model", config.get("model", ""))
        config["dataset"] = recipe.get("dataset", config.get("dataset", ""))
        config["format"] = recipe.get("format", config.get("format", ""))
        tc = config.get("training", {})
        rc = recipe.get("training", {})
        merged = dict(tc)
        merged.update(rc)
        config["training"] = merged
        config["recipe_name"] = recipe.get("name", recipe_name)
        return config

    def search_registry(self, query: str = "") -> List[dict]:
        """Fetch available recipes from the GitHub registry."""
        import urllib.request
        import json
        api_url = "https://api.github.com/repos/sudish80/Epslionic-/contents/recipes"
        try:
            req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                items = json.loads(resp.read().decode())
            results = []
            for item in items:
                if item["name"].endswith(".yaml"):
                    dl_url = item["download_url"]
                    with urllib.request.urlopen(dl_url, timeout=10) as f:
                        import yaml
                        data = yaml.safe_load(f.read().decode())
                    if data and (not query or query.lower() in data.get("name", "").lower()):
                        data["_url"] = dl_url
                        results.append(data)
            return results
        except Exception as e:
            logger.warning("Registry search failed: %s", e)
            return []

    def download_recipe(self, recipe_name: str) -> bool:
        """Download a recipe from the registry to local recipes dir."""
        import urllib.request
        import yaml
        url = _REGISTRY_URL + recipe_name
        if not recipe_name.endswith(".yaml"):
            url += ".yaml"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                content = resp.read().decode()
            data = yaml.safe_load(content)
            if not data or not isinstance(data, dict):
                return False
            dest = self.recipes_dir / (recipe_name if recipe_name.endswith(".yaml") else recipe_name + ".yaml")
            dest.write_text(content, encoding="utf-8")
            logger.info("Downloaded recipe: %s", dest.name)
            return True
        except Exception as e:
            logger.warning("Download failed: %s", e)
            return False

    def get_tool_description(self) -> dict:
        return {
            "name": "recipe_manager",
            "description": "Manage and apply training recipes — pre-configured model+dataset combos",
            "parameters": {
                "action": {"type": "string", "enum": ["list", "get", "apply", "search_registry", "download"]},
                "recipe_name": {"type": "string", "optional": True},
                "domain": {"type": "string", "optional": True},
            },
        }
