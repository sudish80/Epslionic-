"""Marketplace — Recipe and Plugin registry for downloading community content."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("epsionic.marketplace")

_REGISTRY_API = "https://api.github.com/repos/sudish80/Epslionic-/contents"
_RAW_BASE = "https://raw.githubusercontent.com/sudish80/Epslionic-/main"


class Marketplace:
    """Community registry for downloading recipes and plugins."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.recipes_dir = self.workspace / "recipes"
        self.plugins_dir = self.workspace / "plugins"
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

    def list_available(self, category: str = "recipes") -> List[dict]:
        """List items available in the registry."""
        import urllib.request
        path = "recipes" if category == "recipes" else "plugins"
        url = f"{_REGISTRY_API}/{path}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                items = json.loads(resp.read().decode())
            results = []
            for item in items:
                name = item["name"]
                if category == "recipes" and name.endswith(".yaml") or \
                   category == "plugins" and name.endswith(".py"):
                    results.append({
                        "name": name,
                        "type": "file" if item["type"] == "file" else "dir",
                        "url": item["download_url"] if item["type"] == "file" else item["url"],
                        "size": item.get("size", 0),
                    })
            return results
        except Exception as e:
            logger.warning("Registry list failed: %s", e)
            return []

    def download(self, category: str, name: str) -> bool:
        """Download an item from the registry."""
        import urllib.request
        dest_dir = self.recipes_dir if category == "recipes" else self.plugins_dir
        url = f"{_RAW_BASE}/{category}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                content = resp.read()
            dest = dest_dir / name
            dest.write_bytes(content)
            logger.info("Downloaded %s/%s to %s", category, name, dest)
            return True
        except Exception as e:
            logger.warning("Download failed: %s", e)
            return False

    def search(self, query: str, category: str = "recipes") -> List[dict]:
        """Search registry items by name."""
        items = self.list_available(category)
        query_lower = query.lower()
        return [i for i in items if query_lower in i["name"].lower()]

    def get_tool_description(self) -> dict:
        return {
            "name": "marketplace",
            "description": "Browse and download community recipes and plugins from the registry",
            "parameters": {
                "action": {"type": "string", "enum": ["list", "search", "download"]},
                "category": {"type": "string", "enum": ["recipes", "plugins"]},
                "query": {"type": "string", "optional": True},
                "name": {"type": "string", "optional": True},
            },
        }
