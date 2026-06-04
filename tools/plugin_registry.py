"""Plugin Registry / Marketplace — discover, install, enable/disable plugins.
Supports loading plugins from local paths, GitHub repos, and PyPI packages."""

import json
import logging
import os
import subprocess
import sys
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

logger = logging.getLogger("epsionic.tool.plugin_registry")


class PluginRegistry:
    """Central registry for discovering, installing, and managing plugins."""

    def __init__(self, plugins_dir: Path = None, memory_store=None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path("plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory_store
        self._registry: Dict[str, dict] = {}
        self._load_registry()

    @property
    def registry_file(self) -> Path:
        return self.plugins_dir / "registry.json"

    def _load_registry(self):
        if self.registry_file.exists():
            try:
                self._registry = json.loads(self.registry_file.read_text())
            except Exception:
                self._registry = {}
        else:
            self._registry = {}
            self._save_registry()

    def _save_registry(self):
        self.registry_file.write_text(json.dumps(self._registry, indent=2, default=str))

    def discover_local(self) -> List[str]:
        """Discover plugins installed in the plugins directory."""
        found = []
        for f in self.plugins_dir.glob("*.py"):
            name = f.stem
            if name != "__init__" and name not in self._registry:
                self._registry[name] = {
                    "name": name, "source": "local", "path": str(f),
                    "enabled": True, "installed_at": datetime.now().isoformat(),
                    "version": "0.1.0", "description": f"Local plugin: {name}",
                }
                found.append(name)
        for d in self.plugins_dir.iterdir():
            if d.is_dir() and (d / "__init__.py").exists():
                name = d.name
                if name not in self._registry:
                    self._registry[name] = {
                        "name": name, "source": "local", "path": str(d),
                        "enabled": True, "installed_at": datetime.now().isoformat(),
                        "version": "0.1.0", "description": f"Local plugin package: {name}",
                    }
                    found.append(name)
        if found:
            self._save_registry()
        return found

    def install_from_pypi(self, package_name: str, plugin_name: str = None) -> dict:
        """Install a plugin from PyPI."""
        name = plugin_name or package_name.replace("-", "_").replace(".", "_")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {"success": False, "error": result.stderr[:500]}
            self._registry[name] = {
                "name": name, "source": "pypi", "package": package_name,
                "enabled": True, "installed_at": datetime.now().isoformat(),
                "version": "1.0.0", "description": f"PyPI plugin: {package_name}",
            }
            self._save_registry()
            return {"success": True, "plugin": name, "source": "pypi"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def install_from_github(self, repo_url: str, plugin_name: str = None) -> dict:
        """Install a plugin from a GitHub repository."""
        name = plugin_name or repo_url.split("/")[-1].replace(".git", "")
        target_dir = self.plugins_dir / name
        try:
            import subprocess
            if target_dir.exists():
                import shutil; shutil.rmtree(target_dir)
            subprocess.run(["git", "clone", repo_url, str(target_dir)],
                           capture_output=True, text=True, timeout=120, check=True)
            self._registry[name] = {
                "name": name, "source": "github", "repo": repo_url,
                "enabled": True, "installed_at": datetime.now().isoformat(),
                "version": "0.1.0", "description": f"GitHub plugin: {name}",
            }
            self._save_registry()
            return {"success": True, "plugin": name, "source": "github", "path": str(target_dir)}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Git clone timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def install_from_path(self, path: str, plugin_name: str = None) -> dict:
        """Install a plugin from a local file path."""
        src = Path(path)
        if not src.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        name = plugin_name or src.stem
        if src.is_file() and src.suffix == ".py":
            dst = self.plugins_dir / f"{name}.py"
            import shutil
            shutil.copy2(str(src), str(dst))
        elif src.is_dir():
            dst = self.plugins_dir / name
            import shutil
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
        else:
            return {"success": False, "error": f"Unsupported plugin type: {path}"}
        self._registry[name] = {
            "name": name, "source": "path", "path": str(path),
            "enabled": True, "installed_at": datetime.now().isoformat(),
            "version": "0.1.0", "description": f"Local plugin: {name}",
        }
        self._save_registry()
        return {"success": True, "plugin": name, "source": "path"}

    def list_plugins(self) -> List[dict]:
        return list(self._registry.values())

    def get_plugin(self, name: str) -> Optional[dict]:
        return self._registry.get(name)

    def enable(self, name: str) -> bool:
        if name in self._registry:
            self._registry[name]["enabled"] = True
            self._save_registry()
            return True
        return False

    def disable(self, name: str) -> bool:
        if name in self._registry:
            self._registry[name]["enabled"] = False
            self._save_registry()
            return True
        return False

    def uninstall(self, name: str) -> bool:
        if name not in self._registry:
            return False
        plugin = self._registry[name]
        # Remove files
        if plugin.get("source") == "local" or plugin.get("source") == "path":
            p = Path(plugin.get("path", ""))
            if p.exists():
                import shutil
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        elif plugin.get("source") == "pypi":
            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", plugin.get("package", name)],
                               capture_output=True, timeout=30)
            except Exception:
                pass
        del self._registry[name]
        self._save_registry()
        return True

    def search_registry(self, query: str = "") -> List[dict]:
        """Search locally registered plugins."""
        results = []
        q = query.lower()
        for name, info in self._registry.items():
            if not q or q in name.lower() or q in info.get("description", "").lower():
                results.append(info)
        return results

    def get_tool_description(self) -> dict:
        return {
            "install_from_pypi": {"description": "Install plugin from PyPI",
                "parameters": {"package_name": "PyPI package name", "plugin_name": "Optional plugin alias"}},
            "install_from_github": {"description": "Install plugin from GitHub",
                "parameters": {"repo_url": "GitHub repo URL (https://...)", "plugin_name": "Optional plugin name"}},
            "install_from_path": {"description": "Install plugin from local path",
                "parameters": {"path": "Local file or directory path", "plugin_name": "Optional plugin name"}},
            "list_plugins": {"description": "List all registered plugins"},
            "search_registry": {"description": "Search registered plugins",
                "parameters": {"query": "Search term"}},
            "enable": {"description": "Enable a plugin",
                "parameters": {"name": "Plugin name"}},
            "disable": {"description": "Disable a plugin",
                "parameters": {"name": "Plugin name"}},
            "uninstall": {"description": "Remove a plugin entirely",
                "parameters": {"name": "Plugin name"}},
        }
