import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger("epsionic.plugin")


class Plugin:
    """A single loaded plugin with lifecycle hooks.
    Pattern from AutoGPT (184k stars): plugin architecture."""

    def __init__(self, name: str, module, config: dict = None):
        self.name = name
        self._module = module
        self.config = config or {}
        self.enabled = True
        self._hooks = {}

        self._discover_hooks()

    def _discover_hooks(self):
        hook_names = [
            "on_register", "on_unregister", "before_tool", "after_tool",
            "on_error", "on_startup", "on_shutdown",
        ]
        for hook in hook_names:
            fn = getattr(self._module, hook, None)
            if fn and callable(fn):
                self._hooks[hook] = fn

    def has_hook(self, name: str) -> bool:
        return name in self._hooks

    def run_hook(self, name: str, **kwargs) -> Optional[Any]:
        if name in self._hooks:
            try:
                return self._hooks[name](**kwargs)
            except Exception as e:
                logger.error(f"Plugin {self.name} hook {name}: {e}")
        return None

    def __repr__(self):
        return f"Plugin({self.name}, enabled={self.enabled}, hooks={list(self._hooks.keys())})"


class PluginManager:
    """Discovers and manages plugins from a directory.
    Pattern from AutoGPT (184k stars)."""

    def __init__(self, plugins_dir: Path = None):
        self._plugins: Dict[str, Plugin] = {}
        self._plugins_dir = Path(plugins_dir) if plugins_dir else Path("plugins")
        self._plugins_dir.mkdir(parents=True, exist_ok=True)

    @property
    def plugins(self) -> Dict[str, Plugin]:
        return dict(self._plugins)

    @property
    def enabled_plugins(self) -> Dict[str, Plugin]:
        return {n: p for n, p in self._plugins.items() if p.enabled}

    def discover(self) -> List[str]:
        """Scan plugins directory and load all plugins."""
        found = []
        for path in self._plugins_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            name = path.stem
            if name not in self._plugins:
                self._load_plugin(path, name)
                found.append(name)
            else:
                logger.debug(f"Plugin {name} already loaded")
        return found

    def _load_plugin(self, path: Path, name: str):
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                config = getattr(module, "PLUGIN_CONFIG", {})
                plugin = Plugin(name, module, config)
                self._plugins[name] = plugin
                plugin.run_hook("on_register")
                logger.info(f"Loaded plugin: {name}")
        except Exception as e:
            logger.error(f"Failed to load plugin {path.name}: {e}")

    def enable(self, name: str):
        if name in self._plugins:
            self._plugins[name].enabled = True
            logger.info(f"Plugin enabled: {name}")

    def disable(self, name: str):
        if name in self._plugins:
            self._plugins[name].enabled = False
            logger.info(f"Plugin disabled: {name}")

    def run_hook(self, hook: str, **kwargs) -> List[Any]:
        results = []
        for name, plugin in self.enabled_plugins.items():
            if plugin.has_hook(hook):
                result = plugin.run_hook(hook, **kwargs)
                if result is not None:
                    results.append((name, result))
        return results

    def summary(self) -> str:
        lines = ["# Plugin Status", ""]
        if not self._plugins:
            lines.append("No plugins loaded.")
            return "\n".join(lines)
        for name, plugin in self._plugins.items():
            status = "enabled" if plugin.enabled else "disabled"
            hooks = ", ".join(plugin._hooks.keys())
            lines.append(f"## {name}")
            lines.append(f"- Status: {status}")
            lines.append(f"- Hooks: {hooks}")
            lines.append("")
        return "\n".join(lines)

    def create_example(self):
        example = """\"\"\"
Example Epslionic Plugin
AutoGPT-style plugin (184k stars).
Hooks: on_register, before_tool, after_tool, on_error, on_startup, on_shutdown
\"\"\"

import logging
logger = logging.getLogger("epsionic.plugin.example")

PLUGIN_CONFIG = {
    "name": "Example Plugin",
    "version": "1.0.0",
}


def on_register():
    logger.info("Example plugin registered")


def before_tool(tool: str, **params) -> None:
    logger.debug(f"Example: before tool {tool}")


def after_tool(tool: str, result=None) -> None:
    logger.debug(f"Example: after tool {tool}")


def on_error(tool: str, error: Exception, **kw) -> None:
    logger.warning(f"Example: error in {tool}: {error}")
"""
        path = self._plugins_dir / "example.py"
        if not path.exists():
            path.write_text(example.lstrip())
            logger.info(f"Created example plugin: {path}")
