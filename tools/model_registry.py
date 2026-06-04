"""Model Registry — versioned storage for trained models with metrics and lineage."""

import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("epsionic.tool.model_registry")


@dataclass
class ModelVersion:
    version: str
    model_path: str
    base_model: str = ""
    experiment_id: str = ""
    metrics: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    created_at: str = ""
    status: str = "created"  # created, pushed, archived


class ModelRegistry:
    """Versioned model registry with lineage tracking."""

    def __init__(self, memory_store=None, registry_dir: Path = None):
        self.memory = memory_store
        self.registry_dir = registry_dir or Path("/content/models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, model_path: str, name: str, base_model: str = "",
                 experiment_id: str = "", metrics: dict = None,
                 tags: list = None) -> dict:
        registry = self._load_registry(name)
        version = f"v{len(registry) + 1}"
        entry = ModelVersion(
            version=version, model_path=model_path, base_model=base_model,
            experiment_id=experiment_id, metrics=metrics or {},
            tags=tags or [], created_at=datetime.now().isoformat(),
            status="created",
        )
        registry.append(asdict(entry))
        self._save_registry(name, registry)
        logger.info(f"Registered {name}:{version} from {model_path}")
        if self.memory:
            self.memory.write_agent_state(f"model_{name}_{version}", asdict(entry))
        return asdict(entry)

    def list(self, name: str = None) -> List[dict]:
        if name:
            return self._load_registry(name)
        all_versions = []
        for f in self.registry_dir.glob("*.json"):
            all_versions.extend(json.loads(f.read_text()))
        return sorted(all_versions, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_latest(self, name: str) -> Optional[dict]:
        registry = self._load_registry(name)
        return registry[-1] if registry else None

    def get_version(self, name: str, version: str) -> Optional[dict]:
        for v in self._load_registry(name):
            if v["version"] == version:
                return v
        return None

    def tag(self, name: str, version: str, tags: list):
        registry = self._load_registry(name)
        for v in registry:
            if v["version"] == version:
                existing = set(v.get("tags", []))
                existing.update(tags)
                v["tags"] = sorted(existing)
                break
        self._save_registry(name, registry)

    def promote(self, name: str, version: str):
        registry = self._load_registry(name)
        for v in registry:
            if v["version"] == version:
                v["status"] = "promoted"
            elif v["status"] == "promoted":
                v["status"] = "archived"
        self._save_registry(name, registry)

    def export_leaderboard(self, metric: str = "eval_loss", top_k: int = 10) -> list:
        entries = []
        for f in self.registry_dir.glob("*.json"):
            registry = json.loads(f.read_text())
            for v in registry:
                m = v.get("metrics", {})
                val = m.get(metric)
                if val is not None:
                    entries.append({
                        "name": f.stem, "version": v["version"],
                        metric: val, "status": v.get("status", ""),
                        "created_at": v.get("created_at", ""),
                    })
        return sorted(entries, key=lambda x: x.get(metric, float("inf")))[:top_k]

    def _load_registry(self, name: str) -> list:
        path = self.registry_dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return []

    def _save_registry(self, name: str, data: list):
        path = self.registry_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
