"""Cost Tracking — GPU compute budget and API cost monitoring."""

import json
import logging
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger("epsionic.tool.cost_tracker")


@dataclass
class CostEntry:
    timestamp: float
    category: str  # gpu, api, storage, total
    amount: float
    currency: str = "USD"
    description: str = ""
    provider: str = ""


GPU_HOURLY_RATES = {
    "T4": 0.35, "V100": 1.06, "A100": 1.50, "A10G": 0.70,
    "L4": 0.60, "H100": 3.50, "RTX4090": 0.50,
}


class CostTracker:
    """Track GPU compute costs, API usage, and storage."""

    def __init__(self, memory_store=None, budget_limit: float = 50.0,
                 cost_file: Path = None):
        self.memory = memory_store
        self.budget_limit = budget_limit
        self.cost_file = cost_file or Path("/content/costs.jsonl")
        self._session_start = time.time()
        self._gpu_type = self._detect_gpu()

    def _detect_gpu(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).upper()
                for gpu in ["H100", "A100", "A10G", "V100", "T4", "L4", "RTX4090"]:
                    if gpu in name:
                        return gpu
                return "A100"
            return "cpu"
        except Exception:
            return "cpu"

    def track_training(self, duration_hours: float, gpu_type: str = None) -> CostEntry:
        gpu = gpu_type or self._gpu_type
        rate = GPU_HOURLY_RATES.get(gpu, 0.50)
        cost = duration_hours * rate

        entry = CostEntry(
            timestamp=time.time(), category="gpu",
            amount=round(cost, 4), provider=gpu,
            description=f"Training ({gpu}) x {duration_hours:.2f}h @ ${rate}/h",
        )
        self._log_entry(entry)
        return entry

    def track_api_call(self, provider: str, tokens: int = 0,
                       cost_per_1k: float = 0.002) -> CostEntry:
        cost = (tokens / 1000) * cost_per_1k
        entry = CostEntry(
            timestamp=time.time(), category="api",
            amount=round(cost, 6), provider=provider,
            description=f"API {provider} ({tokens} tokens)",
        )
        self._log_entry(entry)
        return entry

    def track_storage(self, size_gb: float, duration_days: float = 1.0) -> CostEntry:
        rate = 0.023  # ~$0.023/GB-month for cloud storage
        cost = size_gb * duration_days * (rate / 30)
        entry = CostEntry(
            timestamp=time.time(), category="storage",
            amount=round(cost, 6), provider="cloud",
            description=f"Storage {size_gb:.2f}GB x {duration_days:.1f}d",
        )
        self._log_entry(entry)
        return entry

    def get_total_cost(self, since: float = 0) -> float:
        entries = self._load_entries()
        recent = [e for e in entries if e.timestamp >= since]
        return round(sum(e.amount for e in recent), 4)

    def get_session_cost(self) -> float:
        return self.get_total_cost(since=self._session_start)

    def get_budget_remaining(self) -> float:
        return round(self.budget_limit - self.get_session_cost(), 4)

    def is_over_budget(self) -> bool:
        return self.get_session_cost() >= self.budget_limit

    def estimate_training_cost(self, gpu_type: str = None,
                                hours: float = 1.0, gpus: int = 1) -> dict:
        gpu = gpu_type or self._gpu_type
        rate = GPU_HOURLY_RATES.get(gpu, 0.50)
        hourly = rate * gpus
        return {
            "gpu": gpu, "rate_per_hour": hourly, "estimated_hours": hours,
            "estimated_cost": round(hourly * hours, 2), "currency": "USD",
        }

    def summary(self) -> dict:
        total = self.get_total_cost()
        session = self.get_session_cost()
        entries = self._load_entries()
        by_category = {}
        for e in entries:
            by_category[e.category] = by_category.get(e.category, 0) + e.amount
        return {
            "total_cost": total, "session_cost": session,
            "budget_limit": self.budget_limit, "budget_remaining": self.get_budget_remaining(),
            "over_budget": self.is_over_budget(),
            "gpu_type": self._gpu_type, "by_category": {k: round(v, 4) for k, v in by_category.items()},
            "entries_count": len(entries),
        }

    def _log_entry(self, entry: CostEntry):
        with open(self.cost_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def _load_entries(self) -> List[CostEntry]:
        if not self.cost_file.exists():
            return []
        entries = []
        try:
            with open(self.cost_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entries.append(CostEntry(**data))
        except (json.JSONDecodeError, IOError):
            pass
        return entries

    def reset_session(self):
        self._session_start = time.time()

    def get_tool_description(self) -> dict:
        return {"estimate_training_cost": {"description": "Estimate GPU training cost",
            "parameters": {"gpu_type": "GPU model (T4, V100, A100)", "hours": "Training hours",
                "gpus": "Number of GPUs"}}}
