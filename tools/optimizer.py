"""Hyperparameter Optimization — Optuna-based search with pruning.
Optimizes training hyperparameters per domain and hardware."""

import json
import logging
import os
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

logger = logging.getLogger("epsionic.tool.optimizer")


DOMAIN_SEARCH_SPACES = {
    "math": {"lr": (1e-5, 5e-4), "lora_r": (4, 32), "batch_size": (1, 4), "epochs": (2, 5)},
    "code": {"lr": (1e-5, 5e-4), "lora_r": (4, 16), "batch_size": (1, 4), "epochs": (2, 5)},
    "medical": {"lr": (5e-6, 2e-4), "lora_r": (4, 16), "batch_size": (1, 4), "epochs": (1, 3)},
    "general": {"lr": (1e-5, 5e-4), "lora_r": (8, 32), "batch_size": (1, 4), "epochs": (2, 5)},
}


class HyperparameterOptimizer:
    """Hyperparameter search using Optuna with pruning."""

    def __init__(self, memory_store=None, studies_dir: Path = None):
        self.memory = memory_store
        self.studies_dir = studies_dir or Path("/content/studies")
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        self._study = None

    def create_study(self, domain: str, n_trials: int = 10,
                     sampler: str = "tpe", pruner: str = "median") -> str:
        """Create and run an Optuna study for a domain."""
        import optuna
        sampler_map = {"tpe": optuna.samplers.TPESampler(), "random": optuna.samplers.RandomSampler(),
                       "grid": optuna.samplers.GridSampler}
        pruner_map = {"median": optuna.pruners.MedianPruner(), "hyperband": optuna.pruners.HyperbandPruner(),
                      "none": optuna.pruners.NopPruner()}

        study_name = f"{domain}_hpo_{len(list(self.studies_dir.glob('*.db')))}"
        storage = f"sqlite:///{self.studies_dir / study_name}.db"

        self._study = optuna.create_study(
            study_name=study_name, storage=storage,
            sampler=sampler_map.get(sampler, optuna.samplers.TPESampler()),
            pruner=pruner_map.get(pruner, optuna.pruners.MedianPruner()),
            direction="minimize", load_if_exists=True,
        )
        return study_name

    def optimize(self, objective_fn: Callable, n_trials: int = 10,
                 timeout: int = None, study_name: str = None) -> dict:
        """Run optimization with optional timeout."""
        if not self._study and study_name:
            self._study = optuna.load_study(study_name=study_name)
        if not self._study:
            return {"error": "No study created"}

        self._study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

        best = self._study.best_params if self._study.best_trials else {}
        result = {
            "best_params": best,
            "best_value": self._study.best_value if self._study.best_trials else None,
            "n_trials": len(self._study.trials),
            "n_complete": len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
        self._save_study(result)
        return result

    def get_search_space(self, domain: str, vram_gb: float = 16.0) -> dict:
        space = dict(DOMAIN_SEARCH_SPACES.get(domain, DOMAIN_SEARCH_SPACES["general"]))
        if vram_gb < 12:
            space["batch_size"] = (1, 2)
            space["lora_r"] = (4, 8)
        elif vram_gb > 20:
            space["batch_size"] = (2, 8)
        return space

    def suggest_params(self, trial, domain: str, vram_gb: float = 16.0) -> dict:
        space = self.get_search_space(domain, vram_gb)
        return {
            "learning_rate": trial.suggest_float("learning_rate", *space.get("lr", (1e-5, 5e-4)), log=True),
            "lora_r": trial.suggest_int("lora_r", *space.get("lora_r", (4, 32))),
            "batch_size": trial.suggest_int("batch_size", *space.get("batch_size", (1, 4))),
            "num_train_epochs": trial.suggest_int("num_train_epochs", *space.get("epochs", (1, 5))),
        }

    def _save_study(self, result: dict):
        path = self.studies_dir / f"study_{len(list(self.studies_dir.glob('study_*.json')))}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        if self.memory:
            self.memory.write_agent_state("last_hpo", result)

    def get_tool_description(self) -> dict:
        return {"optimize": {"description": "Run hyperparameter optimization with Optuna",
            "parameters": {"domain": "Domain to optimize for", "n_trials": "Number of trials",
                "sampler": "tpe|random|grid", "pruner": "median|hyperband|none"}}}
