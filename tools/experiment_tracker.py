"""Experiment Tracking — W&B and MLflow integration.
Logs hyperparameters, metrics, artifacts, and system info."""

import json
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("openclaw.tool.experiment_tracker")


class ExperimentTracker:
    """Track experiments with W&B or MLflow backends."""

    def __init__(self, backend: str = "none", config: dict = None):
        self.backend = backend
        self.config = config or {}
        self._run = None
        self._run_id = None

    def init_run(self, project: str, name: str = None, config: dict = None):
        if self.backend == "wandb":
            import wandb
            self._run = wandb.init(
                project=project, name=name or f"run_{name}",
                config=config or {}, reinit=True,
            )
            self._run_id = self._run.id if self._run else None
        elif self.backend == "mlflow":
            import mlflow
            mlflow.set_experiment(project)
            mlflow.start_run(run_name=name)
            if config:
                mlflow.log_params({k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in config.items()})
            self._run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        logger.info(f"Tracker initialized: {self.backend}/{project}/{name}")

    def log_metrics(self, metrics: dict, step: int = None):
        if self.backend == "wandb" and self._run:
            import wandb
            self._run.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=step or 0)

    def log_params(self, params: dict):
        if self.backend == "wandb" and self._run:
            import wandb
            wandb.config.update(params)
        elif self.backend == "mlflow":
            import mlflow
            for k, v in params.items():
                mlflow.log_param(k, str(v) if not isinstance(v, (int, float, bool)) else v)

    def log_artifact(self, path: str, name: str = None):
        if self.backend == "wandb" and self._run:
            import wandb
            artifact = wandb.Artifact(name or Path(path).name, type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_artifact(path)

    def log_system_metrics(self):
        if self.backend == "wandb" and self._run:
            import wandb
            self._run.log_code()
            wandb.watch(self._run)

    def finish(self):
        if self.backend == "wandb" and self._run:
            import wandb
            self._run.finish()
        elif self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
        self._run = None
        self._run_id = None

    def get_run_url(self) -> Optional[str]:
        if self.backend == "wandb" and self._run:
            return f"https://wandb.ai/{self._run.project}/{self._run.id}"
        return None

    def sweep(self, project: str, config: dict, objective_fn):
        """Run W&B sweep for hyperparameter optimization."""
        if self.backend != "wandb":
            logger.warning("Sweeps require W&B backend")
            return None
        import wandb
        sweep_id = wandb.sweep(config, project=project)
        wandb.agent(sweep_id, function=objective_fn)
        return sweep_id

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id
