"""
Memory Store - Epslionic-inspired file-based memory.
Stores experiment data, training logs, errors, and agent state as JSON files.
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict

logger = logging.getLogger("epsionic.memory")


class MemoryStore:
    def __init__(self, memory_dir: Path, cache_size: int = 100, heartbeat_ttl_days: int = 7):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self._experiments_dir = self.memory_dir / "experiments"
        self._errors_dir = self.memory_dir / "errors"
        self._datasets_dir = self.memory_dir / "datasets"
        self._agent_dir = self.memory_dir / "agent"
        self._heartbeat_dir = self.memory_dir / "heartbeat"
        self._audit_dir = self.memory_dir / "audit"

        for d in [self._experiments_dir, self._errors_dir, self._datasets_dir,
                  self._agent_dir, self._heartbeat_dir, self._audit_dir]:
            d.mkdir(exist_ok=True)

        self._experiment_cache = OrderedDict()
        self._cache_size = cache_size
        self._heartbeat_ttl_days = heartbeat_ttl_days

    @property
    def exp_dir(self):
        return self._experiments_dir

    @property
    def err_dir(self):
        return self._errors_dir

    @property
    def heart_dir(self):
        return self._heartbeat_dir

    @property
    def ds_dir(self):
        return self._datasets_dir

    @property
    def base(self):
        return self.memory_dir

    # --- Agent State (Epslionic SOUL.md / AGENTS.md pattern) ---
    def write_agent_state(self, key: str, data: Any):
        path = self._agent_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def read_agent_state(self, key: str) -> Optional[Any]:
        path = self._agent_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    # --- Experiments ---
    def create_experiment(self, name: str, config: dict, tags: list = None) -> str:
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = {
            "id": exp_id,
            "name": name,
            "config": config,
            "tags": tags or [],
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metrics": {},
            "artifacts": [],
            "errors": [],
        }
        self._write_experiment(exp_id, record)
        return exp_id

    def update_experiment(self, exp_id: str, **updates):
        record = self._read_experiment(exp_id)
        if record is None:
            return
        for k, v in updates.items():
            if k == "metrics" and isinstance(v, dict):
                record["metrics"].update(v)
            elif k == "errors" and isinstance(v, list):
                record["errors"].extend(v)
            elif k == "artifacts" and isinstance(v, list):
                record["artifacts"].extend(v)
            else:
                record[k] = v
        record["updated_at"] = datetime.now().isoformat()
        self._write_experiment(exp_id, record)

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        return self._read_experiment(exp_id)

    def export_experiments_csv(self, path: str = None) -> str:
        import csv, io
        exps = self.list_experiments()
        if not exps:
            return "No experiments to export"
        output_path = path or str(self.memory_dir / "experiments_export.csv")
        with open(output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "name", "status", "created_at", "updated_at", "config"])
            w.writeheader()
            for e in exps:
                row = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in e.items() if k in ["id", "name", "status", "created_at", "updated_at", "config"]}
                w.writerow(row)
        return output_path

    def list_experiments(self, status: Optional[str] = None, domain: Optional[str] = None, tags: Optional[list] = None) -> List[dict]:
        cache_key = f"experiments_{status or 'all'}_{domain or 'any'}_{str(tags or 'any')}"
        if cache_key in self._experiment_cache:
            return self._experiment_cache[cache_key]

        experiments = []
        for f in self._experiments_dir.glob("*.json"):
            try:
                exp = json.loads(f.read_text())
                if status is not None and exp.get("status") != status:
                    continue
                if domain is not None:
                    cfg = exp.get("config", {})
                    exp_domain = cfg.get("domain") if isinstance(cfg, dict) else None
                    if exp_domain != domain:
                        continue
                if tags is not None:
                    exp_tags = exp.get("tags", [])
                    if not all(t in exp_tags for t in tags):
                        continue
                experiments.append(exp)
            except Exception as e:
                logger.warning(f"Skipping corrupt experiment file {f.name}: {e}")
        result = sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)

        self._experiment_cache[cache_key] = result
        while len(self._experiment_cache) > self._cache_size:
            self._experiment_cache.popitem(last=False)
        return result

    def _write_experiment(self, exp_id: str, data: dict):
        path = self._experiments_dir / f"{exp_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        self._experiment_cache.clear()

    def _read_experiment(self, exp_id: str) -> Optional[dict]:
        path = self._experiments_dir / f"{exp_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    # --- Errors ---
    def log_error(self, source: str, error: str, context: dict = None):
        record = {
            "id": f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "source": source,
            "error": error,
            "context": context or {},
            "fixed": False,
            "fix_attempted": False,
            "timestamp": datetime.now().isoformat(),
        }
        path = self._errors_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        return record["id"]

    def mark_error_fixed(self, error_id: str, fix_description: str = ""):
        for f in self._errors_dir.glob("*.json"):
            rec = json.loads(f.read_text())
            if rec["id"] == error_id:
                rec["fixed"] = True
                rec["fix_attempted"] = True
                rec["fix_description"] = fix_description
                rec["fixed_at"] = datetime.now().isoformat()
                f.write_text(json.dumps(rec, indent=2, default=str))
                return

    def get_unfixed(self) -> List[dict]:
        return self.get_unfixed_errors()

    def get_unfixed_errors(self) -> List[dict]:
        errors = []
        for f in self._errors_dir.glob("*.json"):
            rec = json.loads(f.read_text())
            if not rec.get("fixed"):
                errors.append(rec)
        return sorted(errors, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_error_summary(self) -> str:
        errors = self.get_unfixed_errors()
        if not errors:
            return "No unresolved errors."
        lines = ["### Unresolved Errors", ""]
        for e in errors[:10]:
            lines.append(f"- **{e['source']}** ({e['id']}): {e['error'][:200]}")
        return "\n".join(lines)

    # --- Datasets ---
    def record_dataset(self, dataset_id: str, metadata: dict):
        path = self._datasets_dir / f"{dataset_id.replace('/', '_')}.json"
        metadata["recorded_at"] = datetime.now().isoformat()
        path.write_text(json.dumps(metadata, indent=2, default=str))

    def get_known_datasets(self) -> List[dict]:
        datasets = []
        for f in self._datasets_dir.glob("*.json"):
            datasets.append(json.loads(f.read_text()))
        return datasets

    # --- Heartbeat ---
    def write_heartbeat_log(self, status: str, details: str = ""):
        self._cleanup_old_heartbeats()
        record = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details,
        }
        path = self._heartbeat_dir / f"heartbeat_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _cleanup_old_heartbeats(self):
        cutoff = datetime.now() - timedelta(days=self._heartbeat_ttl_days)
        for f in self._heartbeat_dir.glob("*.jsonl"):
            try:
                date_str = f.stem.replace("heartbeat_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date < cutoff:
                    f.unlink()
                    logger.debug(f"Removed old heartbeat log: {f.name}")
            except (ValueError, OSError):
                pass

    def get_recent_heartbeats(self, n: int = 10) -> List[dict]:
        beats = []
        for f in sorted(self._heartbeat_dir.glob("*.jsonl"), reverse=True):
            with open(f) as fh:
                for line in fh:
                    beats.append(json.loads(line.strip()))
            if len(beats) >= n:
                break
        return beats[-n:]

    def write_audit_log(self, action: str, details: dict = None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {},
        }
        path = self._audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def create_job(self, job_type: str, params: dict, priority: int = 0) -> str:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        record = {
            "id": job_id, "type": job_type, "params": params,
            "priority": priority, "status": "queued", "progress": 0.0,
            "result": None, "error": None,
            "created_at": datetime.now().isoformat(),
            "started_at": None, "completed_at": None,
        }
        jobs_dir = self.memory_dir / "jobs"
        jobs_dir.mkdir(exist_ok=True)
        path = jobs_dir / f"{job_id}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        self.write_audit_log("job_created", {"job_id": job_id, "type": job_type})
        return job_id

    def update_job(self, job_id: str, **updates):
        jobs_dir = self.memory_dir / "jobs"
        path = jobs_dir / f"{job_id}.json"
        if not path.exists():
            return
        record = json.loads(path.read_text())
        for k, v in updates.items():
            record[k] = v
        path.write_text(json.dumps(record, indent=2, default=str))

    def get_job(self, job_id: str) -> Optional[dict]:
        jobs_dir = self.memory_dir / "jobs"
        path = jobs_dir / f"{job_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_jobs(self, status: str = None, job_type: str = None) -> list:
        jobs_dir = self.memory_dir / "jobs"
        if not jobs_dir.exists():
            return []
        jobs = []
        for f in jobs_dir.glob("*.json"):
            try:
                job = json.loads(f.read_text())
                if status and job.get("status") != status:
                    continue
                if job_type and job.get("type") != job_type:
                    continue
                jobs.append(job)
            except Exception:
                pass
        return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job or job["status"] in ("completed", "cancelled", "failed"):
            return False
        self.update_job(job_id, status="cancelled", completed_at=datetime.now().isoformat())
        self.write_audit_log("job_cancelled", {"job_id": job_id})
        return True

    def get_training_summary_markdown(self) -> str:
        exps = self.list_experiments()
        if not exps:
            return "No experiments recorded yet."
        lines = ["# Training Summary", ""]
        for exp in exps[:5]:
            status = exp.get("status", "unknown")
            metrics = exp.get("metrics", {})
            lines.append(f"## {exp['name']} ({exp['id']})")
            lines.append(f"- **Status**: {status}")
            lines.append(f"- **Created**: {exp.get('created_at', 'unknown')}")
            if metrics:
                lines.append(f"- **Metrics**: {json.dumps(metrics, indent=2)}")
            lines.append("")
        return "\n".join(lines)
