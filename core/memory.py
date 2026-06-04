"""
Memory Store - Epslionic-inspired file-based memory with SQLite persistence.
Stores experiment data, training logs, errors, and agent state as JSON files + SQLite backup.
"""

import json
import os
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict

logger = logging.getLogger("epsionic.memory")


class SQLiteStore:
    """SQLite persistence layer for agent state, experiments, and jobs."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, name TEXT, config TEXT, tags TEXT,
                status TEXT, created_at TEXT, updated_at TEXT,
                metrics TEXT, artifacts TEXT, errors TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, type TEXT, params TEXT, priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'queued', progress REAL DEFAULT 0.0,
                result TEXT, error TEXT, created_at TEXT,
                started_at TEXT, completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, action TEXT, details TEXT
            );
            CREATE TABLE IF NOT EXISTS flows (
                id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
                nodes TEXT DEFAULT '[]', triggers TEXT DEFAULT '[]',
                version TEXT DEFAULT '1.0', created_at TEXT, updated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        """)
        self._conn.commit()

    # Experiments
    def save_experiment(self, exp: dict):
        c = self._conn.cursor()
        c.execute("""INSERT OR REPLACE INTO experiments
            (id, name, config, tags, status, created_at, updated_at, metrics, artifacts, errors)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (exp.get("id"), exp.get("name"), json.dumps(exp.get("config", {})),
             json.dumps(exp.get("tags", [])), exp.get("status"),
             exp.get("created_at"), exp.get("updated_at"),
             json.dumps(exp.get("metrics", {})),
             json.dumps(exp.get("artifacts", [])),
             json.dumps(exp.get("errors", []))))
        self._conn.commit()

    def list_experiments_sqlite(self, status=None, limit=100) -> list:
        c = self._conn.cursor()
        if status:
            c.execute("SELECT * FROM experiments WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            c.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ["config", "tags", "metrics", "artifacts", "errors"]:
                if isinstance(d.get(field), str):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    # Agent state
    def save_agent_state(self, key: str, value: Any):
        c = self._conn.cursor()
        c.execute("INSERT OR REPLACE INTO agent_state (key, value, updated_at) VALUES (?,?,?)",
                  (key, json.dumps(value, default=str), datetime.now().isoformat()))
        self._conn.commit()

    def read_agent_state(self, key: str) -> Optional[Any]:
        c = self._conn.cursor()
        c.execute("SELECT value FROM agent_state WHERE key=?", (key,))
        row = c.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return row["value"]
        return None

    # Jobs
    def save_job(self, job: dict):
        c = self._conn.cursor()
        c.execute("""INSERT OR REPLACE INTO jobs
            (id, type, params, priority, status, progress, result, error, created_at, started_at, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (job.get("id"), job.get("type"), json.dumps(job.get("params", {})),
             job.get("priority", 0), job.get("status"), job.get("progress", 0.0),
             json.dumps(job.get("result")) if job.get("result") else None,
             job.get("error"), job.get("created_at"),
             job.get("started_at"), job.get("completed_at")))
        self._conn.commit()

    def list_jobs_sqlite(self, status=None, job_type=None, limit=100) -> list:
        c = self._conn.cursor()
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if status:
            query += " AND status=?"
            params.append(status)
        if job_type:
            query += " AND type=?"
            params.append(job_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        c.execute(query, params)
        rows = c.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ["params", "result"]:
                if isinstance(d.get(field), str):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    # Audit
    def save_audit(self, action: str, details: dict = None):
        c = self._conn.cursor()
        c.execute("INSERT INTO audit_log (timestamp, action, details) VALUES (?,?,?)",
                  (datetime.now().isoformat(), action, json.dumps(details or {})))
        self._conn.commit()

    def list_audit(self, limit=50) -> list:
        c = self._conn.cursor()
        c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

    # Flows
    def save_flow(self, flow_id: str, flow_data: dict):
        c = self._conn.cursor()
        c.execute("""INSERT OR REPLACE INTO flows
            (id, name, description, nodes, triggers, version, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (flow_id, flow_data.get("name", ""), flow_data.get("description", ""),
             json.dumps(flow_data.get("nodes", [])),
             json.dumps(flow_data.get("triggers", [])),
             flow_data.get("version", "1.0"),
             flow_data.get("created_at", datetime.now().isoformat()),
             datetime.now().isoformat()))
        self._conn.commit()

    def list_flows_sqlite(self) -> list:
        c = self._conn.cursor()
        c.execute("SELECT id, name, description, nodes, version FROM flows ORDER BY updated_at DESC")
        rows = c.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["nodes"] = json.loads(d["nodes"]) if isinstance(d["nodes"], str) else d["nodes"]
                d["node_count"] = len(d["nodes"]) if isinstance(d["nodes"], list) else 0
            except Exception:
                d["node_count"] = 0
            result.append(d)
        return result

    def close(self):
        self._conn.close()


class MemoryStore:
    def __init__(self, memory_dir: Path, cache_size: int = 100, heartbeat_ttl_days: int = 7,
                 use_sqlite: bool = True):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self._experiments_dir = self.memory_dir / "experiments"
        self._errors_dir = self.memory_dir / "errors"
        self._datasets_dir = self.memory_dir / "datasets"
        self._agent_dir = self.memory_dir / "agent"
        self._heartbeat_dir = self.memory_dir / "heartbeat"
        self._audit_dir = self.memory_dir / "audit"
        self._jobs_dir = self.memory_dir / "jobs"

        for d in [self._experiments_dir, self._errors_dir, self._datasets_dir,
                  self._agent_dir, self._heartbeat_dir, self._audit_dir, self._jobs_dir]:
            d.mkdir(exist_ok=True)

        self._experiment_cache = OrderedDict()
        self._cache_size = cache_size
        self._heartbeat_ttl_days = heartbeat_ttl_days

        # SQLite persistence layer
        self._sqlite = SQLiteStore(self.memory_dir / "epsionic.db") if use_sqlite else None

    def save_state(self) -> dict:
        """Persist all agent state to SQLite."""
        if not self._sqlite:
            return {"success": False, "error": "SQLite not enabled"}
        count = 0
        for f in self._experiments_dir.glob("*.json"):
            try:
                exp = json.loads(f.read_text())
                self._sqlite.save_experiment(exp)
                count += 1
            except Exception:
                pass
        for f in self._jobs_dir.glob("*.json"):
            try:
                job = json.loads(f.read_text())
                self._sqlite.save_job(job)
                count += 1
            except Exception:
                pass
        for f in self._agent_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self._sqlite.save_agent_state(f.stem, data)
                count += 1
            except Exception:
                pass
        self._sqlite.save_audit("state_saved", {"records": count})
        return {"success": True, "records_saved": count}

    def restore_state(self) -> dict:
        """Restore agent state from SQLite."""
        if not self._sqlite:
            return {"success": False, "error": "SQLite not enabled"}
        experiments = self._sqlite.list_experiments_sqlite()
        for exp in experiments:
            self._write_experiment(exp["id"], exp)
        return {"success": True, "experiments_restored": len(experiments)}

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
