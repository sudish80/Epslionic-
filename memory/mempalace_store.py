"""
MemPalaceStore — spatial memory system inspired by the Method of Loci.

Organizes agent memory as a Palace with Wings (domains), Rooms (experiments),
Drawers (individual memories), and Halls (categories). Includes a Knowledge
Graph for entity relationships and an Agent Diary for timeline/journal entries.

Backed by SQLite (stdlib, zero external dependencies).
"""

import json
import logging
import sqlite3
import csv
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("epsionic.mempalace")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wings (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rooms (
    id TEXT PRIMARY KEY,
    wing TEXT NOT NULL REFERENCES wings(name),
    name TEXT NOT NULL,
    room_type TEXT DEFAULT 'experiment',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drawers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL REFERENCES rooms(id),
    hall TEXT DEFAULT 'general',
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    entity_type TEXT DEFAULT 'unknown',
    properties TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    valid_from TEXT,
    valid_to TEXT,
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    entry TEXT NOT NULL,
    topic TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rooms_wing ON rooms(wing);
CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status);
CREATE INDEX IF NOT EXISTS idx_drawers_room ON drawers(room_id);
CREATE INDEX IF NOT EXISTS idx_drawers_hall ON drawers(hall);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
CREATE INDEX IF NOT EXISTS idx_diary_agent ON diary(agent_name);
CREATE INDEX IF NOT EXISTS idx_diary_created ON diary(created_at);
"""


class MemPalaceStore:
    """Spatial memory store using Wings/Rooms/Drawers pattern.

    Compatible with MemoryStore's interface while adding semantic navigation,
    knowledge graphs, decay, and cross-experiment recall.

    Auto-creates the palace directory and SQLite database on init.
    """

    def __init__(self, palace_path: str | Path):
        self.palace_path = Path(palace_path)
        self.palace_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.palace_path / "palace.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        logger.info("MemPalaceStore initialized at %s", self.palace_path)

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ── Wing Management (Domains) ─────────────────────────────────

    def ensure_wing(self, name: str, description: str = "") -> str:
        name = name.lower().replace(" ", "_")
        self._conn.execute(
            "INSERT OR IGNORE INTO wings (name, description, created_at) VALUES (?,?,?)",
            (name, description, datetime.now().isoformat()),
        )
        self._conn.commit()
        return name

    def list_wings(self) -> List[dict]:
        c = self._conn.execute("SELECT * FROM wings ORDER BY name")
        return [dict(r) for r in c.fetchall()]

    # ── Room Management (Experiments) ─────────────────────────────

    def create_experiment(self, name: str, config: dict = None,
                          tags: list = None, domain: str = "general") -> str:
        wing = self.ensure_wing(domain, f"Domain: {domain}")
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO rooms (id, wing, name, room_type, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (exp_id, wing, name, "experiment", "created", now, now),
        )
        config_drawer = {
            "config": config or {},
            "tags": tags or [],
            "domain": domain,
        }
        self._add_drawer(exp_id, "config", json.dumps(config_drawer, default=str))
        self._conn.commit()

        # KG: experiment has_status created
        self._ensure_entity(exp_id, "experiment")
        self._ensure_entity("created", "status")
        self._add_triple(exp_id, "has_status", "created", source="create_experiment")

        logger.info("Created experiment %s in wing %s", exp_id, wing)
        return exp_id

    def update_experiment(self, exp_id: str, **updates):
        c = self._conn.execute("SELECT * FROM rooms WHERE id=?", (exp_id,))
        room = c.fetchone()
        if not room:
            logger.warning("Experiment %s not found", exp_id)
            return
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE rooms SET updated_at=? WHERE id=?", (now, exp_id),
        )

        for k, v in updates.items():
            if k == "metrics" and isinstance(v, dict):
                raw = self._get_last_drawer(exp_id, "metrics")
                existing = json.loads(raw) if raw else {}
                merged = dict(existing)
                merged.update(v)
                self._add_drawer(exp_id, "metrics", json.dumps(merged, default=str))
            elif k == "errors" and isinstance(v, list):
                for err in v:
                    self._add_drawer(exp_id, "errors",
                                     json.dumps(err, default=str))
            elif k == "artifacts" and isinstance(v, list):
                for art in v:
                    self._add_drawer(exp_id, "artifacts",
                                     json.dumps(art, default=str))
            elif k == "status":
                self._conn.execute(
                    "UPDATE rooms SET status=? WHERE id=?", (v, exp_id))
                self._ensure_entity(v, "status")
                self._add_triple(exp_id, "has_status", v,
                                 source="update_experiment")
            else:
                drawer = {"key": k, "value": v}
                self._add_drawer(exp_id, k, json.dumps(drawer, default=str))

        self._conn.commit()

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        c = self._conn.execute("SELECT * FROM rooms WHERE id=?", (exp_id,))
        room = c.fetchone()
        if not room:
            return None
        room_dict = dict(room)

        drawers = self._list_drawers(exp_id)
        config_drawer = self._get_last_drawer(exp_id, "config")
        metrics_drawer = self._get_last_drawer(exp_id, "metrics")
        errors_drawers = [d for d in drawers if d["hall"] == "errors"]

        room_dict["config"] = json.loads(config_drawer) if config_drawer else {}
        room_dict["metrics"] = json.loads(metrics_drawer) if metrics_drawer else {}
        room_dict["errors"] = [json.loads(d["content"]) for d in errors_drawers
                               if isinstance(d.get("content"), str)]
        room_dict["tags"] = room_dict.get("config", {}).get("tags", [])
        return room_dict

    def list_experiments(self, status: str = None, domain: str = None,
                         tags: list = None) -> List[dict]:
        query = "SELECT * FROM rooms WHERE room_type='experiment'"
        params = []
        if status:
            query += " AND status=?"
            params.append(status)
        if domain:
            wing = domain.lower().replace(" ", "_")
            query += " AND wing=?"
            params.append(wing)

        query += " ORDER BY created_at DESC"
        c = self._conn.execute(query, params)
        rooms = [dict(r) for r in c.fetchall()]

        result = []
        for room in rooms:
            config_d = self._get_last_drawer(room["id"], "config")
            metrics_d = self._get_last_drawer(room["id"], "metrics")
            config = json.loads(config_d) if config_d else {}
            metrics = json.loads(metrics_d) if metrics_d else {}
            exp_tags = config.get("tags", []) if isinstance(config, dict) else []

            if tags is not None:
                if not any(t in exp_tags for t in tags):
                    continue

            result.append({
                "id": room["id"],
                "name": room["name"],
                "status": room["status"],
                "domain": room["wing"],
                "config": config if isinstance(config, dict) else {},
                "metrics": metrics if isinstance(metrics, dict) else {},
                "tags": exp_tags,
                "created_at": room["created_at"],
                "updated_at": room["updated_at"],
            })

        return result

    def export_experiments_csv(self, path: str = None) -> str:
        exps = self.list_experiments()
        if not exps:
            return "No experiments to export"
        output = io.StringIO()
        w = csv.DictWriter(output, fieldnames=["id", "name", "status", "domain",
                                                "created_at", "updated_at"])
        w.writeheader()
        for e in exps:
            w.writerow({k: e.get(k, "") for k in
                       ["id", "name", "status", "domain", "created_at", "updated_at"]})
        if path:
            Path(path).write_text(output.getvalue())
            return path
        return output.getvalue()

    # ── Error Logging ────────────────────────────────────────────

    def log_error(self, source: str, error: str, context: dict = None,
                  exp_id: str = None) -> str:
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        record = {
            "id": error_id,
            "source": source,
            "error": str(error)[:500],
            "context": context or {},
            "fixed": False,
            "timestamp": datetime.now().isoformat(),
        }
        if exp_id:
            self._add_drawer(exp_id, "errors", json.dumps(record, default=str))
        else:
            self.ensure_wing("_system")
            self._create_room_if_missing("_orphan_errors", "_system")
            self._add_drawer("_orphan_errors", source, json.dumps(record, default=str))
        self._conn.commit()

        # Extract error pattern for KG
        pattern = self._extract_error_pattern(str(error))
        self._ensure_entity(pattern, "error_pattern")
        self._ensure_entity(source, "source")
        self._add_triple(pattern, "occurred_in", source,
                         source="log_error", confidence=0.7)

        return error_id

    def mark_error_fixed(self, error_id: str, fix: str = ""):
        c = self._conn.execute(
            "SELECT id FROM drawers WHERE content LIKE ?",
            (f"%{error_id}%",),
        )
        row = c.fetchone()
        if row:
            drawer_id = row["id"]
            c2 = self._conn.execute(
                "SELECT content FROM drawers WHERE id=?", (drawer_id,))
            content = json.loads(c2.fetchone()["content"])
            content["fixed"] = True
            content["fix_description"] = fix
            content["fixed_at"] = datetime.now().isoformat()
            self._conn.execute(
                "UPDATE drawers SET content=? WHERE id=?",
                (json.dumps(content, default=str), drawer_id),
            )
            self._conn.commit()

    def get_unfixed_errors(self) -> List[dict]:
        c = self._conn.execute(
            "SELECT content FROM drawers WHERE hall='errors' OR room_id='_orphan_errors'")
        errors = []
        for row in c.fetchall():
            try:
                err = json.loads(row["content"])
                if not err.get("fixed"):
                    errors.append(err)
            except (json.JSONDecodeError, TypeError):
                pass
        return errors

    def get_error_summary(self) -> str:
        errors = self.get_unfixed_errors()
        if not errors:
            return "No unresolved errors."
        lines = ["### Unresolved Errors", ""]
        for e in errors[:10]:
            lines.append(f"- **{e.get('source', 'unknown')}** ({e.get('id', '?')}): {e.get('error', '')[:200]}")
        return "\n".join(lines)

    # ── Dataset Recording ────────────────────────────────────────

    def record_dataset(self, dataset_id: str, metadata: dict,
                       domain: str = None):
        wing = domain or "general"
        self.ensure_wing(wing)
        record = dict(metadata or {})
        record["dataset_id"] = dataset_id
        record["recorded_at"] = datetime.now().isoformat()
        self._add_drawer("_datasets", dataset_id.replace("/", "_"),
                         json.dumps(record, default=str))
        self._ensure_entity(dataset_id, "dataset")
        if domain:
            self._ensure_entity(domain, "domain")
            self._add_triple(dataset_id, "used_in_domain", domain,
                             source="record_dataset")

    # ── Heartbeat / Diary ────────────────────────────────────────

    def write_heartbeat_log(self, status: str, details: str = ""):
        self._diary_write("heartbeat", f"{status}: {details[:500]}")

    def get_recent_heartbeats(self, n: int = 10) -> List[dict]:
        return self._diary_read("heartbeat", n)

    # ── Agent State ──────────────────────────────────────────────

    def write_agent_state(self, key: str, data: Any):
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_state (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(data, default=str), datetime.now().isoformat()),
        )
        self._conn.commit()

    def read_agent_state(self, key: str) -> Optional[Any]:
        c = self._conn.execute(
            "SELECT value FROM agent_state WHERE key=?", (key,))
        row = c.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return row["value"]
        return None

    # ── State Persistence ────────────────────────────────────────

    def save_state(self) -> dict:
        return {"success": True, "store_type": "mempalace",
                "path": str(self.db_path)}

    def restore_state(self) -> dict:
        c = self._conn.execute("SELECT COUNT(*) as cnt FROM rooms")
        count = c.fetchone()["cnt"]
        return {"success": True, "experiments_restored": count}

    # ── Training Summary ─────────────────────────────────────────

    def get_training_summary_markdown(self) -> str:
        lines = ["# MemPalace Training Summary", ""]
        wings = self.list_wings()
        for wing in wings:
            c = self._conn.execute(
                "SELECT * FROM rooms WHERE wing=? AND room_type='experiment' ORDER BY created_at DESC LIMIT 5",
                (wing["name"],),
            )
            rooms = [dict(r) for r in c.fetchall()]
            if rooms:
                lines.append(f"## Wing: {wing['name']}")
                for room in rooms:
                    metrics_d = self._get_last_drawer(room["id"], "metrics")
                    metrics = json.loads(metrics_d) if metrics_d else {}
                    metric_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    )
                    lines.append(
                        f"- {room['name']} ({room['status']}) "
                        f"[{room['created_at'][:10]}]"
                    )
                    if metric_str:
                        lines.append(f"  - Metrics: {metric_str}")
                lines.append("")
        if not wings:
            lines.append("No wings yet. Start training to populate the palace.")
        return "\n".join(lines)

    # ── Enhanced MemPalace Features ──────────────────────────────

    def wake_up(self, wing: str = None) -> str:
        """L0 + L1 context: identity + active experiment summaries."""
        parts = ["[MemPalace L0: Identity]", "Role: Autonomous ML Training Agent",
                 "Palace: {}".format(self.palace_path), ""]
        if wing:
            parts.append("[L1: Active Wing: {}]".format(wing))
            c = self._conn.execute(
                "SELECT * FROM rooms WHERE wing=? AND room_type='experiment' "
                "AND status IN ('running','created') ORDER BY created_at DESC LIMIT 5",
                (wing,),
            )
            rooms = [dict(r) for r in c.fetchall()]
            for room in rooms:
                parts.append("- Room: {} ({}) [{}]".format(
                    room["name"], room["status"], room["created_at"][:10]))
                metrics_d = self._get_last_drawer(room["id"], "metrics")
                if metrics_d:
                    metrics = json.loads(metrics_d)
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            parts.append("  - {}: {:.4f}".format(k, v))
        else:
            wings_list = self.list_wings()
            parts.append("[L1: Active Wings: {}]".format(
                ", ".join(w["name"] for w in wings_list[:5])))
        return "\n".join(parts)

    def recall(self, wing: str = None, room: str = None,
               hall: str = None, n: int = 10) -> str:
        """L2 on-demand retrieval: get drawers filtered by wing/room/hall."""
        query = "SELECT d.*, r.wing FROM drawers d JOIN rooms r ON d.room_id=r.id WHERE 1=1"
        params = []
        if wing:
            query += " AND r.wing=?"
            params.append(wing)
        if room:
            query += " AND d.room_id=?"
            params.append(room)
        if hall:
            query += " AND d.hall=?"
            params.append(hall)
        query += " ORDER BY d.created_at DESC LIMIT ?"
        params.append(n)

        c = self._conn.execute(query, params)
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            return "[MemPalace L2: No results]"
        parts = ["[MemPalace L2: Recall]", ""]
        for r in rows:
            parts.append("- Room: {} | Hall: {} | {}".format(
                r["room_id"][:30], r["hall"], r["created_at"][:19]))
            content = r["content"][:200]
            parts.append("  {}".format(content))
        return "\n".join(parts)

    def search(self, query: str, wing: str = None, room: str = None,
               n: int = 5) -> List[dict]:
        """L3 semantic/keyword search across all drawers.

        Uses SQLite FTS5 for full-text search. Falls back to LIKE if FTS5
        is unavailable.
        """
        results = []
        try:
            sql = ("SELECT d.id, d.content, d.hall, d.created_at, r.wing, r.name as room_name "
                   "FROM drawers_fts f JOIN drawers d ON f.rowid=d.id "
                   "JOIN rooms r ON d.room_id=r.id "
                   "WHERE drawers_fts MATCH ?")
            params = [query]
            if wing:
                sql += " AND r.wing=?"
                params.append(wing)
            if room:
                sql += " AND d.room_id=?"
                params.append(room)
            sql += " ORDER BY rank LIMIT ?"
            params.append(n)
            c = self._conn.execute(sql, params)
            results = [dict(r) for r in c.fetchall()]
        except (sqlite3.OperationalError, AttributeError):
            pass

        if not results:
            c2 = self._conn.execute(
                "SELECT d.*, r.wing, r.name as room_name FROM drawers d "
                "JOIN rooms r ON d.room_id=r.id WHERE d.content LIKE ? ORDER BY d.created_at DESC LIMIT ?",
                ("%{}%".format(query), n),
            )
            results = [dict(r) for r in c2.fetchall()]

        for r in results:
            if isinstance(r.get("content"), str):
                try:
                    r["parsed"] = json.loads(r["content"])
                except (json.JSONDecodeError, TypeError):
                    r["parsed"] = r["content"]
        return results

    # ── Knowledge Graph ──────────────────────────────────────────

    def add_knowledge(self, subject: str, predicate: str, obj: str,
                      confidence: float = 1.0, source: str = ""):
        self._ensure_entity(subject)
        self._ensure_entity(obj)
        self._add_triple(subject, predicate, obj, confidence=confidence,
                         source=source)
        self._conn.commit()

    def query_knowledge(self, entity: str) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM triples WHERE subject=? OR object=? ORDER BY created_at DESC",
            (entity, entity),
        )
        return [dict(r) for r in c.fetchall()]

    def get_knowledge_graph_stats(self) -> dict:
        c1 = self._conn.execute("SELECT COUNT(*) as c FROM entities")
        c2 = self._conn.execute("SELECT COUNT(*) as c FROM triples")
        c3 = self._conn.execute("SELECT DISTINCT predicate FROM triples")
        return {
            "entities": c1.fetchone()["c"],
            "triples": c2.fetchone()["c"],
            "relationship_types": [r["predicate"] for r in c3.fetchall()],
        }

    # ── Training-Specific Extensions ─────────────────────────────

    def search_similar_experiments(self, config: dict, domain: str = None,
                                   n: int = 5) -> List[dict]:
        """Find past experiments with similar hyperparameters."""
        query_parts = []
        for k, v in config.items():
            if isinstance(v, (str, int, float)):
                query_parts.append("{}:{}".format(k, v))
        query = " ".join(query_parts)
        return self.search(query, wing=domain, n=n)

    def get_best_hyperparams(self, domain: str, metric: str = "eval_loss",
                             n: int = 3, minimize: bool = True) -> List[dict]:
        """Return top-N configs sorted by metric value."""
        c = self._conn.execute(
            "SELECT d.*, r.wing FROM drawers d JOIN rooms r ON d.room_id=r.id "
            "WHERE d.hall='metrics' AND r.wing=? ORDER BY d.created_at DESC LIMIT 50",
            (domain,),
        )
        candidates = []
        for row in c.fetchall():
            try:
                metrics = json.loads(row["content"])
                if metric in metrics:
                    config_d = self._get_last_drawer(row["room_id"], "config")
                    config = json.loads(config_d) if config_d else {}
                    candidates.append({
                        "room_id": row["room_id"],
                        metric: metrics[metric],
                        "config": config.get("config", config) if isinstance(config, dict) else config,
                        "created_at": row["created_at"],
                    })
            except (json.JSONDecodeError, TypeError):
                continue

        candidates.sort(key=lambda x: x.get(metric, float("inf")),
                        reverse=not minimize)
        return candidates[:n]

    def recall_auto_fix(self, error_pattern: str) -> Optional[dict]:
        """Query KG for similar errors and their fixes."""
        c = self._conn.execute(
            "SELECT * FROM triples WHERE subject LIKE ? AND predicate='fixed_by' "
            "ORDER BY confidence DESC LIMIT 1",
            ("%{}%".format(error_pattern[:30]),),
        )
        row = c.fetchone()
        if row:
            return {
                "error_pattern": row["subject"],
                "fix": row["object"],
                "confidence": row["confidence"],
                "source": row["source"],
            }
        return None

    def get_cost_effective(self, domain: str, budget: float,
                           n: int = 5) -> List[dict]:
        """Find experiments within budget, sorted by best metric."""
        c = self._conn.execute(
            "SELECT content FROM drawers WHERE hall='metrics' ORDER BY created_at DESC LIMIT 100",
        )
        candidates = []
        for row in c.fetchall():
            try:
                metrics = json.loads(row["content"])
                cost = metrics.get("cost", 0) or metrics.get("total_cost", 0)
                if cost <= budget:
                    candidates.append({
                        "metrics": metrics,
                        "cost": cost,
                    })
            except (json.JSONDecodeError, TypeError):
                continue
        return sorted(candidates, key=lambda x: x["cost"])[:n]

    def record_training_run(self, exp_id: str, domain: str, model: str,
                            dataset: str, config: dict, metrics: dict,
                            cost: float):
        """Batch record a complete training run with KG relationships."""
        self.ensure_wing(domain)
        self._ensure_entity(model, "model")
        self._ensure_entity(dataset, "dataset")
        self._add_triple(exp_id, "trained_on", model, source="record_training_run")
        self._add_triple(exp_id, "used_dataset", dataset,
                         source="record_training_run")
        self._add_drawer(exp_id, "metrics",
                         json.dumps({**metrics, "cost": cost}, default=str))
        if cost > 0:
            self._add_triple(exp_id, "had_cost", str(round(cost, 2)),
                             source="record_training_run")
        self._conn.commit()

    # ── Internal Helpers ─────────────────────────────────────────

    def _create_room_if_missing(self, room_id: str, wing: str = "_system"):
        self._conn.execute(
            "INSERT OR IGNORE INTO rooms (id, name, wing, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (room_id, room_id, wing, "internal",
             datetime.now().isoformat(), datetime.now().isoformat()),
        )

    def _add_drawer(self, room_id: str, hall: str, content: str,
                    metadata: dict = None):
        self._create_room_if_missing(room_id)
        self._conn.execute(
            "INSERT INTO drawers (room_id, hall, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (room_id, hall, content, json.dumps(metadata or {}),
             datetime.now().isoformat()),
        )

    def _list_drawers(self, room_id: str) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM drawers WHERE room_id=? ORDER BY id DESC",
            (room_id,),
        )
        return [dict(r) for r in c.fetchall()]

    def _get_last_drawer(self, room_id: str, hall: str) -> Optional[str]:
        c = self._conn.execute(
            "SELECT content FROM drawers WHERE room_id=? AND hall=? "
            "ORDER BY id DESC LIMIT 1",
            (room_id, hall),
        )
        row = c.fetchone()
        return row["content"] if row else None

    def _ensure_entity(self, name: str, entity_type: str = "unknown"):
        name = str(name)[:200]
        self._conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, properties, created_at) VALUES (?,?,?,?)",
            (name, entity_type, "{}", datetime.now().isoformat()),
        )

    def _add_triple(self, subject: str, predicate: str, obj: str,
                    confidence: float = 1.0, source: str = ""):
        self._conn.execute(
            "INSERT INTO triples (subject, predicate, object, confidence, "
            "valid_from, source, created_at) VALUES (?,?,?,?,?,?,?)",
            (str(subject)[:200], str(predicate)[:100], str(obj)[:200],
             confidence, datetime.now().isoformat(), source,
             datetime.now().isoformat()),
        )

    def _diary_write(self, agent_name: str, entry: str, topic: str = ""):
        self._conn.execute(
            "INSERT INTO diary (agent_name, entry, topic, created_at) VALUES (?,?,?,?)",
            (agent_name, entry, topic, datetime.now().isoformat()),
        )
        self._conn.commit()

    def _diary_read(self, agent_name: str, last_n: int = 10) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM diary WHERE agent_name=? ORDER BY id DESC LIMIT ?",
            (agent_name, last_n),
        )
        return [dict(r) for r in c.fetchall()]

    @staticmethod
    def _extract_error_pattern(error: str) -> str:
        patterns = [
            r"CUDA out of memory",
            r"ModuleNotFoundError|No module named",
            r"expected scalar type",
            r"NaN|inf|division by zero",
            r"ConnectionError|timeout|Connection refused",
            r"HTTP Error 503|HTTP Error 429",
            r"NoneType|object has no attribute",
            r"out of memory|OOM|Killed",
        ]
        for p in patterns:
            m = re.search(p, error, re.IGNORECASE)
            if m:
                return m.group(0)[:100]
        return "unknown_error"

    @property
    def store_type(self) -> str:
        return "mempalace"

    @property
    def exp_dir(self):
        return self.palace_path

    @property
    def err_dir(self):
        return self.palace_path

    @property
    def heart_dir(self):
        return self.palace_path

    @property
    def ds_dir(self):
        return self.palace_path

    @property
    def base(self):
        return self.palace_path
