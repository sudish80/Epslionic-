"""Prompt Manager — version-controlled prompt templates with A/B testing support."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("epsionic.prompts")

_PROMPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    template TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    variables TEXT DEFAULT '[]',
    tags TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id TEXT NOT NULL REFERENCES prompt_templates(id),
    version INTEGER NOT NULL,
    template TEXT NOT NULL,
    changelog TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ab_tests (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt_a_id TEXT NOT NULL,
    prompt_b_id TEXT NOT NULL,
    metric TEXT DEFAULT 'response_length',
    status TEXT DEFAULT 'running',
    created_at TEXT NOT NULL,
    results TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ab_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT NOT NULL REFERENCES ab_tests(id),
    variant TEXT NOT NULL,
    prompt_template TEXT NOT NULL,
    input_text TEXT,
    output_text TEXT,
    latency_ms REAL DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    scored_value REAL,
    created_at TEXT NOT NULL
);
"""


class PromptManager:
    """Version-controlled prompt templates with A/B testing."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_PROMPT_SCHEMA)
        self._conn.commit()

    def create_template(self, name: str, template: str, variables: list = None, tags: str = "") -> dict:
        """Create a new prompt template."""
        import uuid
        tid = f"pt_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO prompt_templates (id, name, template, version, variables, tags, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (tid, name, template, 1, json.dumps(variables or []), tags, now, now),
        )
        self._conn.execute(
            "INSERT INTO prompt_versions (template_id, version, template, changelog, created_at) VALUES (?,?,?,?,?)",
            (tid, 1, template, "Initial version", now),
        )
        self._conn.commit()
        return {"id": tid, "name": name, "version": 1}

    def get_template(self, template_id: str) -> Optional[dict]:
        c = self._conn.execute("SELECT * FROM prompt_templates WHERE id=?", (template_id,))
        row = c.fetchone()
        return dict(row) if row else None

    def list_templates(self, tag: str = None) -> List[dict]:
        query = "SELECT * FROM prompt_templates"
        params = []
        if tag:
            query += " WHERE tags LIKE ?"
            params.append(f"%{tag}%")
        query += " ORDER BY updated_at DESC"
        c = self._conn.execute(query, params)
        return [dict(r) for r in c.fetchall()]

    def update_template(self, template_id: str, template: str, changelog: str = "") -> Optional[dict]:
        """Create a new version of a prompt template."""
        existing = self.get_template(template_id)
        if not existing:
            return None
        new_version = existing["version"] + 1
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE prompt_templates SET template=?, version=?, updated_at=? WHERE id=?",
            (template, new_version, now, template_id),
        )
        self._conn.execute(
            "INSERT INTO prompt_versions (template_id, version, template, changelog, created_at) VALUES (?,?,?,?,?)",
            (template_id, new_version, template, changelog or f"Version {new_version}", now),
        )
        self._conn.commit()
        return self.get_template(template_id)

    def get_version_history(self, template_id: str) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE template_id=? ORDER BY version DESC",
            (template_id,),
        )
        return [dict(r) for r in c.fetchall()]

    def render(self, template_id: str, variables: dict = None) -> Optional[str]:
        """Render a template with variable substitution."""
        tmpl = self.get_template(template_id)
        if not tmpl:
            return None
        text = tmpl["template"]
        for k, v in (variables or {}).items():
            text = text.replace(f"{{{k}}}", str(v))
        return text

    def create_ab_test(self, name: str, prompt_a_id: str, prompt_b_id: str, metric: str = "response_length") -> dict:
        """Create an A/B test comparing two prompt templates."""
        import uuid
        tid = f"ab_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO ab_tests (id, name, prompt_a_id, prompt_b_id, metric, status, created_at, results) VALUES (?,?,?,?,?,?,?,?)",
            (tid, name, prompt_a_id, prompt_b_id, metric, "running", now, "{}"),
        )
        self._conn.commit()
        return {"id": tid, "name": name}

    def record_ab_result(self, test_id: str, variant: str, prompt_template: str, input_text: str = "",
                         output_text: str = "", latency_ms: float = 0, token_count: int = 0,
                         scored_value: float = None):
        self._conn.execute(
            "INSERT INTO ab_results (test_id, variant, prompt_template, input_text, output_text, latency_ms, token_count, scored_value, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (test_id, variant, prompt_template, input_text, output_text, latency_ms, token_count, scored_value, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_ab_results(self, test_id: str) -> dict:
        c = self._conn.execute(
            "SELECT * FROM ab_results WHERE test_id=? ORDER BY created_at", (test_id,)
        )
        rows = [dict(r) for r in c.fetchall()]
        test = self._conn.execute("SELECT * FROM ab_tests WHERE id=?", (test_id,)).fetchone()
        return {
            "test": dict(test) if test else None,
            "results": rows,
        }

    def get_ab_summary(self, test_id: str) -> dict:
        """Get aggregated A/B test results."""
        data = self.get_ab_results(test_id)
        results = data["results"]
        by_variant = {"a": [], "b": []}
        for r in results:
            variant = r.get("variant", "a").lower()
            if variant in by_variant:
                by_variant[variant].append(r)
        summary = {}
        for var, items in by_variant.items():
            if not items:
                continue
            latencies = [i.get("latency_ms", 0) for i in items if i.get("latency_ms")]
            tokens = [i.get("token_count", 0) for i in items if i.get("token_count")]
            scores = [i.get("scored_value", 0) for i in items if i.get("scored_value") is not None]
            summary[var] = {
                "count": len(items),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "avg_token_count": sum(tokens) / len(tokens) if tokens else 0,
                "avg_score": sum(scores) / len(scores) if scores else None,
            }
        return {"test_id": test_id, "variants": summary, "total_results": len(results)}

    def close(self):
        self._conn.close()

    def get_tool_description(self) -> dict:
        return {
            "name": "prompt_manager",
            "description": "Manage version-controlled prompt templates and run A/B tests",
            "parameters": {
                "action": {"type": "string", "enum": ["create", "list", "get", "update", "render", "create_ab_test", "ab_summary"]},
                "name": {"type": "string", "optional": True},
                "template": {"type": "string", "optional": True},
                "template_id": {"type": "string", "optional": True},
                "variables": {"type": "object", "optional": True},
            },
        }
