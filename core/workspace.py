"""Workspace Manager — Multi-tenant workspace isolation with per-user API keys."""

import json
import logging
import sqlite3
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("epsionic.workspace")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    workspace_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS api_keys (
    key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id),
    label TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_used TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL REFERENCES tenants(id),
    action TEXT NOT NULL,
    details TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL
);
"""


class WorkspaceManager:
    """Multi-tenant workspace isolation with per-user API keys."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_tenant(self, name: str, workspace_root: str | Path = None) -> dict:
        """Create a new tenant with a generated API key."""
        import uuid
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        api_key = f"epsk_{secrets.token_hex(24)}"
        ws_path = str(Path(workspace_root or self.db_path.parent) / f"workspaces" / tenant_id)
        self._conn.execute(
            "INSERT INTO tenants (id, name, api_key, workspace_path, created_at) VALUES (?,?,?,?,?)",
            (tenant_id, name, api_key, ws_path, datetime.now().isoformat()),
        )
        self._conn.execute(
            "INSERT INTO api_keys (key, tenant_id, label, created_at) VALUES (?,?,?,?)",
            (api_key, tenant_id, "default", datetime.now().isoformat()),
        )
        self._conn.commit()
        Path(ws_path).mkdir(parents=True, exist_ok=True)
        logger.info("Created tenant %s (%s)", tenant_id, name)
        return {"tenant_id": tenant_id, "api_key": api_key, "workspace_path": ws_path}

    def get_tenant_by_key(self, api_key: str) -> Optional[dict]:
        """Look up a tenant by API key."""
        c = self._conn.execute(
            "SELECT * FROM tenants WHERE api_key=? AND is_active=1", (api_key,)
        )
        row = c.fetchone()
        if row:
            return dict(row)
        c = self._conn.execute(
            "SELECT t.* FROM tenants t JOIN api_keys k ON t.id=k.tenant_id WHERE k.key=? AND k.is_active=1",
            (api_key,),
        )
        row = c.fetchone()
        return dict(row) if row else None

    def generate_api_key(self, tenant_id: str, label: str = "") -> str:
        """Generate a new API key for an existing tenant."""
        api_key = f"epsk_{secrets.token_hex(24)}"
        self._conn.execute(
            "INSERT INTO api_keys (key, tenant_id, label, created_at) VALUES (?,?,?,?)",
            (api_key, tenant_id, label, datetime.now().isoformat()),
        )
        self._conn.commit()
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Deactivate an API key."""
        cur = self._conn.execute(
            "UPDATE api_keys SET is_active=0 WHERE key=?", (api_key,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_tenants(self) -> List[dict]:
        c = self._conn.execute("SELECT * FROM tenants ORDER BY created_at DESC")
        return [dict(r) for r in c.fetchall()]

    def list_api_keys(self, tenant_id: str) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM api_keys WHERE tenant_id=? ORDER BY created_at DESC",
            (tenant_id,),
        )
        return [dict(r) for r in c.fetchall()]

    def log_audit(self, tenant_id: str, action: str, details: dict = None):
        self._conn.execute(
            "INSERT INTO audit_log (tenant_id, action, details, timestamp) VALUES (?,?,?,?)",
            (tenant_id, action, json.dumps(details or {}), datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_audit_log(self, tenant_id: str, limit: int = 50) -> List[dict]:
        c = self._conn.execute(
            "SELECT * FROM audit_log WHERE tenant_id=? ORDER BY timestamp DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [dict(r) for r in c.fetchall()]

    def close(self):
        self._conn.close()
