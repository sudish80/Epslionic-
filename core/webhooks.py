"""Webhook Manager — POST to external URLs on training events."""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("epsionic.webhooks")


class WebhookManager:
    """Fire webhooks to external URLs on agent events (train complete, error, etc.)."""

    def __init__(self, db_path: str | Path = None):
        self._hooks: List[dict] = []
        self._counter = 0
        self._executor = threading.Thread(target=self._worker, daemon=True)
        self._executor.start()

    def register(self, url: str, events: List[str], secret: str = "", headers: dict = None) -> dict:
        """Register a webhook for one or more events."""
        self._counter += 1
        hook = {
            "id": f"wh_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._counter}",
            "url": url,
            "events": events,
            "secret": secret,
            "headers": headers or {},
            "created_at": datetime.now().isoformat(),
            "enabled": True,
        }
        self._hooks.append(hook)
        logger.info("Registered webhook %s -> %s (%s)", hook["id"], url, events)
        return hook

    def remove(self, hook_id: str) -> bool:
        before = len(self._hooks)
        self._hooks[:] = [h for h in self._hooks if h["id"] != hook_id]
        return len(self._hooks) < before

    def list(self) -> List[dict]:
        return list(self._hooks)

    def fire(self, event: str, payload: dict):
        """Fire all webhooks registered for the given event."""
        for hook in self._hooks:
            if not hook["enabled"]:
                continue
            if event not in hook["events"]:
                continue
            threading.Thread(
                target=self._send,
                args=(hook, event, payload),
                daemon=True,
            ).start()

    def _send(self, hook: dict, event: str, payload: dict):
        """Send a single webhook request."""
        import urllib.request
        body = json.dumps({
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "payload": payload,
        }, default=str).encode()
        req = urllib.request.Request(
            hook["url"],
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Event": event,
                "X-Webhook-Id": hook["id"],
                **hook["headers"],
            },
            method="POST",
        )
        if hook.get("secret"):
            import hmac, hashlib
            sig = hmac.new(
                hook["secret"].encode(), body, hashlib.sha256
            ).hexdigest()
            req.add_header("X-Webhook-Signature", sig)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug("Webhook %s -> %s: %s", hook["id"], event, resp.status)
        except Exception as e:
            logger.warning("Webhook %s failed: %s", hook["id"], e)

    def _worker(self):
        """Background worker (keeps thread alive)."""
        import time
        while True:
            time.sleep(1)
