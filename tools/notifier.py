"""Multi-channel Notifier — Email, Slack, Discord, Telegram, Pushover.
Sends alerts on training completion, errors, and budget thresholds."""

import json
import logging
import os
import smtplib
import time
from email.message import EmailMessage
from typing import Optional, Dict, Any, List
from pathlib import Path
from urllib.parse import urlencode

logger = logging.getLogger("openclaw.tool.notifier")


class Notifier:
    """Send notifications through multiple channels."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._rate_limits: Dict[str, float] = {}

    def send(self, title: str, message: str, channels: List[str] = None,
             level: str = "info") -> dict:
        """Send notification to all configured channels."""
        channels = channels or self.config.get("channels", ["log"])
        results = {}

        for ch in channels:
            if not self._check_rate_limit(ch):
                results[ch] = {"success": False, "error": "rate_limited"}
                continue
            try:
                fn = getattr(self, f"_send_{ch}", None)
                if fn:
                    fn(title, message, level)
                    results[ch] = {"success": True}
                else:
                    results[ch] = {"success": False, "error": f"Unknown channel: {ch}"}
            except Exception as e:
                results[ch] = {"success": False, "error": str(e)}
        return results

    def notify_done(self, experiment_name: str, metrics: dict = None,
                    channels: List[str] = None):
        """Shorthand for training completion notification."""
        m = metrics or {}
        msg = (
            f"Experiment: {experiment_name}\n"
            f"Status: Complete\n"
            f"Loss: {m.get('eval_loss', 'N/A')}\n"
            f"Duration: {m.get('duration', 'N/A')}\n"
        )
        return self.send("Training Complete", msg, channels)

    def notify_error(self, experiment_name: str, error: str,
                     channels: List[str] = None):
        return self.send("Training Failed", f"{experiment_name}: {error}", channels, level="error")

    def notify_budget(self, cost: float, limit: float, channels: List[str] = None):
        pct = (cost / limit) * 100 if limit > 0 else 0
        return self.send("Budget Alert", f"Cost: ${cost:.2f} / ${limit:.2f} ({pct:.1f}%)", channels, level="warn")

    def _send_log(self, title: str, message: str, level: str):
        fn = getattr(logger, level, logger.info)
        fn(f"[{title}] {message}")

    def _send_email(self, title: str, message: str, level: str):
        cfg = self.config.get("email", {})
        if not cfg.get("smtp_server"):
            return
        msg = EmailMessage()
        msg.set_content(message)
        msg["Subject"] = f"[OpenClaw] {title}"
        msg["From"] = cfg.get("from_addr")
        msg["To"] = cfg.get("to_addr")
        with smtplib.SMTP(cfg["smtp_server"], cfg.get("smtp_port", 587)) as s:
            if cfg.get("use_tls", True):
                s.starttls()
            if cfg.get("username"):
                s.login(cfg["username"], cfg.get("password", ""))
            s.send_message(msg)

    def _send_slack(self, title: str, message: str, level: str):
        import requests
        webhook = self.config.get("slack", {}).get("webhook_url")
        if not webhook:
            return
        color = {"info": "good", "warn": "warning", "error": "danger"}.get(level, "good")
        requests.post(webhook, json={
            "attachments": [{"color": color, "title": title, "text": message}],
        }, timeout=10)

    def _send_discord(self, title: str, message: str, level: str):
        import requests
        webhook = self.config.get("discord", {}).get("webhook_url")
        if not webhook:
            return
        requests.post(webhook, json={"content": f"**{title}**\n{message}"}, timeout=10)

    def _send_telegram(self, title: str, message: str, level: str):
        import requests
        cfg = self.config.get("telegram", {})
        bot_token = cfg.get("bot_token")
        chat_id = cfg.get("chat_id")
        if not bot_token or not chat_id:
            return
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": f"*{title}*\n{message}", "parse_mode": "Markdown"},
            timeout=10,
        )

    def _send_pushover(self, title: str, message: str, level: str):
        import requests
        cfg = self.config.get("pushover", {})
        if not cfg.get("user_key") or not cfg.get("api_token"):
            return
        requests.post("https://api.pushover.net/1/messages.json", json={
            "token": cfg["api_token"], "user": cfg["user_key"],
            "title": title, "message": message, "priority": 1 if level == "error" else 0,
        }, timeout=10)

    def _check_rate_limit(self, channel: str) -> bool:
        now = time.time()
        last = self._rate_limits.get(channel, 0)
        if now - last < 1.0:
            return False
        self._rate_limits[channel] = now
        return True

    def get_tool_description(self) -> dict:
        return {"send": {"description": "Send notification via configured channels",
            "parameters": {"title": "Notification title", "message": "Body",
                "channels": "List of channels (log, email, slack, discord, telegram, pushover)",
                "level": "info|warn|error"}}}
