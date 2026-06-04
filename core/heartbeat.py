"""
Heartbeat Monitor - Epslionic-inspired proactive training monitoring.
Checks training progress, detects failures, and triggers auto-fix.
"""

import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Dict

logger = logging.getLogger("epsionic.heartbeat")


class HeartbeatMonitor:
    def __init__(self, memory_store, interval_seconds: int = 60):
        self.memory = memory_store
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_fns: List[Callable[[], str]] = []

    def add_check(self, name: str, fn: Callable[[], str]):
        """Register a health check function. Returns a status string."""
        self._check_fns.append((name, fn))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"Heartbeat monitor started (interval={self.interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Heartbeat tick failed: {e}")
            time.sleep(self.interval)

    def _tick(self):
        issues = []
        all_ok = True

        for name, fn in self._check_fns:
            try:
                result = fn()
                if result:
                    issues.append(f"  - **{name}**: {result}")
                    all_ok = False
            except Exception as e:
                issues.append(f"  - **{name}**: check error - {e}")
                all_ok = False

        status = "HEARTBEAT_OK" if all_ok else "ISSUES_DETECTED"
        details = "\n".join(issues) if issues else "All systems operational."
        self.memory.write_heartbeat_log(status, details)

        if not all_ok:
            logger.warning(f"Heartbeat detected issues:\n{details}")

    def get_heartbeat_markdown(self) -> str:
        """Generate a HEARTBEAT.md style report."""
        recent = self.memory.get_recent_heartbeats(5)
        lines = ["# HEARTBEAT.md - Training Agent Status", ""]

        if recent:
            lines.append(f"Last check: {recent[-1]['timestamp']}")
            lines.append(f"Status: {recent[-1]['status']}")
            lines.append("")

        experiments = self.memory.list_experiments()
        running = [e for e in experiments if e.get("status") == "running"]
        failed = [e for e in experiments if e.get("status") == "failed"]

        if running:
            lines.append(f"### Active Training Runs ({len(running)})")
            for e in running:
                lines.append(f"- **{e['name']}**: started {e.get('created_at', '?')}")
            lines.append("")

        if failed:
            lines.append(f"### Failed Runs ({len(failed)})")
            for e in failed:
                lines.append(f"- **{e['name']}**: {len(e.get('errors', []))} errors")
            lines.append("")

        errors = self.memory.get_unfixed_errors()
        if errors:
            lines.append(f"### Unresolved Errors ({len(errors)})")
            for e in errors[:5]:
                lines.append(f"- [{e['source']}] {e['error'][:150]}")
            lines.append("")

        if not running and not failed and not errors:
            lines.append("No active training runs. Ready for new tasks.")

        return "\n".join(lines)

    def create_heartbeat_skill(self) -> str:
        """Generate a skill file for the heartbeat loop."""
        return """---
title: "Heartbeat Training Monitor"
description: "Proactively monitors training runs, detects failures, and reports status"
---

# Heartbeat Monitor Skill

Every heartbeat tick, evaluate:

1. **Training Progress**: Are any experiments running? If so, check their latest metrics.
   - If loss is NaN or infinite, flag as error.
   - If loss is spiking, suggest reducing learning rate.
   - If training is complete, log results.

2. **Error Check**: Are there unresolved errors?
   - Attempt auto-fix for each unfixed error.
   - If fix succeeds, mark error as fixed.

3. **Resource Check**: Check GPU memory usage.
   - If CUDA OOM, suggest reducing batch size or using gradient checkpointing.

4. **Dataset Status**: Are there pending dataset downloads?
   - If a download failed, retry with alternative source.

5. **Report**: Log status to heartbeat memory.
   - If all OK: HEARTBEAT_OK
   - If issues: Report details
"""
