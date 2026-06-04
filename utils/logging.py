"""Structured JSON logging — machine-parseable logs with levels, context, and rotation."""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


_structured_handlers = []


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        if hasattr(record, "ctx"):
            entry["context"] = record.ctx
        return json.dumps(entry, default=str)


def setup_structured_logging(log_dir: Path = None, level: str = "INFO",
                             max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    log_dir = log_dir or Path("/content/epsionic_workspace/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "epsionic.jsonl",
        maxBytes=max_bytes, backupCount=backup_count,
    )
    handler.setFormatter(StructuredFormatter())
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    _structured_handlers.append(handler)
    return handler


def log_event(logger: logging.Logger, event: str, level: str = "INFO",
              ctx: dict = None):
    record = logger.makeRecord(
        logger.name, getattr(logging, level.upper(), logging.INFO),
        "", 0, event, (), None,
    )
    record.ctx = ctx or {}
    for h in _structured_handlers:
        if h.level <= record.levelno:
            h.emit(record)
