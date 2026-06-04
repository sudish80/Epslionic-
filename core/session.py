"""
Session Manager - OpenClaw-inspired per-session isolation with lane queues.
Each training run gets its own session with serial execution within the session.
"""

import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger("openclaw.session")


class SessionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Session:
    id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    name: str = ""
    status: SessionStatus = SessionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    lane: "Lane" = None

    _task_queue: deque = field(default_factory=deque)
    _current_task: Optional[Callable] = None

    def enqueue(self, task: Callable, task_name: str = ""):
        self._task_queue.append((task, task_name))

    @property
    def queue_size(self) -> int:
        return len(self._task_queue)


class Lane:
    """A serial execution lane - OpenClaw's lane queue pattern."""

    def __init__(self, lane_id: str):
        self.id = lane_id
        self._queue = deque()
        self._current = None
        self._lock = threading.Lock()

    def submit(self, session_id: str, task: Callable, task_name: str = ""):
        with self._lock:
            self._queue.append((session_id, task, task_name))

    def process_next(self) -> bool:
        with self._lock:
            if not self._queue:
                return False
            session_id, task, task_name = self._queue.popleft()
            self._current = (session_id, task, task_name)

        try:
            logger.info(f"[Lane {self.id}] Executing: {task_name}")
            result = task()
            logger.info(f"[Lane {self.id}] Completed: {task_name}")
            return True
        except Exception as e:
            logger.error(f"[Lane {self.id}] Failed: {task_name} - {e}")
            return False
        finally:
            with self._lock:
                self._current = None

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current is not None


class SessionManager:
    """Manages multiple training sessions with lane queue isolation."""

    def __init__(self, max_concurrent: int = 2):
        self._sessions: Dict[str, Session] = {}
        self._lanes: Dict[str, Lane] = {}
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def create_session(self, name: str = "", config: dict = None) -> Session:
        lane_id = self._assign_lane()
        session = Session(name=name or f"session_{len(self._sessions)+1}",
                          config=config or {},
                          lane=self._lanes[lane_id])
        with self._lock:
            self._sessions[session.id] = session
        logger.info(f"Created session {session.id} on lane {lane_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def list_sessions(self, status: Optional[SessionStatus] = None) -> list:
        if status:
            return [s for s in self._sessions.values() if s.status == status]
        return list(self._sessions.values())

    def submit_task(self, session_id: str, task: Callable, task_name: str = ""):
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        lane = session.lane
        lane.submit(session_id, task, task_name)
        session.enqueue(task, task_name)
        self._ensure_worker()

    def cancel_session(self, session_id: str):
        session = self.get_session(session_id)
        if session:
            session.status = SessionStatus.CANCELLED

    def get_status_markdown(self) -> str:
        lines = ["## Session Status", ""]
        lines.append(f"| Session ID | Name | Status | Queue | Lane |")
        lines.append(f"|---|---|---|---|---|")
        for s in self._sessions.values():
            lines.append(
                f"| {s.id[:12]} | {s.name} | {s.status.value} | "
                f"{s.queue_size} | {s.lane.id if s.lane else 'none'} |"
            )
        return "\n".join(lines)

    def _assign_lane(self) -> str:
        with self._lock:
            available_lanes = [f"lane_{i}" for i in range(self._max_concurrent)]
            for lane_id in available_lanes:
                if lane_id not in self._lanes:
                    self._lanes[lane_id] = Lane(lane_id)
                    return lane_id
            min_lane = min(self._lanes.values(), key=lambda l: l.queue_length)
            return min_lane.id

    def _ensure_worker(self):
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        while self._running:
            any_work = False
            for lane in self._lanes.values():
                if not lane.is_busy and lane.process_next():
                    any_work = True
            if not any_work:
                time.sleep(0.5)
            if all(len(s._task_queue) == 0 for s in self._sessions.values()):
                time.sleep(1)

    def shutdown(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
