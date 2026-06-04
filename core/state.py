"""State machine with valid transitions.
Pattern from Home-Assistant (86k stars): strict state machine."""

from typing import Dict, Set, List, Optional


VALID_TRANSITIONS: Dict[str, Set[str]] = {
    "idle": {"running", "stopped"},
    "running": {"idle", "stopped", "paused"},
    "paused": {"running", "idle", "stopped"},
    "stopped": {"idle"},
    "error": {"idle", "stopped"},
}


class StateTransitionError(Exception):
    """Raised on invalid state transition."""


class StateMachine:
    """Lightweight state machine with transition validation."""

    def __init__(self, initial: str = "idle"):
        self._state = initial
        self._history: List[str] = [initial]

    @property
    def state(self) -> str:
        return self._state

    @property
    def history(self) -> List[str]:
        return list(self._history)

    def transition(self, target: str) -> str:
        if target == self._state:
            return self._state
        valid = VALID_TRANSITIONS.get(self._state, set())
        if target not in valid:
            allowed = ", ".join(sorted(valid))
            raise StateTransitionError(
                f"Cannot transition from '{self._state}' to '{target}'. "
                f"Allowed: [{allowed}]"
            )
        self._history.append(target)
        self._state = target
        return target

    def can_transition(self, target: str) -> bool:
        valid = VALID_TRANSITIONS.get(self._state, set())
        return target in valid

    def reset(self):
        self._state = "idle"
        self._history = ["idle"]

    def status(self) -> str:
        return f"State: {self._state} (history: {' -> '.join(self._history)})"
