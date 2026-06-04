"""Security utilities — input sanitization, validation, safe subprocess helpers."""

import os
import re
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Any

# ── Regex patterns ──────────────────────────────────────────────────────────

_REPO_URL_PATTERN = re.compile(
    r"^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?$"
)
_PYPI_PACKAGE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")
_PLUGIN_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DOMAIN_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
_FLOW_ID_PATTERN = re.compile(r"^flow_\d{8}_\d{6}_\d{6}$")
_NODE_ID_PATTERN = re.compile(r"^node_\d+$")
_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+_\d{8}_\d{6}$")
_SAFE_PATH_PATTERN = re.compile(r"^[\w./\\:\-@]+$")
_HOSTNAME_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$")
_PORT_PATTERN = re.compile(r"^\d{1,5}$")


def sanitize_github_url(url: str) -> Optional[str]:
    """Validate and return a GitHub repo URL, or None if invalid."""
    if not isinstance(url, str) or len(url) > 200:
        return None
    url = url.strip()
    if _REPO_URL_PATTERN.match(url):
        return url
    return None


def sanitize_pypi_package(name: str) -> Optional[str]:
    """Validate a PyPI package name."""
    if not isinstance(name, str) or len(name) > 200:
        return None
    name = name.strip()
    if _PYPI_PACKAGE_PATTERN.match(name):
        return name
    return None


def sanitize_plugin_name(name: str) -> Optional[str]:
    """Validate a plugin name (alphanumeric + underscore only)."""
    if not isinstance(name, str) or len(name) > 100:
        return None
    if _PLUGIN_NAME_PATTERN.match(name):
        return name
    return None


def sanitize_filename_component(name: str) -> Optional[str]:
    """Sanitize a filename component to prevent path traversal."""
    if not isinstance(name, str) or not name:
        return None
    # Remove any path separators and directory traversal
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name)
    cleaned = cleaned.replace("..", "")
    if not cleaned or len(cleaned) > 255:
        return None
    return cleaned


def validate_path_within(base_dir: Path, user_path: str) -> Optional[Path]:
    """Ensure a user-supplied path is within the base directory (prevents path traversal)."""
    if not isinstance(user_path, str) or len(user_path) > 4096:
        return None
    resolved = base_dir.resolve()
    try:
        target = (resolved / user_path).resolve()
        target.relative_to(resolved)
        return target
    except (ValueError, RuntimeError, OSError):
        return None


def safe_subprocess(args: List[str], timeout: int = 60, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess safely with a timeout and restricted shell access."""
    # Validate every arg is a plain string (no injection via objects)
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError(f"Unsafe subprocess argument type: {type(arg).__name__}")
    # Ensure we never use shell=True
    kwargs.pop("shell", None)
    return subprocess.run(args, timeout=timeout, shell=False, **kwargs)


def validate_port(port: Any) -> Optional[int]:
    """Validate a network port number (1-65535)."""
    try:
        p = int(port)
        if 1 <= p <= 65535:
            return p
        return None
    except (ValueError, TypeError):
        return None


def validate_host(host: str) -> Optional[str]:
    """Validate a hostname or IP."""
    if not isinstance(host, str) or len(host) > 255:
        return None
    # Allow 0.0.0.0, localhost, and valid hostnames
    if host in ("0.0.0.0", "127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:0"):
        return host
    if _HOSTNAME_PATTERN.match(host):
        return host
    return None


def sanitize_json_string(data: Any, max_depth: int = 10) -> Any:
    """Recursively sanitize a JSON-serializable structure to remove dangerous patterns."""
    if max_depth <= 0:
        return None
    if isinstance(data, str):
        # Strip null bytes and control characters (except tab/newline)
        cleaned = "".join(c for c in data if c >= " " or c in "\t\n")
        # Truncate extremely long strings
        if len(cleaned) > 100000:
            cleaned = cleaned[:100000]
        return cleaned
    if isinstance(data, dict):
        return {k: sanitize_json_string(v, max_depth - 1) for k, v in data.items()
                if isinstance(k, str) and len(k) < 1000}
    if isinstance(data, list):
        return [sanitize_json_string(v, max_depth - 1) for v in data[:1000]]
    if isinstance(data, (int, float, bool)):
        return data
    if data is None:
        return None
    return str(data)[:10000]


def is_safe_path(path_str: str) -> bool:
    """Check if a path string is safe (no injection characters)."""
    if not isinstance(path_str, str) or len(path_str) > 4096:
        return False
    path_str = path_str.strip()
    # Reject shell metacharacters
    dangerous = set("`$|;&><(){}[]'\"!#~")
    if any(c in path_str for c in dangerous):
        return False
    return bool(_SAFE_PATH_PATTERN.match(path_str))
