"""Colab utilities: Drive mount, secrets, and environment helpers."""

import os
import logging

logger = logging.getLogger("epsionic.colab")


def mount_drive():
    """Mount Google Drive in Colab. No-op elsewhere."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        logger.info("Google Drive mounted at /content/drive")
        return True
    except ImportError:
        logger.debug("google.colab not available (not in Colab)")
        return False
    except Exception as e:
        logger.warning(f"Drive mount failed: {e}")
        return False


def get_secret(key: str, default: str = "") -> str:
    """Get a Colab secret or env var. Falls back to os.environ."""
    try:
        from google.colab import userdata
        return userdata.get(key) or os.environ.get(key, default)
    except ImportError:
        return os.environ.get(key, default)


def save_checkpoint(local_path: str, drive_path: str = None):
    """Save a checkpoint to Google Drive."""
    import shutil
    from pathlib import Path
    if drive_path is None:
        drive_path = f"/content/drive/MyDrive/epsionic_checkpoints/{Path(local_path).name}"
    try:
        Path(drive_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(local_path).is_dir():
            shutil.copytree(local_path, drive_path, dirs_exist_ok=True)
        else:
            shutil.copy2(local_path, drive_path)
        logger.info(f"Checkpoint saved to {drive_path}")
        return drive_path
    except Exception as e:
        logger.warning(f"Checkpoint save failed: {e}")
        return None


def load_checkpoint(drive_path: str, local_path: str = None):
    """Load a checkpoint from Google Drive."""
    import shutil
    from pathlib import Path
    if local_path is None:
        local_path = f"/content/{Path(drive_path).name}"
    try:
        if Path(drive_path).is_dir():
            shutil.copytree(drive_path, local_path, dirs_exist_ok=True)
        else:
            shutil.copy2(drive_path, local_path)
        logger.info(f"Checkpoint loaded from {drive_path}")
        return local_path
    except Exception as e:
        logger.warning(f"Checkpoint load failed: {e}")
        return None
