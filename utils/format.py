"""Formatting utilities: safe print, emoji stripping, rich formatting."""

import os
import re

_STRIP_EMOJI = False

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0000200D\U0000FE0F\u274C\u2B50\u26A0\u26A1"
    "\u2795\u2796\u2705\u274E\u3030\u2B55\u2757\u2753\u2049\u203C"
    "\U0001F000-\U0001FFFF\U00002000-\U00002BFF]+"
)


def strip_emoji(text: str) -> str:
    """Replace emoji with descriptive text in brackets."""
    def _replace(m):
        ch = m.group(0)
        name = {
            "\U0001F916": "[robot]", "\u2705": "[OK]", "\u274C": "[FAIL]",
            "\u26A0\uFE0F": "[warn]", "\u26A1": "[zap]", "\U0001F504": "[refresh]",
            "\U0001F680": "[rocket]", "\U0001F4CA": "[chart]", "\U0001F4C8": "[graph]",
            "\U0001F4E6": "[package]", "\U0001F493": "[heartbeat]", "\U0001F3F7\uFE0F": "[control]",
            "\U0001F9E0": "[brain]", "\U0001F916": "[ai]",
        }.get(ch, f"[emoji]")
        if len(ch) > 2:
            return f"[emoji]"
        return name
    return EMOJI_PATTERN.sub(_replace, text)


def safe_print(*args, **kwargs):
    """Print with emoji stripping and cp1252 fallback."""
    text = " ".join(str(a) for a in args)
    if _STRIP_EMOJI:
        text = strip_emoji(text)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        print(strip_emoji(text), **kwargs)


def set_emoji_mode(strip: bool = False):
    global _STRIP_EMOJI
    _STRIP_EMOJI = strip
