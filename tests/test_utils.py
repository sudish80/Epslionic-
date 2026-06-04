"""Tests for utility modules: device, colab, format."""

import pytest


class TestDevice:
    """Device manager tests."""

    def test_device_creation(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        assert dm.backend in ("cpu", "cuda", "mps")
        assert dm.name is not None

    def test_recommended_batch(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        batch = dm.recommended_batch_size()
        assert isinstance(batch, int)
        assert batch >= 1

    def test_summary(self):
        from epsionic.utils.device import DeviceManager
        dm = DeviceManager()
        summary = dm.summary()
        assert "Device" in summary
        assert "Backend" in summary


class TestFormat:
    """Format utilities: safe_print, strip_emoji."""

    def test_strip_emoji(self):
        from epsionic.utils.format import strip_emoji
        result = strip_emoji("Hello \U0001F916 world")
        assert "Hello" in result
        assert "world" in result

    def test_strip_emoji_no_change(self):
        from epsionic.utils.format import strip_emoji
        result = strip_emoji("plain text")
        assert result == "plain text"

    def test_set_emoji_mode(self):
        import epsionic.utils.format as fmt
        fmt.set_emoji_mode(True)
        assert fmt._STRIP_EMOJI is True
        fmt.set_emoji_mode(False)
        assert fmt._STRIP_EMOJI is False

    def test_emoji_pattern_compiles(self):
        from epsionic.utils.format import EMOJI_PATTERN
        assert EMOJI_PATTERN is not None


class TestColab:
    """Colab utility tests (all no-ops outside Colab)."""

    def test_mount_drive_noop(self):
        from epsionic.utils.colab import mount_drive
        result = mount_drive()
        assert result is False

    def test_get_secret_fallback(self):
        from epsionic.utils.colab import get_secret
        result = get_secret("NONEXISTENT_KEY")
        assert result == ""
