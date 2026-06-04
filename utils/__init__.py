__test__ = False

from .device import DeviceManager
from .colab import mount_drive, get_secret, save_checkpoint, load_checkpoint
from .format import safe_print, strip_emoji, set_emoji_mode