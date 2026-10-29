"""
Native Windows Virtual Input Dispatcher.
Implements programmatic mouse movements, clicks, text typing, and key combinations
using pure Win32 user32 APIs via ctypes.
"""

import time
import ctypes
from typing import List, Optional

user32 = ctypes.windll.user32


def enable_dpi_awareness():
    """Sets Per-Monitor DPI awareness on Windows so cursor coordinates match physical screenshot pixels."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# Enable DPI awareness immediately on import
enable_dpi_awareness()

# Mouse event constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

# Keyboard event constants
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Common Virtual Key Codes
VK_MAP = {
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "space": 0x20,
    "backspace": 0x08,
    "escape": 0x1B,
    "esc": 0x1B,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "delete": 0x2E,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
}


class VirtualInput:
    """
    Dispatches programmatic mouse and keyboard inputs to Windows.
    """

    @staticmethod
    def get_cursor_pos() -> tuple[int, int]:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    @staticmethod
    def set_cursor_pos(x: int, y: int):
        user32.SetCursorPos(int(x), int(y))

    @classmethod
    def mouse_click(cls, x: int, y: int, button: str = "left", double: bool = False):
        """Moves to (x, y) and performs a click or double click."""
        cls.set_cursor_pos(x, y)
        time.sleep(0.05)

        if button.lower() == "left":
            down_flag = MOUSEEVENTF_LEFTDOWN
            up_flag = MOUSEEVENTF_LEFTUP
        elif button.lower() == "right":
            down_flag = MOUSEEVENTF_RIGHTDOWN
            up_flag = MOUSEEVENTF_RIGHTUP
        else:
            down_flag = MOUSEEVENTF_LEFTDOWN
            up_flag = MOUSEEVENTF_LEFTUP

        user32.mouse_event(down_flag, 0, 0, 0, 0)
        time.sleep(0.04)
        user32.mouse_event(up_flag, 0, 0, 0, 0)

        if double:
            time.sleep(0.08)
            user32.mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(0.04)
            user32.mouse_event(up_flag, 0, 0, 0, 0)

    @classmethod
    def type_text(cls, text: str, key_delay_s: float = 0.02):
        """Types string characters using Windows Unicode key events."""
        for char in text:
            # Send character as Unicode
            user32.keybd_event(0, ord(char), KEYEVENTF_UNICODE, 0)
            time.sleep(key_delay_s)
            user32.keybd_event(0, ord(char), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0)
            time.sleep(key_delay_s)

    @classmethod
    def send_key_combo(cls, keys: List[str]):
        """Dispatches a key combination (e.g. ['ctrl', 's'])."""
        vk_codes = []
        for k in keys:
            k_lower = k.lower().strip()
            if k_lower in VK_MAP:
                vk_codes.append(VK_MAP[k_lower])
            elif len(k_lower) == 1:
                vk_codes.append(user32.VkKeyScanW(ord(k_lower)) & 0xFF)

        # Press down in order
        for vk in vk_codes:
            user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)

        time.sleep(0.05)

        # Release in reverse order
        for vk in reversed(vk_codes):
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.02)

