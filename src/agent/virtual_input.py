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
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

# Extended keys on Windows (navigation cluster and special keys)
EXTENDED_KEYS = {
    0x21,  # PAGE UP
    0x22,  # PAGE DOWN
    0x23,  # END
    0x24,  # HOME
    0x25,  # LEFT
    0x26,  # UP
    0x27,  # RIGHT
    0x28,  # DOWN
    0x2D,  # INSERT
    0x2E,  # DELETE
    0x5B,  # LWIN
    0x5C,  # RWIN
}

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
    "windows": 0x5B,
    "cmd": 0x5B,
    "pageup": 0x21,
    "pgup": 0x21,
    "pagedown": 0x22,
    "pgdn": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "ins": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "capslock": 0x14,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


# Win32 SendInput Structures (64-bit and 32-bit compatible)
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT_UNION)]


user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint


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
        """Types string characters using Windows SendInput Unicode events."""
        for char in text:
            # Key down
            inp_down = INPUT()
            inp_down.type = INPUT_KEYBOARD
            inp_down.union.ki.wVk = 0
            inp_down.union.ki.wScan = ord(char)
            inp_down.union.ki.dwFlags = KEYEVENTF_UNICODE
            user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))

            time.sleep(key_delay_s)

            # Key up
            inp_up = INPUT()
            inp_up.type = INPUT_KEYBOARD
            inp_up.union.ki.wVk = 0
            inp_up.union.ki.wScan = ord(char)
            inp_up.union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))

            time.sleep(key_delay_s)

    @classmethod
    def send_key_combo(cls, keys: List[str]):
        """Dispatches a key combination (e.g. ['ctrl', 's'], ['end'], ['ctrl', 'g'])."""
        vk_codes = []
        for k in keys:
            k_lower = k.lower().strip()
            if k_lower in VK_MAP:
                vk_codes.append(VK_MAP[k_lower])
            elif len(k_lower) == 1:
                scan = user32.VkKeyScanW(ord(k_lower)) & 0xFF
                if scan != 0xFF:
                    vk_codes.append(scan)

        if not vk_codes:
            return

        # Press down in order
        for vk in vk_codes:
            flags = KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.union.ki.wVk = vk
            inp.union.ki.wScan = 0
            inp.union.ki.dwFlags = flags
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            time.sleep(0.02)

        time.sleep(0.05)

        # Release in reverse order
        for vk in reversed(vk_codes):
            flags = (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0) | KEYEVENTF_KEYUP
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.union.ki.wVk = vk
            inp.union.ki.wScan = 0
            inp.union.ki.dwFlags = flags
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            time.sleep(0.02)

