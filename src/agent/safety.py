"""
Zero-Tolerance Safety Monitor for Autonomous State Replication.
Monitors physical hardware input (keyboard, mouse).
If any physical user interaction is detected while the agent is active,
immediately halts execution and displays: 'user input received. program halting'.
"""

import time
import ctypes
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger("safety_monitor")


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class UserInputHaltException(Exception):
    """Raised when physical user input is detected during autonomous execution."""
    pass


class SafetyMonitor:
    """
    Continuous background hardware monitor using native Win32 APIs.
    Detects any user keystroke or mouse movement and triggers an immediate halt.
    """

    def __init__(
        self,
        mouse_threshold_px: int = 6,
        poll_interval_s: float = 0.01,
        enable_dialog: bool = False,
        on_halt_callback: Optional[Callable[[str], None]] = None,
    ):
        self.mouse_threshold_px = mouse_threshold_px
        self.poll_interval_s = poll_interval_s
        self.enable_dialog = enable_dialog
        self.on_halt_callback = on_halt_callback

        self._user32 = ctypes.windll.user32
        self._armed = False
        self._halted = False
        self._halt_reason = ""
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._baseline_x = 0
        self._baseline_y = 0

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason

    def arm(self):
        """Arms the monitor, capturing the current baseline mouse position."""
        with self._lock:
            self._halted = False
            self._halt_reason = ""
            self._armed = True

            pt = POINT()
            self._user32.GetCursorPos(ctypes.byref(pt))
            self._baseline_x = pt.x
            self._baseline_y = pt.y

        # Clear any residual keystroke state in Win32 message queue
        for vkey in range(1, 256):
            self._user32.GetAsyncKeyState(vkey)

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

        logger.info("SafetyMonitor armed. Zero-tolerance hardware halt active.")

    def disarm(self):
        """Disarms the safety monitor."""
        with self._lock:
            self._armed = False
        logger.info("SafetyMonitor disarmed.")

    def update_agent_cursor(self, new_x: int, new_y: int):
        """
        Updates baseline coordinates when the AGENT moves the mouse,
        preventing the agent's own programmatic movements from triggering false halts.
        """
        with self._lock:
            self._baseline_x = new_x
            self._baseline_y = new_y

    def trigger_halt(self, reason: str = "user input received. program halting"):
        """Forces an immediate halt and displays notification."""
        with self._lock:
            if not self._halted:
                self._halted = True
                self._halt_reason = reason
                self._armed = False

        self.display_notification(reason)

        if self.on_halt_callback:
            try:
                self.on_halt_callback(reason)
            except Exception as e:
                logger.error(f"Error executing on_halt_callback: {e}")

    def check_or_raise(self):
        """Raises UserInputHaltException if user input was detected."""
        with self._lock:
            if self._halted:
                raise UserInputHaltException(self._halt_reason or "user input received. program halting")

    def display_notification(self, message: str = "user input received. program halting"):
        """Displays the required user input halt notification."""
        banner = "=" * 60
        print(f"\n{banner}\n[SAFETY HALT] {message}\n{banner}\n", flush=True)

        if self.enable_dialog:
            # Non-blocking Windows Notification popup in background thread
            def _popup():
                try:
                    # MB_OK | MB_ICONWARNING | MB_TOPMOST | MB_SETFOREGROUND
                    MB_TOPMOST = 0x00040000
                    MB_ICONWARNING = 0x00000030
                    ctypes.windll.user32.MessageBoxW(
                        0,
                        message,
                        "Safety Alert",
                        MB_TOPMOST | MB_ICONWARNING,
                    )
                except Exception:
                    pass

            threading.Thread(target=_popup, daemon=True).start()

    def _monitor_loop(self):
        """High-frequency polling loop for physical inputs."""
        pt = POINT()
        while True:
            with self._lock:
                if not self._armed:
                    break

            # 1. Check mouse displacement
            if self._user32.GetCursorPos(ctypes.byref(pt)):
                with self._lock:
                    dx = abs(pt.x - self._baseline_x)
                    dy = abs(pt.y - self._baseline_y)
                    if dx > self.mouse_threshold_px or dy > self.mouse_threshold_px:
                        logger.warning(f"Physical mouse movement detected: delta=({dx}, {dy})")
                        self.trigger_halt("user input received. program halting")
                        break

            # 2. Check all physical keyboard virtual keys (0x01 to 0xFE)
            # Skip left/right click here since mouse movement handles it, but check keyboard keys
            # VK_CANCEL=0x03 to VK_OEM_CLEAR=0xFE
            for vkey in range(0x08, 0xFF):
                state = self._user32.GetAsyncKeyState(vkey)
                if state & 0x8000:
                    logger.warning(f"Physical keystroke detected: VirtualKey=0x{vkey:02X}")
                    self.trigger_halt("user input received. program halting")
                    break

            with self._lock:
                if self._halted or not self._armed:
                    break

            time.sleep(self.poll_interval_s)
