"""
Command-Line Action Agent for Autonomous Desktop Interaction.
Finds a UI element on screen by OCR text and performs the requested action (click, type, keypress)
at that exact location under the supervision of the Zero-Tolerance SafetyMonitor.

Examples:
    # Click 'Terminal' menu
    python -m src.agent.cli --find "Terminal" --action click

    # Double-click an item
    python -m src.agent.cli --find "solix_demo" --action double-click

    # Click a search bar, type text, and hit Enter
    python -m src.agent.cli --find "Search" --action click --type "main.py" --press enter

    # Dry-run test (locate target without dispatching physical click)
    python -m src.agent.cli --find "File" --dry-run
"""

import sys
import time
import ctypes
import argparse
import logging
from pathlib import Path
from PIL import ImageGrab

from src.agent.ocr_engine import WindowsOCREngine
from src.agent.safety import SafetyMonitor, UserInputHaltException
from src.agent.virtual_input import VirtualInput
from src.agent.schemas import OCRTargetAnchor
from src.config import BASE_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("agent_cli")

CACHE_DIR = BASE_DIR / "data" / "cache"


def wait_for_hotkey_activation(timeout_s: float = None) -> bool:
    """
    Waits for the global chord Ctrl + T + H to activate the agent.
    Allows the user to switch away from the terminal and set up their target application.
    """
    user32 = ctypes.windll.user32
    VK_CONTROL = 0x11
    VK_T = 0x54  # 'T'
    VK_H = 0x48  # 'H'

    print("\n" + "=" * 65)
    print("  [STANDBY MODE] Waiting for activation hotkey: Ctrl + T + H")
    print("=" * 65)
    print("  1. Switch to your desired target application or window.")
    print("  2. Set up your screen.")
    print("  3. Press [ Ctrl + T + H ] simultaneously to activate.")
    print("  (Any subsequent user input while active will halt execution)")
    print("=" * 65 + "\n", flush=True)

    start_t = time.time()
    while True:
        if timeout_s and (time.time() - start_t > timeout_s):
            print(f"\n[TIMEOUT] Activation hotkey wait timed out ({timeout_s}s).")
            return False

        ctrl = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
        t_key = bool(user32.GetAsyncKeyState(VK_T) & 0x8000)
        h_key = bool(user32.GetAsyncKeyState(VK_H) & 0x8000)

        if ctrl and t_key and h_key:
            print(">>> [ACTIVATED] Ctrl + T + H detected! Starting action execution... <<<\n", flush=True)
            # Wait for keys to be physically released before continuing
            while (
                (user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
                or (user32.GetAsyncKeyState(VK_T) & 0x8000)
                or (user32.GetAsyncKeyState(VK_H) & 0x8000)
            ):
                time.sleep(0.02)
            time.sleep(0.2)  # Settle delay so UI settles after key release
            return True

        time.sleep(0.02)


def execute_action(
    find_text: str,
    action: str = "click",
    type_payload: str = None,
    press_key: str = None,
    offset_x: int = 0,
    offset_y: int = 0,
    dry_run: bool = False,
    exact_match: bool = False,
    delay_s: float = 0.2,
    use_hotkey: bool = True,
    hotkey_timeout_s: float = None,
) -> bool:
    """
    Finds a UI target on the screen via local OCR and performs the action at that location.
    If use_hotkey is True, waits for Ctrl + T + H before capturing screen and executing.
    """
    if use_hotkey:
        activated = wait_for_hotkey_activation(timeout_s=hotkey_timeout_s)
        if not activated:
            return False

    from src.agent.virtual_input import enable_dpi_awareness
    enable_dpi_awareness()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_path = CACHE_DIR / "current_action_screen.jpg"

    print(f"\n[1/4] Capturing desktop screen...")
    screen = ImageGrab.grab()
    screen.save(screenshot_path, format="JPEG", quality=85)
    print(f"      Screen size: {screen.width}x{screen.height}")

    print(f"[2/4] Scanning screen with on-device OCR for target: '{find_text}'...")
    engine = WindowsOCREngine()
    start_t = time.time()
    boxes = engine.recognize_image(screenshot_path)
    ocr_time = time.time() - start_t
    print(f"      OCR completed in {ocr_time:.2f}s ({len(boxes)} text elements found).")

    anchor = OCRTargetAnchor(
        target_text=find_text,
        exact_match=exact_match,
        bounding_box=(0, 0, 10, 10),
        click_offset_from_center=(offset_x, offset_y),
    )

    coords = engine.locate_target(anchor, boxes)
    if not coords:
        print(f"\n[ERROR] Target text '{find_text}' was NOT found anywhere on the current screen.")
        print("        Hint: Check capitalization or try a shorter substring.")
        return False

    target_x, target_y = coords

    # Ensure coordinates match Windows system metrics under display scaling
    sys_w = ctypes.windll.user32.GetSystemMetrics(0)
    sys_h = ctypes.windll.user32.GetSystemMetrics(1)
    if screen.width != sys_w and screen.width > 0:
        scale_x = sys_w / screen.width
        scale_y = sys_h / screen.height
        target_x = int(target_x * scale_x)
        target_y = int(target_y * scale_y)

    print(f"\n[3/4] Target located successfully:")
    print(f"      Text:        '{find_text}'")
    print(f"      Coordinates: ({target_x}, {target_y})")
    print(f"      Action:      {action.upper()}")

    if dry_run:
        print("\n[DRY RUN] Target located! Note: physical clicks are skipped in dry-run mode.")
        print(f"          Remove '--dry-run' to actually execute the {action.upper()} at ({target_x}, {target_y}).")
        return True

    # Arm safety monitor
    safety = SafetyMonitor(enable_dialog=True)
    safety.arm()

    try:
        # Check safety before moving
        safety.check_or_raise()

        if delay_s > 0:
            time.sleep(delay_s)

        print(f"[4/4] Executing action '{action}' at ({target_x}, {target_y})...")

        safety.pause_for_agent_action()
        if action in ("click", "left-click"):
            VirtualInput.mouse_click(target_x, target_y, button="left", double=False)
        elif action in ("double-click", "double_click"):
            VirtualInput.mouse_click(target_x, target_y, button="left", double=True)
        elif action in ("right-click", "right_click"):
            VirtualInput.mouse_click(target_x, target_y, button="right", double=False)
        elif action in ("move", "hover"):
            VirtualInput.set_cursor_pos(target_x, target_y)
        else:
            safety.resume_after_agent_action()
            print(f"[ERROR] Unknown action '{action}'.")
            return False
        time.sleep(0.05)
        safety.resume_after_agent_action()

        # If user requested typing
        if type_payload:
            time.sleep(0.15)
            safety.check_or_raise()
            print(f"      Typing text payload: '{type_payload}'")
            safety.pause_for_agent_action()
            VirtualInput.type_text(type_payload)
            safety.resume_after_agent_action()

        # If user requested a keypress / combo
        if press_key:
            time.sleep(0.15)
            safety.check_or_raise()
            keys = [k.strip() for k in press_key.split("+")]
            print(f"      Pressing key combo: {keys}")
            safety.pause_for_agent_action()
            VirtualInput.send_key_combo(keys)
            safety.resume_after_agent_action()

        time.sleep(0.1)
        safety.check_or_raise()

        print(f"\n[SUCCESS] Action '{action}' on '{find_text}' executed successfully at ({target_x}, {target_y})!\n")
        return True

    except UserInputHaltException as e:
        print(f"\n[ABORTED] {e}")
        return False
    finally:
        safety.disarm()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Action Agent: Locates a UI element via OCR and executes an action at that location."
    )
    parser.add_argument(
        "--find",
        "-f",
        type=str,
        required=True,
        help="Text string to find on the live screen (e.g. 'Terminal', 'File', 'Submit')",
    )
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        choices=["click", "double-click", "right-click", "move", "hover"],
        default="click",
        help="Action to perform at target location (default: click)",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default=None,
        help="Text payload to type after clicking target (optional)",
    )
    parser.add_argument(
        "--press",
        "-p",
        type=str,
        default=None,
        help="Key or combo to press after action (e.g. 'enter', 'ctrl+s', 'esc')",
    )
    parser.add_argument(
        "--offset-x",
        type=int,
        default=0,
        help="Horizontal offset in pixels from target center (default: 0)",
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=0,
        help="Vertical offset in pixels from target center (default: 0)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require exact case-sensitive token match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Locate target and verify coordinates without dispatching physical clicks",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds before dispatching action (default: 0.2)",
    )
    parser.add_argument(
        "--now",
        action="store_true",
        help="Execute immediately without waiting for activation hotkey Ctrl + T + H",
    )
    parser.add_argument(
        "--hotkey-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds to wait for activation hotkey (default: unlimited)",
    )

    args = parser.parse_args()

    success = execute_action(
        find_text=args.find,
        action=args.action,
        type_payload=args.type,
        press_key=args.press,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        dry_run=args.dry_run,
        exact_match=args.exact,
        delay_s=args.delay,
        use_hotkey=(not args.now),
        hotkey_timeout_s=args.hotkey_timeout,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
