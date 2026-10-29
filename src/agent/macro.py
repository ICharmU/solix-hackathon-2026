"""
Human-Readable Desktop Macro Engine (Inspired by World of Warcraft Macros).

A simple, transparent, line-by-line DSL for automating desktop tasks across programs.
Both humans and AI agents can create, inspect, edit, and refine these macros in plain text.

Macro Syntax:
    # Comments start with #
    /click "Terminal"               # Left click target element by OCR text
    /doubleclick "Item"             # Double click target
    /rightclick "src"               # Right click target
    /hover "Menu"                   # Move mouse to target
    /type "npm run build"           # Type text string
    /press enter                    # Press key or combo (e.g. ctrl+s, alt+f4)
    /wait 1.0s                      # Wait for duration (e.g. 0.5s, 500ms)
    /assert "Build succeeded"       # Verify text is visible before proceeding

Usage:
    # Run a macro (waits for Ctrl + T + H activation by default):
    pixi run macro run macros/deploy.macro

    # Run immediately without hotkey:
    pixi run macro run macros/deploy.macro --now

    # Dry-run test (verifies all targets exist on screen without clicking):
    pixi run macro run macros/deploy.macro --dry-run
"""

import os
import sys
import time
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import ImageGrab

from src.agent.ocr_engine import WindowsOCREngine
from src.agent.safety import SafetyMonitor, UserInputHaltException
from src.agent.virtual_input import VirtualInput, enable_dpi_awareness
from src.agent.schemas import OCRTargetAnchor
from src.agent.cli import wait_for_hotkey_activation
from src.config import BASE_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("macro_engine")

MACROS_DIR = BASE_DIR / "macros"
CACHE_DIR = BASE_DIR / "data" / "cache"


class MacroStep:
    """Represents a single parsed line of a macro."""

    def __init__(self, line_num: int, command: str, argument: str, raw_line: str):
        self.line_num = line_num
        self.command = command.lower().strip()
        self.argument = argument.strip()
        self.raw_line = raw_line

    def __repr__(self):
        return f"<MacroStep L{self.line_num}: /{self.command} {self.argument}>"


class Macro:
    """A collection of MacroSteps loaded from a plain-text macro script."""

    def __init__(self, name: str, steps: List[MacroStep], file_path: Optional[Path] = None):
        self.name = name
        self.steps = steps
        self.file_path = file_path

    @classmethod
    def from_text(cls, text: str, name: str = "custom_macro") -> "Macro":
        steps = []
        for line_num, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if not stripped.startswith("/"):
                # Treat bare commands as /command
                stripped = "/" + stripped

            # Match /command "argument" or /command argument
            match = re.match(r"^/([a-zA-Z_]+)\s*(.*)$", stripped)
            if match:
                cmd = match.group(1).lower()
                arg = match.group(2).strip()

                # Strip surrounding quotes if present
                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                    arg = arg[1:-1]

                steps.append(MacroStep(line_num=line_num, command=cmd, argument=arg, raw_line=line))
            else:
                logger.warning(f"Line {line_num}: Could not parse macro line: '{line}'")

        return cls(name=name, steps=steps)

    @classmethod
    def from_file(cls, path: Path) -> "Macro":
        if not path.exists():
            raise FileNotFoundError(f"Macro file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return cls.from_text(content, name=path.stem)

    def to_text(self) -> str:
        lines = [f"# Macro: {self.name}"]
        for step in self.steps:
            if step.command in ("click", "doubleclick", "rightclick", "hover", "assert", "type"):
                lines.append(f"/{step.command} \"{step.argument}\"")
            else:
                lines.append(f"/{step.command} {step.argument}")
        return "\n".join(lines) + "\n"

    def save(self, path: Optional[Path] = None):
        target = path or self.file_path
        if not target:
            MACROS_DIR.mkdir(parents=True, exist_ok=True)
            target = MACROS_DIR / f"{self.name}.macro"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_text(), encoding="utf-8")
        self.file_path = target
        logger.info(f"Saved macro to {target}")


class MacroRunner:
    """
    Executes macros step by step with live OCR targeting,
    zero-tolerance safety monitoring, and detailed failure diagnostics.
    """

    def __init__(self, ocr_engine: Optional[WindowsOCREngine] = None, safety_monitor: Optional[SafetyMonitor] = None):
        self.ocr_engine = ocr_engine or WindowsOCREngine()
        self.safety_monitor = safety_monitor or SafetyMonitor(enable_dialog=True)
        enable_dpi_awareness()

    def run(
        self,
        macro: Macro,
        use_hotkey: bool = True,
        dry_run: bool = False,
        hotkey_timeout_s: float = None,
    ) -> bool:
        """
        Executes the macro.
        Returns True if all steps completed, False if any step failed or was halted.
        """
        print("\n" + "=" * 65)
        print(f"  MACRO: {macro.name} ({len(macro.steps)} steps)")
        print("=" * 65)

        for s in macro.steps:
            print(f"  L{s.line_num:>2}: /{s.command:<12} {s.argument}")
        print("=" * 65)

        if use_hotkey:
            activated = wait_for_hotkey_activation(timeout_s=hotkey_timeout_s)
            if not activated:
                return False

        if dry_run:
            print("\n[DRY RUN MODE] Simulating execution without physical clicks.")

        self.safety_monitor.arm()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            for idx, step in enumerate(macro.steps, start=1):
                self.safety_monitor.check_or_raise()

                print(f"\n[Step {idx}/{len(macro.steps)}] Line {step.line_num}: /{step.command} '{step.argument}'")
                success = self._execute_step(step, dry_run=dry_run)
                if not success:
                    print(f"\n[MACRO FAILED] Execution halted at line {step.line_num}: {step.raw_line}")
                    print("                Tip: You can edit the macro text file or ask the agent to refine it.")
                    return False

                # Settle buffer
                time.sleep(0.15)
                self.safety_monitor.check_or_raise()

            print(f"\n[SUCCESS] Macro '{macro.name}' completed all {len(macro.steps)} steps successfully!\n")
            return True

        except UserInputHaltException as e:
            print(f"\n[ABORTED] {e}")
            return False
        finally:
            self.safety_monitor.disarm()

    def _execute_step(self, step: MacroStep, dry_run: bool = False) -> bool:
        """Dispatches an individual macro step."""
        cmd = step.command
        arg = step.argument

        if cmd in ("click", "doubleclick", "rightclick", "hover"):
            # 1. Grab current screen
            screenshot_path = CACHE_DIR / "macro_live_screen.jpg"
            screen = ImageGrab.grab()
            screen.save(screenshot_path, format="JPEG", quality=85)

            # 2. Run OCR to find target text
            self.safety_monitor.check_or_raise()
            boxes = self.ocr_engine.recognize_image(screenshot_path)
            self.safety_monitor.check_or_raise()

            anchor = OCRTargetAnchor(
                target_text=arg,
                exact_match=False,
                bounding_box=(0, 0, 10, 10),
            )
            coords = self.ocr_engine.locate_target(anchor, boxes)

            if not coords:
                print(f"      [ERROR] Could not find text '{arg}' on the current screen.")
                return False

            target_x, target_y = coords

            # Normalize coordinates under display scaling
            import ctypes
            sys_w = ctypes.windll.user32.GetSystemMetrics(0)
            sys_h = ctypes.windll.user32.GetSystemMetrics(1)
            if screen.width != sys_w and screen.width > 0:
                target_x = int(target_x * (sys_w / screen.width))
                target_y = int(target_y * (sys_h / screen.height))

            print(f"      Found '{arg}' at ({target_x}, {target_y})")

            if not dry_run:
                self.safety_monitor.pause_for_agent_action()
                if cmd == "click":
                    VirtualInput.mouse_click(target_x, target_y, button="left", double=False)
                elif cmd == "doubleclick":
                    VirtualInput.mouse_click(target_x, target_y, button="left", double=True)
                elif cmd == "rightclick":
                    VirtualInput.mouse_click(target_x, target_y, button="right", double=False)
                elif cmd == "hover":
                    VirtualInput.set_cursor_pos(target_x, target_y)
                time.sleep(0.05)
                self.safety_monitor.resume_after_agent_action()

            return True

        elif cmd == "type":
            print(f"      Typing: '{arg}'")
            if not dry_run:
                self.safety_monitor.pause_for_agent_action()
                VirtualInput.type_text(arg)
                self.safety_monitor.resume_after_agent_action()
            return True

        elif cmd == "press":
            keys = [k.strip() for k in arg.split("+")]
            print(f"      Pressing keys: {keys}")
            if not dry_run:
                self.safety_monitor.pause_for_agent_action()
                VirtualInput.send_key_combo(keys)
                self.safety_monitor.resume_after_agent_action()
            return True

        elif cmd == "wait":
            duration = self._parse_duration(arg)
            print(f"      Waiting {duration:.2f}s...")
            time.sleep(duration)
            return True

        elif cmd == "assert":
            # Verifies that text is currently visible on screen
            screenshot_path = CACHE_DIR / "macro_live_screen.jpg"
            screen = ImageGrab.grab()
            screen.save(screenshot_path, format="JPEG", quality=85)

            boxes = self.ocr_engine.recognize_image(screenshot_path)
            anchor = OCRTargetAnchor(target_text=arg, exact_match=False, bounding_box=(0, 0, 10, 10))
            coords = self.ocr_engine.locate_target(anchor, boxes)
            if coords:
                print(f"      [ASSERT PASSED] Verified '{arg}' is present at ({coords[0]}, {coords[1]}).")
                return True
            else:
                print(f"      [ASSERT FAILED] Text '{arg}' was not found on screen.")
                return False

        else:
            print(f"      [ERROR] Unknown macro command: '/{cmd}'")
            return False

    @staticmethod
    def _parse_duration(arg: str) -> float:
        """Parses duration strings like '1.5s', '500ms', '2'."""
        arg = arg.strip().lower()
        if arg.endswith("ms"):
            return float(arg[:-2]) / 1000.0
        elif arg.endswith("s"):
            return float(arg[:-1])
        else:
            try:
                return float(arg)
            except ValueError:
                return 0.5


def main():
    parser = argparse.ArgumentParser(description="Desktop Macro Engine (WoW-style Macro Automation).")
    subparsers = parser.add_subparsers(dest="subcommand", help="Macro command: run, new, list")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a macro file")
    run_parser.add_argument("macro_path", type=str, help="Path to .macro file")
    run_parser.add_argument("--now", action="store_true", help="Execute immediately without waiting for Ctrl+T+H")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate without dispatching physical inputs")

    # 'new' subcommand
    new_parser = subparsers.add_parser("new", help="Create a sample template macro")
    new_parser.add_argument("macro_name", type=str, help="Name for the new macro")

    # 'ask' / 'create' subcommand
    ask_parser = subparsers.add_parser("ask", help="Formulate a new macro from a natural language query")
    ask_parser.add_argument("query", type=str, help="Natural language description of what the macro should do")
    ask_parser.add_argument("--name", "-n", type=str, default=None, help="Custom name for the macro file")
    ask_parser.add_argument("--run", action="store_true", help="Immediately run the macro after generating it")
    ask_parser.add_argument("--now", action="store_true", help="When running, execute immediately without waiting for Ctrl+T+H")
    ask_parser.add_argument("--dry-run", action="store_true", help="Simulate without dispatching physical clicks")

    # 'list' subcommand
    subparsers.add_parser("list", help="List all available macros in macros/")

    args = parser.parse_args()

    if args.subcommand == "run":
        p = Path(args.macro_path)
        if not p.exists() and not p.suffix:
            p = MACROS_DIR / f"{args.macro_path}.macro"

        if not p.exists():
            print(f"[ERROR] Macro not found: {p}")
            sys.exit(1)

        macro = Macro.from_file(p)
        runner = MacroRunner()
        success = runner.run(macro, use_hotkey=(not args.now), dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    elif args.subcommand == "new":
        MACROS_DIR.mkdir(parents=True, exist_ok=True)
        target = MACROS_DIR / f"{args.macro_name}.macro"
        sample_content = (
            f"# Macro: {args.macro_name}\n"
            "# Line-by-line desktop actions (World of Warcraft style)\n\n"
            "/click \"Terminal\"\n"
            "/type \"git status\"\n"
            "/press enter\n"
            "/wait 0.5s\n"
        )
        target.write_text(sample_content, encoding="utf-8")
        print(f"[CREATED] New macro template created at: {target}")
        print("\nYou can edit this file in any text editor, then run it with:")
        print(f"  pixi run macro run {target}")

    elif args.subcommand in ("ask", "create"):
        from src.agent.macro_builder import MacroFormulator
        formulator = MacroFormulator()
        print(f"\n[AI FORMULATOR] Analyzing query: '{args.query}'...")
        print("                Scanning live screen for active UI anchors...")
        macro = formulator.formulate_macro(query=args.query, name=args.name, inspect_live_screen=True)
        macro.save()
        print(f"\n[FORMULATED] Macro saved to: {macro.file_path}\n")
        print("=" * 60)
        print(macro.to_text())
        print("=" * 60)

        if args.run:
            runner = MacroRunner()
            success = runner.run(macro, use_hotkey=(not args.now), dry_run=args.dry_run)
            sys.exit(0 if success else 1)
        else:
            print("\nTo run this macro:")
            print(f"  pixi run macro run {macro.file_path.name}")
            print("\nYou can also open the file to tweak any line or timing.\n")

    elif args.subcommand == "list":
        MACROS_DIR.mkdir(parents=True, exist_ok=True)
        files = list(MACROS_DIR.glob("*.macro"))
        print(f"\nAvailable Macros in {MACROS_DIR}:")
        if not files:
            print("  (No macros found. Create one with: pixi run macro new my_macro)")
        for f in files:
            print(f"  - {f.name}")
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

