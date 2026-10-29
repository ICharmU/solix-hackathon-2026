"""
Interactive Live Demo for Autonomous State Replication Agent.

Usage:
    # 1. Test the Zero-Tolerance Hardware Safety Interruption:
    .pixi\\envs\\default\\python.exe -m src.agent.demo --mode safety

    # 2. Test Live Screen OCR Element Grounding:
    .pixi\\envs\\default\\python.exe -m src.agent.demo --mode ocr

    # 3. Test Full Autonomous State Replicator:
    .pixi\\envs\\default\\python.exe -m src.agent.demo --mode replicate
"""

import sys
import time
import argparse
from pathlib import Path
from PIL import ImageGrab

from src.agent.ocr_engine import WindowsOCREngine
from src.agent.safety import SafetyMonitor
from src.agent.virtual_input import VirtualInput
from src.agent.replicator import TaskReplicator
from src.agent.schemas import TaskTrajectory, ReplicatedAction, OCRTargetAnchor
from src.config import BASE_DIR


def demo_safety_monitor():
    print("\n" + "=" * 65)
    print("   DEMO 1: ZERO-TOLERANCE HARDWARE SAFETY MONITOR")
    print("=" * 65)
    print("\nArming safety monitor...")
    print("Zero-tolerance active: ANY physical mouse movement or keypress")
    print("will immediately abort the agent and display the safety alert.")
    print("\n>>> TAKE YOUR HAND OFF THE MOUSE FOR 2 SECONDS... <<<")

    monitor = SafetyMonitor(enable_dialog=True)
    time.sleep(2.0)

    monitor.arm()
    print("\n[ARMED] The safety monitor is now active.")
    print(">>> NOW MOVE YOUR MOUSE OR PRESS ANY KEY ON YOUR KEYBOARD <<<")

    timeout_s = 15.0
    start_t = time.time()
    while not monitor.is_halted and (time.time() - start_t < timeout_s):
        time.sleep(0.05)

    if monitor.is_halted:
        print("\n[VERIFIED] Hardware interrupt caught successfully!")
        print(f"Reason: {monitor.halt_reason}")
    else:
        print("\n[TIMEOUT] No user input detected within 15 seconds.")
        monitor.disarm()


def demo_live_ocr(search_term: str = None):
    print("\n" + "=" * 65)
    print("   DEMO 2: LIVE DESKTOP OCR ELEMENT GROUNDING")
    print("=" * 65)
    print("Taking live screenshot of your desktop...")

    temp_img = BASE_DIR / "data" / "cache" / "live_demo_screenshot.png"
    temp_img.parent.mkdir(parents=True, exist_ok=True)

    img = ImageGrab.grab()
    img.save(temp_img)
    print(f"Captured screen: {img.width}x{img.height} pixels.")

    print("Running on-device Windows Media OCR...")
    engine = WindowsOCREngine()
    start_t = time.time()
    boxes = engine.recognize_image(temp_img)
    elapsed = time.time() - start_t

    print(f"OCR finished in {elapsed:.2f}s. Extracted {len(boxes)} text elements.")
    print("\nSample of detected UI elements on your screen right now:")
    print(f"{'Text':<25} {'Coordinates (X, Y)':<22} {'Size (W x H)'}")
    print("-" * 65)

    for box in boxes[:20]:
        print(f"{box.text[:24]:<25} ({box.center_x:>4}, {box.center_y:>4}){'':<12} {box.width}x{box.height}")

    if search_term:
        print(f"\nSearching for UI target: '{search_term}'...")
        anchor = OCRTargetAnchor(
            target_text=search_term,
            exact_match=False,
            bounding_box=(0, 0, 10, 10),
        )
        coords = engine.locate_target(anchor, boxes)
        if coords:
            print(f"[FOUND] Target '{search_term}' is at screen coordinates: X={coords[0]}, Y={coords[1]}")
        else:
            print(f"[NOT FOUND] Could not find '{search_term}' on screen.")


def demo_replicate():
    print("\n" + "=" * 65)
    print("   DEMO 3: AUTONOMOUS STATE REPLICATION LOOP")
    print("=" * 65)
    print("Constructing a sample trajectory...")

    # Look for a target currently visible on screen (e.g. 'File' or 'Terminal' or 'Help')
    temp_img = BASE_DIR / "data" / "cache" / "live_demo_screenshot.png"
    img = ImageGrab.grab()
    img.save(temp_img)

    engine = WindowsOCREngine()
    boxes = engine.recognize_image(temp_img)

    candidate_texts = ["File", "Edit", "Terminal", "Help", "Run", "View", "Code"]
    found_target = None
    for cand in candidate_texts:
        for b in boxes:
            if cand.lower() in b.text.lower():
                found_target = b
                break
        if found_target:
            break

    if not found_target:
        found_target = boxes[0] if boxes else None

    if found_target:
        print(f"Anchoring simulated action to visible UI element: '{found_target.text}' at ({found_target.center_x}, {found_target.center_y})")
        anchor = OCRTargetAnchor(
            target_text=found_target.text,
            exact_match=False,
            bounding_box=(found_target.x, found_target.y, found_target.width, found_target.height),
        )
    else:
        anchor = OCRTargetAnchor(
            target_text="Test",
            exact_match=False,
            bounding_box=(100, 100, 50, 20),
        )

    trajectory = TaskTrajectory(
        task_name="interactive_demo_replication",
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        screen_resolution=(img.width, img.height),
        steps=[
            ReplicatedAction(
                step_number=1,
                action_type="LEFT_CLICK",
                anchor=anchor,
                post_delay_ms=500,
            ),
        ],
    )

    print("\nExecuting in DRY-RUN mode (safe simulation without physical clicks)...")
    replicator = TaskReplicator(ocr_engine=engine, safety_monitor=SafetyMonitor(enable_dialog=False))
    success = replicator.execute_trajectory(trajectory, dry_run=True)

    if success:
        print("\n[SUCCESS] Autonomous replication sequence verified!")
    else:
        print("\n[HALTED] Replication was safely intercepted.")


def main():
    parser = argparse.ArgumentParser(description="Live demo for Autonomous State Replication Agent.")
    parser.add_argument(
        "--mode",
        choices=["safety", "ocr", "replicate", "all"],
        default="all",
        help="Demo mode to run: 'safety', 'ocr', 'replicate', or 'all' (default: all)",
    )
    parser.add_argument(
        "--find",
        type=str,
        default="File",
        help="Text to search for on screen in OCR mode (default: 'File')",
    )
    args = parser.parse_args()

    if args.mode in ("safety", "all"):
        demo_safety_monitor()

    if args.mode in ("ocr", "all"):
        demo_live_ocr(search_term=args.find)

    if args.mode in ("replicate", "all"):
        demo_replicate()


if __name__ == "__main__":
    main()
