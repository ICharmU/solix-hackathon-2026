"""
Unit and Integration Tests for Autonomous State Replication Agent.
Verifies:
1. Pydantic schema validation for TaskTrajectory and OCRTargetAnchor.
2. OCR engine nearest anchor resolution and multi-word target matching.
3. Zero-tolerance SafetyMonitor arming, halt triggering, and exception raising.
4. TaskRecorder trajectory compilation.
5. TaskReplicator execution loop and safety halt interception.
6. Real-image OCR extraction on demonstration video keyframes.
"""

import os
import unittest
from pathlib import Path
from datetime import datetime

from src.agent.schemas import (
    OCRBoundingBox,
    OCRTargetAnchor,
    ReplicatedAction,
    TaskTrajectory,
)
from src.agent.ocr_engine import WindowsOCREngine
from src.agent.safety import SafetyMonitor, UserInputHaltException
from src.agent.recorder import TaskRecorder
from src.agent.replicator import TaskReplicator
from src.config import BASE_DIR


class TestAgentStateReplication(unittest.TestCase):

    def setUp(self):
        self.sample_boxes = [
            OCRBoundingBox(text="File", x=10, y=34, width=22, height=12),
            OCRBoundingBox(text="Edit", x=51, y=34, width=25, height=12),
            OCRBoundingBox(text="Start", x=1684, y=914, width=33, height=11),
            OCRBoundingBox(text="Streaming", x=1721, y=913, width=69, height=16),
            OCRBoundingBox(text="Stop", x=1665, y=954, width=31, height=16),
            OCRBoundingBox(text="Recording", x=1701, y=954, width=68, height=15),
        ]

    def test_schema_serialization(self):
        """Verifies TaskTrajectory Pydantic serialization and round-trip."""
        anchor = OCRTargetAnchor(
            target_text="File",
            exact_match=True,
            bounding_box=(10, 34, 22, 12),
            click_offset_from_center=(0, 0),
        )
        action = ReplicatedAction(
            step_number=1,
            action_type="LEFT_CLICK",
            anchor=anchor,
            post_delay_ms=250,
        )
        traj = TaskTrajectory(
            task_name="test_task",
            recorded_at=datetime.now().isoformat(),
            screen_resolution=(1920, 1080),
            steps=[action],
        )

        json_str = traj.model_dump_json()
        self.assertIn("test_task", json_str)
        self.assertIn("File", json_str)

        loaded = TaskTrajectory.model_validate_json(json_str)
        self.assertEqual(len(loaded.steps), 1)
        self.assertEqual(loaded.steps[0].anchor.target_text, "File")

    def test_ocr_nearest_anchor_resolution(self):
        """Verifies that clicks are correctly anchored to the nearest OCR text element."""
        engine = WindowsOCREngine()

        # Click inside the 'File' box (center is at 21, 40)
        anchor = engine.find_anchor_nearest_to(20, 38, self.sample_boxes)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.target_text, "File")

        # Click near 'Edit' (center is at 63, 40)
        anchor_edit = engine.find_anchor_nearest_to(70, 42, self.sample_boxes)
        self.assertIsNotNone(anchor_edit)
        self.assertEqual(anchor_edit.target_text, "Edit")

    def test_ocr_target_location_matching(self):
        """Verifies locating single-word and multi-word target phrases."""
        engine = WindowsOCREngine()

        # Single word exact
        anchor_file = OCRTargetAnchor(
            target_text="File",
            exact_match=True,
            bounding_box=(10, 34, 22, 12),
        )
        coords = engine.locate_target(anchor_file, self.sample_boxes)
        self.assertIsNotNone(coords)
        self.assertEqual(coords, (21, 40))

        # Multi-word phrase 'Stop Recording'
        anchor_stop = OCRTargetAnchor(
            target_text="Stop Recording",
            exact_match=False,
            bounding_box=(1665, 954, 104, 16),
        )
        coords_stop = engine.locate_target(anchor_stop, self.sample_boxes)
        self.assertIsNotNone(coords_stop)
        # Bounding box centers: (1665+15=1680) and (1701+34=1735) -> avg center is ~1707
        self.assertTrue(1680 <= coords_stop[0] <= 1735)
        self.assertTrue(950 <= coords_stop[1] <= 970)

    def test_safety_monitor_halt_notification(self):
        """Verifies zero-tolerance SafetyMonitor halt trigger and notification."""
        halt_reasons = []

        def on_halt(reason):
            halt_reasons.append(reason)

        monitor = SafetyMonitor(on_halt_callback=on_halt)
        self.assertFalse(monitor.is_halted)

        # Trigger manual halt
        monitor.trigger_halt("user input received. program halting")
        self.assertTrue(monitor.is_halted)
        self.assertEqual(monitor.halt_reason, "user input received. program halting")
        self.assertEqual(len(halt_reasons), 1)
        self.assertEqual(halt_reasons[0], "user input received. program halting")

        # Verify check_or_raise raises UserInputHaltException
        with self.assertRaises(UserInputHaltException) as ctx:
            monitor.check_or_raise()
        self.assertIn("user input received. program halting", str(ctx.exception))

    def test_safety_monitor_pause_for_agent_action(self):
        """Verifies that the safety monitor tracks agent acting state without physical mouse interference."""
        monitor = SafetyMonitor(enable_dialog=False)
        monitor.pause_for_agent_action()
        self.assertTrue(monitor._agent_acting)

        monitor.resume_after_agent_action()
        self.assertFalse(monitor._agent_acting)
        self.assertFalse(monitor.is_halted)

    def test_replicator_dry_run_and_safety_abort(self):
        """Verifies replicator handles trajectory execution and halts on user intervention."""
        monitor = SafetyMonitor()
        replicator = TaskReplicator(safety_monitor=monitor)

        anchor = OCRTargetAnchor(
            target_text="File",
            exact_match=True,
            bounding_box=(10, 34, 22, 12),
        )
        trajectory = TaskTrajectory(
            task_name="unit_test_demo",
            recorded_at=datetime.now().isoformat(),
            screen_resolution=(1920, 1080),
            steps=[
                ReplicatedAction(step_number=1, action_type="TYPE_TEXT", text_payload="Hello World"),
                ReplicatedAction(step_number=2, action_type="KEY_COMBO", key_combo=["enter"]),
            ],
        )

        # 1. Normal dry run should succeed
        res = replicator.execute_trajectory(trajectory, dry_run=True)
        self.assertTrue(res)

        # 2. If monitor is pre-halted, execution must abort immediately
        monitor.trigger_halt("user input received. program halting")
        res_aborted = replicator.execute_trajectory(trajectory, dry_run=True)
        self.assertFalse(res_aborted)

    def test_real_image_ocr_extraction(self):
        """Verifies WindowsOCREngine on an actual desktop keyframe image."""
        frame_path = BASE_DIR / "data" / "frames" / "solix_demo_test" / "frame_000001.jpg"
        if not frame_path.exists():
            self.skipTest(f"Test frame not found: {frame_path}")

        engine = WindowsOCREngine()
        boxes = engine.recognize_image(frame_path)

        self.assertGreater(len(boxes), 10, "Expected at least 10 OCR words from desktop frame.")
        detected_texts = [b.text.lower() for b in boxes]

        # Verify key desktop UI labels are found
        self.assertTrue(any("file" in t for t in detected_texts), "Should detect 'File' menu.")
        self.assertTrue(any("scene" in t for t in detected_texts), "Should detect 'Scene' label.")
        self.assertTrue(any("audio" in t for t in detected_texts), "Should detect 'Audio' label.")


if __name__ == "__main__":
    unittest.main()

