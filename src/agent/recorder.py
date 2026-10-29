"""
Task Demonstration Recorder.
Captures user physical interactions, grabs screen states, extracts OCR anchors,
and compiles a clean TaskTrajectory model for autonomous replication.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import ImageGrab

from src.agent.schemas import (
    OCRBoundingBox,
    OCRTargetAnchor,
    ReplicatedAction,
    TaskTrajectory,
)
from src.agent.ocr_engine import WindowsOCREngine
from src.config import BASE_DIR

logger = logging.getLogger("task_recorder")

TRAJECTORIES_DIR = BASE_DIR / "data" / "trajectories"
SCREENSHOTS_DIR = BASE_DIR / "data" / "recordings"


class TaskRecorder:
    """
    Records demonstration steps by capturing visual screen states and grounding
    actions via local Windows Media OCR anchors.
    """

    def __init__(self, ocr_engine: Optional[WindowsOCREngine] = None):
        self.ocr_engine = ocr_engine or WindowsOCREngine()
        self.current_trajectory: Optional[TaskTrajectory] = None
        self._step_counter = 0

    def start_recording(self, task_name: str) -> TaskTrajectory:
        """Initializes a new recording session."""
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        screen = ImageGrab.grab()
        width, height = screen.size

        self._step_counter = 0
        self.current_trajectory = TaskTrajectory(
            task_name=task_name,
            recorded_at=datetime.now().isoformat(),
            screen_resolution=(width, height),
            steps=[],
        )
        logger.info(f"Started recording task '{task_name}' at {width}x{height} resolution.")
        return self.current_trajectory

    def record_click(
        self,
        x: int,
        y: int,
        action_type: str = "LEFT_CLICK",
        post_delay_ms: int = 400,
    ) -> ReplicatedAction:
        """
        Captures the screen immediately before the click, runs OCR to find the nearest
        text anchor, and logs the action step.
        """
        if self.current_trajectory is None:
            raise RuntimeError("Recording session not started. Call start_recording() first.")

        self._step_counter += 1

        # Capture pre-action screenshot
        screenshot_path = SCREENSHOTS_DIR / f"{self.current_trajectory.task_name}_step_{self._step_counter:03d}.jpg"
        img = ImageGrab.grab()
        img.save(screenshot_path, format="JPEG", quality=85)

        # Run OCR to locate target text
        boxes = self.ocr_engine.recognize_image(screenshot_path)
        anchor = self.ocr_engine.find_anchor_nearest_to(x, y, boxes)

        if anchor is None:
            logger.warning(f"No OCR text found near click ({x}, {y}). Using fallback center anchor.")
            anchor = OCRTargetAnchor(
                target_text="",
                exact_match=False,
                bounding_box=(x, y, 1, 1),
                click_offset_from_center=(0, 0),
            )
        else:
            logger.info(f"Step {self._step_counter}: Click ({x}, {y}) anchored to text '{anchor.target_text}'.")

        action = ReplicatedAction(
            step_number=self._step_counter,
            action_type=action_type,
            anchor=anchor,
            post_delay_ms=post_delay_ms,
        )

        self.current_trajectory.steps.append(action)
        return action

    def record_type_text(self, text: str, post_delay_ms: int = 200) -> ReplicatedAction:
        """Logs a text typing action."""
        if self.current_trajectory is None:
            raise RuntimeError("Recording session not started. Call start_recording() first.")

        self._step_counter += 1
        action = ReplicatedAction(
            step_number=self._step_counter,
            action_type="TYPE_TEXT",
            text_payload=text,
            post_delay_ms=post_delay_ms,
        )
        self.current_trajectory.steps.append(action)
        logger.info(f"Step {self._step_counter}: Type text payload '{text}'.")
        return action

    def record_key_combo(self, keys: List[str], post_delay_ms: int = 300) -> ReplicatedAction:
        """Logs a hotkey combination (e.g. ['ctrl', 's'])."""
        if self.current_trajectory is None:
            raise RuntimeError("Recording session not started. Call start_recording() first.")

        self._step_counter += 1
        action = ReplicatedAction(
            step_number=self._step_counter,
            action_type="KEY_COMBO",
            key_combo=keys,
            post_delay_ms=post_delay_ms,
        )
        self.current_trajectory.steps.append(action)
        logger.info(f"Step {self._step_counter}: Key combination {keys}.")
        return action

    def save_trajectory(self, output_path: Optional[Path] = None) -> Path:
        """Saves recorded trajectory to JSON file."""
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory to save.")

        if output_path is None:
            output_path = TRAJECTORIES_DIR / f"{self.current_trajectory.task_name}_trajectory.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.current_trajectory.model_dump_json(indent=2))

        logger.info(f"Saved task trajectory ({len(self.current_trajectory.steps)} steps) to {output_path}")
        return output_path

