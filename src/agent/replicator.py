"""
Autonomous State Replicator Engine.
Drives the user interface autonomously using OCR text anchors to locate UI targets,
dispatching native Win32 virtual mouse and keyboard inputs under the supervision
of the zero-tolerance SafetyMonitor.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Callable
from PIL import ImageGrab

from src.agent.schemas import TaskTrajectory, ReplicatedAction
from src.agent.ocr_engine import WindowsOCREngine
from src.agent.virtual_input import VirtualInput
from src.agent.safety import SafetyMonitor, UserInputHaltException
from src.config import BASE_DIR

logger = logging.getLogger("task_replicator")

TEMP_DIR = BASE_DIR / "data" / "cache"


class TaskReplicator:
    """
    Autonomously executes recorded task trajectories by visually searching for
    OCR text targets on the live desktop screen and dispatching virtual inputs.
    """

    def __init__(
        self,
        ocr_engine: Optional[WindowsOCREngine] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
    ):
        self.ocr_engine = ocr_engine or WindowsOCREngine()
        self.safety_monitor = safety_monitor or SafetyMonitor()
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    def execute_trajectory(
        self,
        trajectory: TaskTrajectory,
        dry_run: bool = False,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """
        Executes a task trajectory step by step.
        Returns True if all steps completed successfully, False if aborted or failed.
        """
        logger.info(f"Starting autonomous replication of task: '{trajectory.task_name}' ({len(trajectory.steps)} steps)")
        if dry_run:
            logger.info("[DRY RUN MODE] Simulating execution without dispatching inputs.")

        # Arm the zero-tolerance safety monitor
        if not self.safety_monitor.is_halted:
            self.safety_monitor.arm()

        try:
            for step in trajectory.steps:
                # Check for physical user intervention before every step
                self.safety_monitor.check_or_raise()

                if step_callback:
                    step_callback(step.step_number)
                    self.safety_monitor.check_or_raise()

                logger.info(f"\n--- Executing Step {step.step_number}/{len(trajectory.steps)}: {step.action_type} ---")
                success = self._execute_step(step, dry_run=dry_run)
                if not success:
                    logger.error(f"Step {step.step_number} failed. Halting replication.")
                    return False

                # Settle delay
                time.sleep(step.post_delay_ms / 1000.0)

                # Check safety after step
                self.safety_monitor.check_or_raise()

            logger.info(f"\n[SUCCESS] Completed all {len(trajectory.steps)} steps of task '{trajectory.task_name}' successfully.")
            return True

        except UserInputHaltException as e:
            logger.warning(f"Replication halted by user safety intervention: {e}")
            return False
        except Exception as e:
            logger.error(f"Replication terminated with error: {e}")
            return False
        finally:
            self.safety_monitor.disarm()

    def _execute_step(self, step: ReplicatedAction, dry_run: bool = False) -> bool:
        """Executes a single replicated action step."""
        if step.action_type in ("LEFT_CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            if step.anchor is None:
                logger.error("Click action missing target OCR anchor.")
                return False

            # Capture current screen
            current_screenshot = TEMP_DIR / "current_screen_recon.jpg"
            screen_img = ImageGrab.grab()
            screen_img.save(current_screenshot, format="JPEG", quality=85)

            # Check safety before OCR
            self.safety_monitor.check_or_raise()

            # OCR scan current screen
            boxes = self.ocr_engine.recognize_image(current_screenshot)

            # Check safety after OCR
            self.safety_monitor.check_or_raise()
            target_coords = self.ocr_engine.locate_target(step.anchor, boxes)

            if target_coords is None:
                logger.error(f"Could not locate OCR anchor '{step.anchor.target_text}' on live screen.")
                # Fallback to original demonstrated bounding box if text is somehow missed
                x, y, w, h = step.anchor.bounding_box
                target_coords = (x + w // 2, y + h // 2)
                logger.warning(f"Falling back to original demonstrated coordinates ({target_coords[0]}, {target_coords[1]}).")

            click_x, click_y = target_coords
            logger.info(f"Target located at ({click_x}, {click_y}) for text '{step.anchor.target_text}'.")

            if not dry_run:
                self.safety_monitor.pause_for_agent_action()
                is_double = (step.action_type == "DOUBLE_CLICK")
                btn = "right" if step.action_type == "RIGHT_CLICK" else "left"
                VirtualInput.mouse_click(click_x, click_y, button=btn, double=is_double)
                time.sleep(0.05)
                self.safety_monitor.resume_after_agent_action()

            return True

        elif step.action_type == "TYPE_TEXT":
            payload = step.text_payload or ""
            logger.info(f"Typing text payload: '{payload}'")
            if not dry_run:
                VirtualInput.type_text(payload)
            return True

        elif step.action_type == "KEY_COMBO":
            combo = step.key_combo or []
            logger.info(f"Dispatching key combination: {combo}")
            if not dry_run:
                VirtualInput.send_key_combo(combo)
            return True

        else:
            logger.error(f"Unknown action type: {step.action_type}")
            return False
