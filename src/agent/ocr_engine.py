"""
Local OCR Grounding Engine for Windows.
Wraps Windows native Media OCR without cloud API calls or heavy ML dependencies.
Extracts screen elements with bounding boxes and provides target anchor resolution.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from src.agent.schemas import OCRBoundingBox, OCRTargetAnchor

logger = logging.getLogger("ocr_engine")

PS1_SCRIPT_PATH = Path(__file__).resolve().parent / "ocr_engine.ps1"


class WindowsOCREngine:
    """
    Executes Windows Media OCR on image files or screen buffers,
    returning structured bounding boxes and resolving target text locations.
    """

    def __init__(self, script_path: Optional[Path] = None):
        self.script_path = script_path or PS1_SCRIPT_PATH
        if not self.script_path.exists():
            raise FileNotFoundError(f"OCR PowerShell helper script not found at {self.script_path}")

    def recognize_image(self, image_path: Path) -> List[OCRBoundingBox]:
        """
        Runs local OCR on an image file and returns all detected text boxes.
        """
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []

        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.script_path),
            "-ImagePath",
            str(image_path),
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                stdin=subprocess.DEVNULL,
                timeout=15.0,
            )
            output = res.stdout.strip()
            if not output:
                return []

            # Parse JSON from stdout
            data = json.loads(output)
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = [data]
                else:
                    return []

            boxes = []
            for item in data:
                boxes.append(
                    OCRBoundingBox(
                        text=str(item.get("Text", "")),
                        x=int(item.get("X", 0)),
                        y=int(item.get("Y", 0)),
                        width=int(item.get("Width", 0)),
                        height=int(item.get("Height", 0)),
                    )
                )
            return boxes

        except subprocess.CalledProcessError as e:
            logger.error(f"OCR PowerShell script failed with code {e.returncode}: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode OCR JSON output: {e}\nRaw output: {res.stdout[:200]}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error running OCR: {e}")
            return []

    def find_anchor_nearest_to(
        self,
        x: int,
        y: int,
        boxes: List[OCRBoundingBox],
        max_distance: int = 250,
    ) -> Optional[OCRTargetAnchor]:
        """
        Finds the OCR text element closest to the click coordinates (x, y)
        to anchor a user demonstration action.
        """
        if not boxes:
            return None

        best_box = None
        best_dist = float("inf")

        for box in boxes:
            # Check if point is inside bounding box
            if (box.x <= x <= box.x + box.width) and (box.y <= y <= box.y + box.height):
                offset_x = x - box.center_x
                offset_y = y - box.center_y
                return OCRTargetAnchor(
                    target_text=box.text,
                    exact_match=True,
                    bounding_box=(box.x, box.y, box.width, box.height),
                    click_offset_from_center=(offset_x, offset_y),
                )

            # Euclidean distance to box center
            dist = ((box.center_x - x) ** 2 + (box.center_y - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_box = box

        if best_box and best_dist <= max_distance:
            offset_x = x - best_box.center_x
            offset_y = y - best_box.center_y
            return OCRTargetAnchor(
                target_text=best_box.text,
                exact_match=True,
                bounding_box=(best_box.x, best_box.y, best_box.width, best_box.height),
                click_offset_from_center=(offset_x, offset_y),
            )

        return None

    def locate_target(
        self,
        anchor: OCRTargetAnchor,
        boxes: List[OCRBoundingBox],
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the target coordinates on the current screen based on the anchor.
        Supports single word and multi-word phrase matching.
        Returns (click_x, click_y) or None if target is not found.
        """
        if not boxes:
            return None

        target_text = anchor.target_text.strip().lower()

        # 1. Exact token match
        for box in boxes:
            if box.text.strip().lower() == target_text:
                click_x = box.center_x + anchor.click_offset_from_center[0]
                click_y = box.center_y + anchor.click_offset_from_center[1]
                return click_x, click_y

        # 2. Substring or multi-word search
        words = target_text.split()
        if len(words) > 1:
            # Try to match adjacent boxes on the same horizontal line
            for i in range(len(boxes) - len(words) + 1):
                window = boxes[i : i + len(words)]
                window_text = " ".join(b.text.strip().lower() for b in window)
                if target_text in window_text:
                    # Target center is middle of window
                    avg_center_x = (window[0].center_x + window[-1].center_x) // 2
                    avg_center_y = (window[0].center_y + window[-1].center_y) // 2
                    return (
                        avg_center_x + anchor.click_offset_from_center[0],
                        avg_center_y + anchor.click_offset_from_center[1],
                    )

        # 3. Partial containment match (target string is inside the box text)
        if not anchor.exact_match and len(target_text) >= 2:
            for box in boxes:
                box_text = box.text.strip().lower()
                if target_text in box_text:
                    click_x = box.center_x + anchor.click_offset_from_center[0]
                    click_y = box.center_y + anchor.click_offset_from_center[1]
                    return click_x, click_y

        return None

