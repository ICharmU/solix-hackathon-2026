"""
Data Schemas for Autonomous State Replication Agent.
Models OCR anchors, recorded actions, and task execution trajectories.
"""

from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field


class OCRBoundingBox(BaseModel):
    """Bounding box coordinates for an OCR detected element."""
    text: str
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


class OCRTargetAnchor(BaseModel):
    """Target element representation anchored by visual OCR text."""
    target_text: str = Field(..., description="The textual label of the target UI element")
    exact_match: bool = Field(default=False, description="Whether to require exact text match")
    bounding_box: Tuple[int, int, int, int] = Field(..., description="[x, y, width, height] observed during demonstration")
    click_offset_from_center: Tuple[int, int] = Field(
        default=(0, 0),
        description="Relative offset from target center (e.g. for clicking an input box to the right of a label)"
    )


class ReplicatedAction(BaseModel):
    """An atomic input action to replicate."""
    step_number: int
    action_type: str = Field(
        ...,
        description="'LEFT_CLICK', 'DOUBLE_CLICK', 'RIGHT_CLICK', 'TYPE_TEXT', 'KEY_COMBO'"
    )
    anchor: Optional[OCRTargetAnchor] = Field(
        default=None,
        description="OCR anchor for mouse target (None for pure typing/hotkey actions)"
    )
    text_payload: Optional[str] = Field(default=None, description="Text to type for TYPE_TEXT actions")
    key_combo: Optional[List[str]] = Field(default=None, description="Keys to press (e.g. ['ctrl', 's'])")
    post_delay_ms: int = Field(default=300, description="Debounce delay in ms after action dispatch")


class TaskTrajectory(BaseModel):
    """A full recorded task sequence ready for autonomous replication."""
    task_name: str
    recorded_at: str
    screen_resolution: Tuple[int, int]
    steps: List[ReplicatedAction] = Field(default_factory=list)

