"""
Autonomous State Replication Package.
Enables real-time demonstration recording, local OCR element grounding,
autonomous UI replication, and zero-tolerance hardware safety halting.
"""

from src.agent.schemas import (
    OCRBoundingBox,
    OCRTargetAnchor,
    ReplicatedAction,
    TaskTrajectory,
)

__all__ = [
    "OCRBoundingBox",
    "OCRTargetAnchor",
    "ReplicatedAction",
    "TaskTrajectory",
]
