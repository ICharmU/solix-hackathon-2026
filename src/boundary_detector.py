"""
Task Boundary Detection and Timeline Segmentation.
Identifies breakpoints based on consecutive frame embedding difference thresholds
and segments the multimodal timeline into discrete task routines.
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from src.schemas import TaskSegment, TimelineSlot, TimelineManifest


class BoundaryDetector:
    """
    Detects task boundaries along the timeline using embedding derivative thresholds
    and slices the recording into contiguous TaskSegments.
    """

    def __init__(self, diff_threshold: float = 0.5, sub_diff_threshold: float = 0.25):
        """
        Args:
            diff_threshold: Heuristic threshold for macro task breakpoints (default: 0.5).
            sub_diff_threshold: Secondary threshold for detecting internal subroutine shifts (default: 0.25).
        """
        self.diff_threshold = diff_threshold
        self.sub_diff_threshold = sub_diff_threshold

    def compute_consecutive_differences(self, slots: List[TimelineSlot]) -> np.ndarray:
        """
        Computes cosine distance Δ(t) = 1 - cos(e_t, e_{t-1}) across all consecutive slots.
        """
        if len(slots) < 2:
            return np.array([], dtype=np.float32)

        diffs = []
        for i in range(1, len(slots)):
            e_prev = np.array(slots[i - 1].embedding or [], dtype=np.float32)
            e_curr = np.array(slots[i].embedding or [], dtype=np.float32)

            if len(e_prev) == 0 or len(e_curr) == 0:
                diffs.append(0.0)
                continue

            norm_prev = np.linalg.norm(e_prev)
            norm_curr = np.linalg.norm(e_curr)

            if norm_prev > 1e-6 and norm_curr > 1e-6:
                dot = float(np.dot(e_prev, e_curr) / (norm_prev * norm_curr))
            else:
                dot = float(np.dot(e_prev, e_curr))

            dot = max(-1.0, min(1.0, dot))
            diffs.append(1.0 - dot)

        return np.array(diffs, dtype=np.float32)

    def detect_breakpoints(self, slots: List[TimelineSlot], use_macro_only: bool = False) -> List[int]:
        """
        Identifies slot indices where task transitions occur.
        Always includes index 0 as the starting point.

        Args:
            slots: Timeline slots with embeddings.
            use_macro_only: If True, only splits at Δ > diff_threshold (e.g. > 0.5).
                            If False, also splits at app changes or Δ > sub_diff_threshold.
        """
        if not slots:
            return []

        breakpoint_indices = [0]
        consecutive_diffs = self.compute_consecutive_differences(slots)

        for i, diff in enumerate(consecutive_diffs, start=1):
            is_macro_break = diff >= self.diff_threshold

            if use_macro_only:
                if is_macro_break:
                    breakpoint_indices.append(i)
            else:
                # Hierarchical: macro breakpoint, significant sub-routine shift, or app transition
                prev_app = slots[i - 1].descriptor.active_application if slots[i - 1].descriptor else ""
                curr_app = slots[i].descriptor.active_application if slots[i].descriptor else ""
                app_changed = (prev_app != curr_app and bool(prev_app) and bool(curr_app))

                if is_macro_break or (app_changed and diff >= self.sub_diff_threshold):
                    breakpoint_indices.append(i)

        return sorted(list(set(breakpoint_indices)))

    def segment_timeline(
        self, slots: List[TimelineSlot], use_macro_only: bool = False
    ) -> List[TaskSegment]:
        """
        Slices the timeline slots into contiguous TaskSegments.
        Computes the centroid embedding and metadata summary for each segment.
        """
        if not slots:
            return []

        breakpoints = self.detect_breakpoints(slots, use_macro_only=use_macro_only)
        # Add termination index if not already present
        if len(slots) not in breakpoints:
            breakpoints.append(len(slots))

        segments: List[TaskSegment] = []

        for seg_idx in range(len(breakpoints) - 1):
            start_idx = breakpoints[seg_idx]
            end_idx = breakpoints[seg_idx + 1]  # exclusive
            seg_slots = slots[start_idx:end_idx]

            if not seg_slots:
                continue

            start_t = seg_slots[0].timestamp
            end_t = seg_slots[-1].timestamp
            duration = max(1, end_t - start_t + 1)

            # Compute dominant application & window
            apps = [
                s.descriptor.active_application for s in seg_slots
                if s.descriptor and s.descriptor.active_application
            ]
            primary_app = Counter(apps).most_common(1)[0][0] if apps else "Unknown"

            titles = [
                s.descriptor.window_title for s in seg_slots
                if s.descriptor and s.descriptor.window_title
            ]
            primary_title = Counter(titles).most_common(1)[0][0] if titles else ""

            # Consolidated interaction description
            actions = [
                s.descriptor.observed_interaction for s in seg_slots
                if s.descriptor and s.descriptor.observed_interaction
            ]
            distinct_actions = list(dict.fromkeys(actions))
            summary_action = "; ".join(distinct_actions[:3]) if distinct_actions else "User activity"

            # Compute centroid embedding
            embeddings = [s.embedding for s in seg_slots if s.embedding]
            centroid = None
            if embeddings:
                arr = np.array(embeddings, dtype=np.float32)
                mean_vec = np.mean(arr, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-6:
                    centroid = (mean_vec / norm).tolist()
                else:
                    centroid = mean_vec.tolist()

            seg = TaskSegment(
                segment_id=f"seg_{len(segments)+1:02d}",
                start_time=start_t,
                end_time=end_t,
                duration_seconds=duration,
                primary_application=primary_app,
                primary_window=primary_title,
                summary_interaction=summary_action,
                centroid_embedding=centroid,
                slots_count=len(seg_slots),
                slot_timestamps=[s.timestamp for s in seg_slots],
            )
            segments.append(seg)

        return segments

