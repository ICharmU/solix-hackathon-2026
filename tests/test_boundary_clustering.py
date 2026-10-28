"""
Unit and integration tests for Phase 3:
Boundary detection, task segmentation, subroutine clustering, and sequence stitching.
"""

import unittest
import numpy as np
from pathlib import Path

from src.schemas import TimelineSlot, FrameDescriptor, TimelineManifest, SessionMetadata
from src.boundary_detector import BoundaryDetector
from src.workflow_stitcher import WorkflowStitcher, process_timeline_workflow, cosine_similarity


class TestBoundaryClustering(unittest.TestCase):

    def _generate_synthetic_vector(self, base_val: float, dim: int = 1536) -> list:
        """Generates a deterministic normalized synthetic embedding."""
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = base_val
        vec[1] = 1.0 - abs(base_val)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist()

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=4)

    def test_breakpoint_detection_at_threshold_05(self):
        """Tests that transitions with difference > 0.5 are flagged as distinct task breakpoints."""
        v_obs = self._generate_synthetic_vector(1.0)
        v_chrome = self._generate_synthetic_vector(-1.0)  # orthogonal / opposite -> diff ~ 2.0 or 1.0

        slots = [
            TimelineSlot(
                timestamp=0,
                frame_path="frame_0.jpg",
                descriptor=FrameDescriptor(active_application="OBS Studio"),
                embedding=v_obs,
            ),
            TimelineSlot(
                timestamp=1,
                frame_path="frame_1.jpg",
                descriptor=FrameDescriptor(active_application="OBS Studio"),
                embedding=v_obs,
            ),
            # Switch to Chrome: difference > 0.5
            TimelineSlot(
                timestamp=2,
                frame_path="frame_2.jpg",
                descriptor=FrameDescriptor(active_application="Google Chrome"),
                embedding=v_chrome,
            ),
            TimelineSlot(
                timestamp=3,
                frame_path="frame_3.jpg",
                descriptor=FrameDescriptor(active_application="Google Chrome"),
                embedding=v_chrome,
            ),
        ]

        detector = BoundaryDetector(diff_threshold=0.5)
        diffs = detector.compute_consecutive_differences(slots)
        self.assertEqual(len(diffs), 3)
        self.assertAlmostEqual(diffs[0], 0.0, places=3)
        self.assertGreater(diffs[1], 0.5)  # Breakpoint at slot 2

        breakpoints = detector.detect_breakpoints(slots, use_macro_only=True)
        self.assertEqual(breakpoints, [0, 2])

        segments = detector.segment_timeline(slots, use_macro_only=True)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].primary_application, "OBS Studio")
        self.assertEqual(segments[0].duration_seconds, 2)
        self.assertEqual(segments[1].primary_application, "Google Chrome")
        self.assertEqual(segments[1].duration_seconds, 2)

    def test_interruption_sequence_stitching(self):
        """Tests that an interleaved pattern A1 -> B -> A2 stitches A1 and A2, tagging B as interruption."""
        stitcher = WorkflowStitcher(cluster_sim_threshold=0.85)

        v_code = self._generate_synthetic_vector(0.9)
        v_browser = self._generate_synthetic_vector(0.1)

        from src.schemas import TaskSegment
        seg_a1 = TaskSegment(
            segment_id="seg_01",
            start_time=0,
            end_time=10,
            duration_seconds=10,
            primary_application="Visual Studio Code",
            centroid_embedding=v_code,
        )
        seg_b = TaskSegment(
            segment_id="seg_02",
            start_time=11,
            end_time=15,
            duration_seconds=5,
            primary_application="Google Chrome",
            centroid_embedding=v_browser,
        )
        seg_a2 = TaskSegment(
            segment_id="seg_03",
            start_time=16,
            end_time=30,
            duration_seconds=15,
            primary_application="Visual Studio Code",
            centroid_embedding=v_code,
        )

        segments = [seg_a1, seg_b, seg_a2]
        clustered_segs, clusters = stitcher.cluster_segments(segments)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clustered_segs[0].cluster_id, clustered_segs[2].cluster_id)
        self.assertNotEqual(clustered_segs[0].cluster_id, clustered_segs[1].cluster_id)

        final_segs, stitched = stitcher.stitch_discontinuous_sequences(clustered_segs, clusters)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0].primary_segment_ids, ["seg_01", "seg_03"])
        self.assertEqual(stitched[0].interleaved_segment_ids, ["seg_02"])
        self.assertEqual(stitched[0].total_active_seconds, 25)
        self.assertEqual(stitched[0].total_interruption_seconds, 5)
        self.assertTrue(final_segs[1].is_interruption)

    def test_end_to_end_real_manifest_processing(self):
        """Tests processing the real solix_demo_test timeline manifest."""
        manifest_p = Path("data/processed/solix_demo_test/timeline_manifest.json")
        if not manifest_p.exists():
            self.skipTest("solix_demo_test manifest not found.")

        out_p = Path("data/processed/solix_demo_test/test_segmented_workflow.json")
        workflow = process_timeline_workflow(
            manifest_path=manifest_p,
            output_path=out_p,
            diff_threshold=0.5,
            use_macro_only=False,
        )

        self.assertGreater(workflow.total_segments, 0)
        self.assertGreater(workflow.total_clusters, 0)
        self.assertGreater(workflow.total_stitched_sequences, 0)
        self.assertTrue(out_p.exists())

        # Clean up test output
        if out_p.exists():
            out_p.unlink()


if __name__ == "__main__":
    unittest.main()

