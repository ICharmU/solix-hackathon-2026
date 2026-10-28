"""
Unit tests for Phase 4: Tier 2 Workflow Synthesis and Automation Blueprint generation.
"""

import json
import unittest
from pathlib import Path

from src.schemas import SegmentedWorkflowManifest, TaskSegment, SubroutineCluster, StitchedSequence
from src.tier2_synthesis import (
    Tier2Synthesizer,
    generate_mock_synthesis,
    format_markdown_report,
    run_synthesis_pipeline,
)


class TestTier2Synthesis(unittest.TestCase):

    def setUp(self):
        self.sample_manifest = SegmentedWorkflowManifest(
            session_id="unit_test_synthesis_session",
            source_video="data/test.mp4",
            created_at="2026-10-28T12:00:00Z",
            difference_threshold=0.5,
            total_segments=2,
            total_clusters=2,
            total_stitched_sequences=2,
            segments=[
                TaskSegment(
                    segment_id="seg_01",
                    start_time=0,
                    end_time=10,
                    duration_seconds=11,
                    primary_application="OBS Studio",
                    summary_interaction="Adjusting audio mixer",
                ),
                TaskSegment(
                    segment_id="seg_02",
                    start_time=11,
                    end_time=30,
                    duration_seconds=20,
                    primary_application="VS Code",
                    summary_interaction="Editing code",
                ),
            ],
            clusters=[
                SubroutineCluster(
                    cluster_id="cluster_01",
                    label="OBS Routine",
                    segment_ids=["seg_01"],
                    total_duration_seconds=11,
                    primary_applications=["OBS Studio"],
                ),
                SubroutineCluster(
                    cluster_id="cluster_02",
                    label="VS Code Routine",
                    segment_ids=["seg_02"],
                    total_duration_seconds=20,
                    primary_applications=["VS Code"],
                ),
            ],
            stitched_sequences=[
                StitchedSequence(
                    sequence_id="seq_01",
                    macro_task_label="OBS Routine",
                    primary_segment_ids=["seg_01"],
                    total_active_seconds=11,
                ),
                StitchedSequence(
                    sequence_id="seq_02",
                    macro_task_label="VS Code Routine",
                    primary_segment_ids=["seg_02"],
                    total_active_seconds=20,
                ),
            ],
        )

    def test_mock_synthesis_generation(self):
        report = generate_mock_synthesis(self.sample_manifest)
        self.assertEqual(report.session_id, "unit_test_synthesis_session")
        self.assertGreater(len(report.macro_steps), 0)
        self.assertGreater(len(report.friction_ledger), 0)
        self.assertGreater(len(report.automation_specs), 0)
        self.assertIn("flowchart TD", report.mermaid_diagram)

    def test_markdown_formatting(self):
        report = generate_mock_synthesis(self.sample_manifest)
        md = format_markdown_report(report)
        self.assertIn("# Tier 2 Workflow Synthesis & Automation Blueprint", md)
        self.assertIn("```mermaid", md)
        self.assertIn("## 4. Time-Allocation & Friction Ledger", md)
        self.assertIn("## 5. Production Automation Blueprints", md)

    def test_pipeline_execution_mock(self):
        input_p = Path("data/processed/solix_demo_test/segmented_workflow.json")
        if not input_p.exists():
            self.skipTest("segmented_workflow.json not found.")

        out_json = Path("data/processed/solix_demo_test/test_synth_output.json")
        out_md = Path("data/processed/solix_demo_test/test_synth_output.md")

        report = run_synthesis_pipeline(
            input_manifest_path=input_p,
            output_json_path=out_json,
            report_markdown_path=out_md,
            mock=True,
        )

        self.assertTrue(out_json.exists())
        self.assertTrue(out_md.exists())

        # Clean up test outputs
        if out_json.exists():
            out_json.unlink()
        if out_md.exists():
            out_md.unlink()


if __name__ == "__main__":
    unittest.main()

