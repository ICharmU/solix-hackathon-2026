"""
Unit and integration tests for Tier 1 Multimodal Descriptor Extraction and Embeddings.
"""

import json
from pathlib import Path
import unittest

from src.schemas import FrameDescriptor, SessionMetadata, TimelineManifest, TimelineSlot
from src.tier1_descriptor import enrich_manifest

class TestTier1Descriptor(unittest.TestCase):
    def test_01_frame_descriptor_raw_text(self):
        desc = FrameDescriptor(
            active_application="Microsoft Excel",
            window_title="Invoices_Q3.xlsx",
            schema_context="Columns: ID, Vendor, Amount, Status",
            observed_interaction="Entering invoice row",
            background_apps=["Google Chrome", "Slack"]
        )
        raw_text = desc.to_raw_text(audio_context="Checking vendor payment terms")
        self.assertIn("App: Microsoft Excel", raw_text)
        self.assertIn("Title: Invoices_Q3.xlsx", raw_text)
        self.assertIn("Context: Columns: ID, Vendor, Amount, Status", raw_text)
        self.assertIn("Action: Entering invoice row", raw_text)
        self.assertIn("Background: Google Chrome, Slack", raw_text)
        self.assertIn("Audio: Checking vendor payment terms", raw_text)

    def test_02_timeline_slot_serialization(self):
        desc = FrameDescriptor(
            active_application="VS Code",
            window_title="preprocess.py",
            schema_context="Code editor",
            observed_interaction="Typing code",
            background_apps=[]
        )
        slot = TimelineSlot(
            timestamp=12,
            frame_path="frames/frame_000012.jpg",
            has_audio=True,
            audio_text="Writing test functions",
            is_static=False,
            diff_score=0.045,
            descriptor=desc,
            raw_text=desc.to_raw_text("Writing test functions"),
            embedding=[0.1, -0.2, 0.3]
        )

        dumped_json = slot.model_dump_json()
        parsed = json.loads(dumped_json)

        self.assertIn("descriptor", parsed)
        self.assertEqual(parsed["descriptor"]["active_application"], "VS Code")
        self.assertEqual(parsed["raw_text"], slot.raw_text)
        self.assertEqual(parsed["embedding"], [0.1, -0.2, 0.3])

    def test_03_enrich_manifest_mock(self):
        slots = [
            TimelineSlot(timestamp=0, frame_path="frame_0.jpg", is_static=False, diff_score=1.0),
            TimelineSlot(timestamp=1, frame_path="frame_1.jpg", is_static=True, diff_score=0.001),
            TimelineSlot(timestamp=2, frame_path="frame_2.jpg", is_static=False, diff_score=0.05),
        ]
        meta = SessionMetadata(
            session_id="mock_session",
            source_video="mock.mp4",
            duration_seconds=3.0,
            total_frames=3,
            fps=1,
            has_audio_track=False,
            created_at="2026-10-28T00:00:00Z"
        )
        manifest = TimelineManifest(metadata=meta, slots=slots)

        enriched = enrich_manifest(manifest, mock=True)
        self.assertEqual(len(enriched.slots), 3)
        for s in enriched.slots:
            self.assertIsNotNone(s.descriptor, "Every slot must have a descriptor")
            self.assertIsNotNone(s.raw_text, "Every slot must have raw_text")
            self.assertIsNotNone(s.embedding, "Every slot must have an embedding")
            self.assertGreater(len(s.embedding), 0, "Embedding must be a non-empty vector")

if __name__ == "__main__":
    unittest.main()

