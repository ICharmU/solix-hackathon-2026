"""
Integration and unit tests for Phase 2: Ingestion & Frame-Audio Preprocessing.
"""

import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from src.media_extractor import MediaExtractor
from src.frame_analyzer import FrameAnalyzer
from src.timeline_aligner import TimelineAligner
from src.schemas import AudioSegment, SessionMetadata
from src.preprocess import run_preprocessing

class TestPreprocessingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("data/test_fixtures")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.sample_video = Path("data/sample_test.mp4")

        # Verify or generate sample video
        if not cls.sample_video.exists():
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=4:size=1280x720:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
                str(cls.sample_video)
            ]
            subprocess.run(cmd, check=True, capture_output=True)

    def test_01_probe_media(self):
        extractor = MediaExtractor()
        info = extractor.probe(self.sample_video)
        self.assertTrue(info["has_video"], "Probe should detect video track")
        self.assertTrue(info["has_audio"], "Probe should detect audio track")
        self.assertAlmostEqual(info["duration"], 4.0, delta=0.5)
        self.assertEqual(info["width"], 1280)
        self.assertEqual(info["height"], 720)

    def test_02_extract_frames(self):
        extractor = MediaExtractor()
        out_dir = self.test_dir / "frames_test"
        frames = extractor.extract_frames(self.sample_video, out_dir, fps=1)
        self.assertGreaterEqual(len(frames), 3, "At 1 FPS over 4s, should extract ~4 frames")
        # Verify frame image exists and is readable
        with Image.open(frames[0]) as img:
            self.assertEqual(img.size, (1920, 1080), "Frame should be scaled/padded to 1080p")

    def test_03_extract_audio(self):
        extractor = MediaExtractor()
        out_wav = self.test_dir / "audio_test.wav"
        extracted = extractor.extract_audio(self.sample_video, out_wav, sample_rate=16000)
        self.assertIsNotNone(extracted, "Audio extraction should succeed")
        self.assertTrue(out_wav.exists())
        self.assertGreater(out_wav.stat().st_size, 1000)

    def test_04_frame_analyzer(self):
        analyzer = FrameAnalyzer(diff_threshold=0.01)
        frames_dir = self.test_dir / "static_test"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Create two identical frames and one distinct frame
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img3 = Image.new("RGB", (100, 100), color=(0, 255, 0))

        p1, p2, p3 = frames_dir / "f1.jpg", frames_dir / "f2.jpg", frames_dir / "f3.jpg"
        img1.save(p1)
        img2.save(p2)
        img3.save(p3)

        results = analyzer.analyze_sequence([p1, p2, p3])
        self.assertEqual(len(results), 3)
        self.assertFalse(results[0][1], "First frame should not be marked static")
        self.assertTrue(results[1][1], "Identical second frame must be marked static")
        self.assertFalse(results[2][1], "Distinct third frame must not be marked static")

    def test_05_timeline_alignment(self):
        frames = [Path(f"frame_{i}.jpg") for i in range(4)]
        metrics = [(1.0, False), (0.001, True), (0.15, False), (0.001, True)]
        segments = [
            AudioSegment(start=0.5, end=2.2, text="Entering invoice numbers"),
            AudioSegment(start=2.8, end=3.5, text="Switching to spreadsheet")
        ]
        meta = SessionMetadata(
            session_id="test_session",
            source_video=str(self.sample_video),
            duration_seconds=4.0,
            total_frames=4,
            fps=1,
            has_audio_track=True,
            created_at="2026-10-28T00:00:00Z"
        )

        manifest = TimelineAligner.align(
            frame_paths=frames,
            frame_metrics=metrics,
            audio_segments=segments,
            metadata=meta,
            base_dir=Path(".")
        )

        self.assertEqual(len(manifest.slots), 4)
        # Slot 0 [0.0 - 1.0) overlaps with seg 1 (start 0.5)
        self.assertTrue(manifest.slots[0].has_audio)
        self.assertIn("Entering invoice numbers", manifest.slots[0].audio_text)
        # Slot 1 [1.0 - 2.0) overlaps with seg 1 (end 2.2)
        self.assertTrue(manifest.slots[1].has_audio)
        self.assertTrue(manifest.slots[1].is_static)
        # Slot 3 [3.0 - 4.0) overlaps with seg 2 (end 3.5)
        self.assertTrue(manifest.slots[3].has_audio)
        self.assertIn("Switching to spreadsheet", manifest.slots[3].audio_text)

    def test_06_end_to_end_preprocessing(self):
        manifest_path = run_preprocessing(
            video_path=self.sample_video,
            session_id="test_e2e_session",
            fps=1,
            skip_transcription=False,
            mock_transcription=True,
            mock_vlm=True
        )

        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("slots", data)
        self.assertGreater(len(data["slots"]), 0)
        self.assertEqual(data["metadata"]["session_id"], "test_e2e_session")

if __name__ == "__main__":
    unittest.main()
