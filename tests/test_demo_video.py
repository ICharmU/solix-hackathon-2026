"""
Automated test suite specifically validating the user-provided demo video 'solix_demo_test.mp4'.
"""

import json
from pathlib import Path
import unittest
from PIL import Image

from src.media_extractor import MediaExtractor
from src.frame_analyzer import FrameAnalyzer
from src.preprocess import run_preprocessing

class TestSolixDemoVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_video = Path("tests/fixtures/solix_demo_test.mp4")
        if not cls.demo_video.exists():
            raise FileNotFoundError(
                f"Required demo video fixture not found at {cls.demo_video}. "
                "Ensure solix_demo_test.mp4 is located in tests/fixtures/."
            )
        cls.output_base = Path("data/test_fixtures/demo_video_run")
        cls.output_base.mkdir(parents=True, exist_ok=True)

    def test_01_fixture_presence_and_probe(self):
        """Verifies demo video attributes match expected screen recording profile."""
        extractor = MediaExtractor()
        info = extractor.probe(self.demo_video)

        self.assertTrue(info["has_video"], "Demo recording must have a video track.")
        self.assertTrue(info["has_audio"], "Demo recording must have an audio track.")
        self.assertEqual(info["width"], 1920, "Video resolution width must be 1920.")
        self.assertEqual(info["height"], 1080, "Video resolution height must be 1080.")
        self.assertAlmostEqual(info["duration"], 56.0, delta=1.5, msg="Expected ~56s recording duration.")

    def test_02_frame_extraction_1fps(self):
        """Validates that 1 FPS extraction yields 56 full 1080p frames."""
        extractor = MediaExtractor()
        frames_dir = self.output_base / "frames"
        frames = extractor.extract_frames(self.demo_video, frames_dir, fps=1)

        self.assertEqual(len(frames), 56, f"Expected exactly 56 frames for 56s video, got {len(frames)}.")
        with Image.open(frames[0]) as img:
            self.assertEqual(img.size, (1920, 1080), "First frame must be 1920x1080.")
        with Image.open(frames[-1]) as img:
            self.assertEqual(img.size, (1920, 1080), "Last frame must be 1920x1080.")

    def test_03_audio_track_extraction(self):
        """Validates that audio is cleanly extracted to 16kHz mono PCM WAV."""
        extractor = MediaExtractor()
        audio_wav = self.output_base / "audio.wav"
        extracted_wav = extractor.extract_audio(self.demo_video, audio_wav, sample_rate=16000)

        self.assertIsNotNone(extracted_wav, "Audio extraction must produce a WAV path.")
        self.assertTrue(extracted_wav.exists())
        self.assertGreater(extracted_wav.stat().st_size, 500_000, "Audio WAV file must be non-empty.")

    def test_04_perceptual_diff_analysis(self):
        """Tests that perceptual analyzer correctly scores static vs active screen intervals."""
        extractor = MediaExtractor()
        frames_dir = self.output_base / "frames"
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        if not frames:
            frames = extractor.extract_frames(self.demo_video, frames_dir, fps=1)

        analyzer = FrameAnalyzer(diff_threshold=0.008)
        metrics = analyzer.analyze_sequence(frames)

        self.assertEqual(len(metrics), len(frames))
        # First frame is always initialized as non-static
        self.assertFalse(metrics[0][1], "Frame 0 should not be static.")
        # Ensure that static frames are detected in this real session
        static_frames = [is_stat for _, is_stat in metrics if is_stat]
        self.assertGreater(len(static_frames), 0, "Real recording should contain static frames.")

    def test_05_full_preprocessing_manifest(self):
        """Tests end-to-end manifest creation on the demo video."""
        manifest_path = run_preprocessing(
            video_path=self.demo_video,
            session_id="unit_test_demo_session",
            fps=1,
            skip_transcription=True,  # Skip Whisper API network call in fast test suite
            mock_vlm=True
        )

        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        meta = manifest_data["metadata"]
        slots = manifest_data["slots"]

        self.assertEqual(meta["session_id"], "unit_test_demo_session")
        self.assertEqual(len(slots), 56, "Manifest must contain 56 slots.")
        self.assertEqual(slots[0]["timestamp"], 0)
        self.assertEqual(slots[-1]["timestamp"], 55)

        # Check slot properties including Tier 1 descriptors and embeddings
        for slot in slots:
            self.assertIn("timestamp", slot)
            self.assertIn("frame_path", slot)
            self.assertIn("diff_score", slot)
            self.assertIn("is_static", slot)
            self.assertIn("descriptor", slot)
            self.assertIn("raw_text", slot)
            self.assertIn("embedding", slot)
            self.assertIsNotNone(slot["descriptor"])
            self.assertIsNotNone(slot["raw_text"])
            self.assertIsNotNone(slot["embedding"])

if __name__ == "__main__":
    unittest.main()

