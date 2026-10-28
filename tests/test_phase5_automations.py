"""
Unit Tests for Phase 5: Executable Automations and Interactive Dashboard.
"""

import sys
import unittest
import py_compile
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.automation_generator import generate_automation_package, AUTOMATIONS_DIR
from src.automations.auto_table_validator import validate_table_schema
from src.automations.auto_doc_reference import search_local_references
from src.dashboard_generator import build_dashboard_html, generate_dashboard
from src.config import BASE_DIR, PROCESSED_DIR


class TestPhase5Automations(unittest.TestCase):
    """Test suite covering Phase 5 deliverables."""

    def setUp(self):
        self.synthesis_path = PROCESSED_DIR / "solix_demo_test" / "workflow_synthesis.json"

    def test_automation_generator_produces_files(self):
        """Verifies that automation generator creates all required standalone scripts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_out = Path(tmp_dir)
            files = generate_automation_package(self.synthesis_path, output_dir=tmp_out)
            self.assertEqual(len(files), 3)

            filenames = [f.name for f in files]
            self.assertIn("auto_recording_daemon.py", filenames)
            self.assertIn("auto_table_validator.py", filenames)
            self.assertIn("auto_doc_reference.py", filenames)

            for f in files:
                self.assertTrue(f.exists())
                self.assertGreater(f.stat().st_size, 200)

    def test_automation_scripts_valid_python(self):
        """Ensures that all generated automation scripts compile cleanly without syntax errors."""
        scripts = [
            AUTOMATIONS_DIR / "auto_recording_daemon.py",
            AUTOMATIONS_DIR / "auto_table_validator.py",
            AUTOMATIONS_DIR / "auto_doc_reference.py",
        ]
        for script in scripts:
            self.assertTrue(script.exists(), f"Missing script: {script}")
            # py_compile will raise py_compile.PyCompileError if syntax is invalid
            compiled = py_compile.compile(str(script), doraise=True)
            self.assertIsNotNone(compiled)

    def test_table_validator_logic(self):
        """Tests pure-Python table schema validator with matching and missing columns."""
        sample_html = """
        <html>
          <body>
            <table>
              <tr>
                <th>Timestamp</th>
                <th>Application</th>
                <th>Status</th>
              </tr>
              <tr>
                <td>0s</td>
                <td>OBS Studio</td>
                <td>Active</td>
              </tr>
            </table>
          </body>
        </html>
        """
        # Should pass
        is_valid, matched, missing = validate_table_schema(sample_html, ["Timestamp", "Application", "Status"])
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(matched), 3)

        # Should fail on missing columns
        is_valid2, matched2, missing2 = validate_table_schema(sample_html, ["Timestamp", "NonExistentColumn"])
        self.assertFalse(is_valid2)
        self.assertIn("NonExistentColumn", missing2)
        self.assertIn("Timestamp", matched2)

    def test_doc_reference_search(self):
        """Tests local file query indexing, skipping hidden virtualenv directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            # Create a test markdown file
            test_md = td / "notes.md"
            test_md.write_text("# Schema Definitions\nThis explains the Pipeline Architecture.", encoding="utf-8")

            # Create a hidden dir that should be skipped
            hidden_dir = td / ".pixi"
            hidden_dir.mkdir()
            (hidden_dir / "ignored.py").write_text("Architecture = 1", encoding="utf-8")

            matches = search_local_references("Architecture", td)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["file"], "notes.md")
            self.assertIn("Architecture", matches[0]["content"])

    def test_dashboard_generator_output(self):
        """Tests dashboard generator building HTML with all required Generative UI tokens and sections."""
        dummy_timeline = {
            "slots": [
                {
                    "timestamp": 0,
                    "descriptor": {"active_application": "OBS Studio", "window_title": "OBS", "observed_interaction": "Audio adjust"},
                    "embedding": [0.1, 0.2, 0.3],
                    "raw_text": "OBS Studio",
                },
                {
                    "timestamp": 1,
                    "descriptor": {"active_application": "Chrome", "window_title": "Docs", "observed_interaction": "Reading docs"},
                    "embedding": [-0.1, 0.5, 0.2],
                    "raw_text": "Chrome",
                },
            ]
        }
        dummy_segmented = {
            "total_segments": 2,
            "total_clusters": 2,
            "segments": [
                {"segment_id": "seg_01", "start_time": 0, "end_time": 0, "duration_seconds": 1, "primary_application": "OBS Studio", "cluster_id": "cluster_01", "is_interruption": False, "summary_interaction": "Adjust audio", "slot_timestamps": [0]},
                {"segment_id": "seg_02", "start_time": 1, "end_time": 1, "duration_seconds": 1, "primary_application": "Chrome", "cluster_id": "cluster_02", "is_interruption": False, "summary_interaction": "Reading", "slot_timestamps": [1]},
            ],
        }
        dummy_synthesis = {
            "session_id": "test_session",
            "total_duration_seconds": 2.0,
            "executive_summary": "Test executive summary.",
            "macro_steps": [{"step_number": 1, "title": "Setup", "application": "OBS Studio", "time_range": "0s - 1s", "duration_seconds": 1, "intent_and_logic": "Setup"}],
            "friction_ledger": [{"category": "Setup", "observed_time_seconds": 1, "percentage_of_session": 50.0, "severity": "Low", "friction_description": "Setup latency", "automation_potential": "Automate"}],
        }
        dummy_automations = {
            "AUTO-01": "print('AUTO-01')",
            "AUTO-02": "print('AUTO-02')",
            "AUTO-03": "print('AUTO-03')",
        }

        html = build_dashboard_html(
            session_id="test_session",
            timeline_data=dummy_timeline,
            segmented_data=dummy_segmented,
            synthesis_data=dummy_synthesis,
            automations_code=dummy_automations,
        )

        self.assertIn("https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js", html)
        self.assertIn("scrubber-slider", html)
        self.assertIn("timeline-svg", html)
        self.assertIn("tab-automations", html)
        self.assertIn("AUTO-01", html)
        self.assertIn("OBS Studio", html)

    def test_generate_dashboard_file_creation(self):
        """Verifies generate_dashboard generates files on disk."""
        out_path = generate_dashboard("solix_demo_test")
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()

