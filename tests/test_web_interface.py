"""
Unit and integration tests for Solix FactLens web interface and FactEngine.
Verifies fact indexing, query resolution, API responses, and strict no-emoji constraint.
"""

import unittest
import json
import re
from pathlib import Path

from src.config import BASE_DIR, PROCESSED_DIR
from src.web.fact_engine import FactEngine

STATIC_DIR = BASE_DIR / "src" / "web" / "static"


class TestFactEngine(unittest.TestCase):
    """Tests indexing and query resolution in FactEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FactEngine("solix_demo_test")

    def test_session_indexed_properly(self):
        """Verifies session deliverables were indexed."""
        summary = self.engine.get_summary_facts()
        self.assertEqual(summary["session_id"], "solix_demo_test")
        self.assertGreater(summary["duration_seconds"], 0)
        self.assertGreater(summary["total_frames"], 0)
        self.assertGreater(summary["total_automations"], 0)
        self.assertGreater(summary["total_steps"], 0)
        self.assertIn("OBS Studio", summary["active_applications"])

    def test_friction_facts(self):
        """Verifies friction ledger facts."""
        friction_facts = self.engine.get_friction_facts()
        self.assertIsInstance(friction_facts, list)
        self.assertGreater(len(friction_facts), 0)
        first = friction_facts[0]
        self.assertEqual(first["type"], "friction")
        self.assertIn("percentage", first)
        self.assertIn("severity", first)

    def test_automation_facts(self):
        """Verifies automation blueprints and python code assets."""
        automations = self.engine.get_automation_facts()
        self.assertGreaterEqual(len(automations), 3)
        blueprint_ids = [a["id"] for a in automations]
        self.assertIn("AUTO-01", blueprint_ids)
        self.assertIn("AUTO-02", blueprint_ids)
        self.assertIn("AUTO-03", blueprint_ids)

        for auto in automations:
            self.assertTrue(len(auto["code"]) > 10, f"Code for {auto['id']} should not be empty")

    def test_step_facts(self):
        """Verifies chronological process macro steps."""
        steps = self.engine.get_step_facts()
        self.assertGreaterEqual(len(steps), 5)
        step_1 = steps[0]
        self.assertEqual(step_1["step_number"], 1)
        self.assertIn("evidence_keyframe", step_1)

    def test_time_allocation_facts(self):
        """Verifies application time distribution calculation."""
        time_facts = self.engine.get_time_facts()
        self.assertGreater(len(time_facts), 0)
        total_pct = sum(t["percentage"] for t in time_facts)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.5)

    def test_query_resolutions(self):
        """Tests that user queries return the correct fact cards."""
        # Overview query
        overview = self.engine.query_facts("")
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0]["card_type"], "summary")

        # Bottlenecks query
        frictions = self.engine.query_facts("where is the bottleneck?")
        self.assertTrue(any(c["card_type"] == "friction" for c in frictions))

        # Automations query
        autos = self.engine.query_facts("show python scripts")
        self.assertTrue(any(c["card_type"] == "automation" for c in autos))

        # Steps query
        flow = self.engine.query_facts("what was the process flow?")
        self.assertTrue(any(c["card_type"] == "step" for c in flow))

        # Direct timestamp query
        ts_fact = self.engine.query_facts("what happened at second 25?")
        self.assertTrue(any(c["card_type"] == "timeline_slot" for c in ts_fact))

    def test_timeline_slot_lookup(self):
        """Verifies exact second slot retrieval."""
        slot = self.engine.get_timeline_slot(25)
        self.assertIsNotNone(slot)
        self.assertEqual(slot["timestamp"], 25)
        self.assertEqual(slot["keyframe_file"], "frame_000026.jpg")
        self.assertTrue(len(slot["active_application"]) > 0)


class TestWebStaticAssets(unittest.TestCase):
    """Verifies UI files exist and strictly conform to the user's constraints."""

    def test_static_files_exist(self):
        """Ensures index.html, style.css, and app.js are present."""
        self.assertTrue((STATIC_DIR / "index.html").exists())
        self.assertTrue((STATIC_DIR / "style.css").exists())
        self.assertTrue((STATIC_DIR / "app.js").exists())

    def test_no_emojis_constraint(self):
        """
        Strict check: Enforces that zero emojis exist in the website code
        (HTML, CSS, JS) per the user's explicit requirement: 'No emojis on the website.'
        """
        # Emoji unicode ranges
        emoji_pattern = re.compile(
            r"[\U00010000-\U0010ffff"  # Supplementary Multilingual Plane (common emojis)
            r"\u2600-\u26ff"            # Miscellaneous Symbols
            r"\u2700-\u27bf"            # Dingbats
            r"\u2b50\u2b55"             # Stars and circles
            r"\u231a-\u231b"            # Hourglass/watch
            r"\u23e9-\u23ec"            # Fast-forward/rewind
            r"\u23f0-\u23f3"            # Alarm clock, stopwatch
            r"\u2934-\u2935"            # Arrow curving up/down
            r"]"
        )

        for filename in ["index.html", "style.css", "app.js"]:
            filepath = STATIC_DIR / filename
            content = filepath.read_text(encoding="utf-8")
            matches = emoji_pattern.findall(content)
            self.assertEqual(
                len(matches),
                0,
                f"Found forbidden emoji(s) {matches} in {filename}! Rule: No emojis on the website."
            )

    def test_html_accessibility_elements(self):
        """Verifies accessibility elements like aria labels, keyboard shortcuts, roles."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('aria-label="Request facts from the workflow"', html)
        self.assertIn('role="region"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn('Hotkeys:', html)
        self.assertIn('id="omnibar-input"', html)
        self.assertIn('id="scrubber-range"', html)


class TestFactLensRequestHandler(unittest.TestCase):
    """Tests API dispatching logic in FactLensRequestHandler."""

    class MockHandler:
        def __init__(self):
            self.server = type("MockServer", (), {"engines": {}})()
            self.last_json = None
            self.last_status = None
            self.last_error = None

        def get_engine(self, session_id):
            if session_id not in self.server.engines:
                self.server.engines[session_id] = FactEngine(session_id)
            return self.server.engines[session_id]

        def send_json(self, data, status_code=200):
            self.last_json = data
            self.last_status = status_code

        def send_error(self, code, message=""):
            self.last_error = (code, message)

    def setUp(self):
        from src.web.app import FactLensRequestHandler
        self.mock_handler = self.MockHandler()
        self.handle_api = FactLensRequestHandler.handle_api.__get__(self.mock_handler)

    def test_api_sessions(self):
        """Tests /api/sessions route."""
        self.handle_api("/api/sessions", {})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertIn("sessions", self.mock_handler.last_json)
        self.assertIn("solix_demo_test", self.mock_handler.last_json["sessions"])

    def test_api_session_summary(self):
        """Tests /api/sessions/solix_demo_test/summary route."""
        self.handle_api("/api/sessions/solix_demo_test/summary", {})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertEqual(self.mock_handler.last_json["session_id"], "solix_demo_test")

    def test_api_session_facts_bottlenecks(self):
        """Tests /api/sessions/solix_demo_test/facts?category=bottlenecks route."""
        self.handle_api("/api/sessions/solix_demo_test/facts", {"category": ["bottlenecks"]})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertIn("facts", self.mock_handler.last_json)

    def test_api_session_facts_automations(self):
        """Tests /api/sessions/solix_demo_test/facts?category=automations route."""
        self.handle_api("/api/sessions/solix_demo_test/facts", {"category": ["automations"]})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertIn("facts", self.mock_handler.last_json)

    def test_api_session_query(self):
        """Tests /api/sessions/solix_demo_test/query?q=friction route."""
        self.handle_api("/api/sessions/solix_demo_test/query", {"q": ["friction"]})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertGreater(self.mock_handler.last_json["count"], 0)

    def test_api_session_slot(self):
        """Tests /api/sessions/solix_demo_test/slot/20 route."""
        self.handle_api("/api/sessions/solix_demo_test/slot/20", {})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertEqual(self.mock_handler.last_json["timestamp"], 20)

    def test_api_automation_code(self):
        """Tests /api/automations/auto_recording_daemon.py route."""
        self.handle_api("/api/automations/auto_recording_daemon.py", {})
        self.assertIsNotNone(self.mock_handler.last_json)
        self.assertIn("code", self.mock_handler.last_json)
        self.assertIn("PID_FILE", self.mock_handler.last_json["code"])


if __name__ == "__main__":
    unittest.main()
