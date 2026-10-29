"""
Unit Tests for Desktop Macro Engine (WoW-style Macro Automation).
Verifies:
1. Macro parsing from text and files.
2. Step parsing for /click, /type, /press, /wait, /assert.
3. Serialization back to plain text.
4. MacroRunner dry-run execution and step failure diagnostics.
"""

import unittest
from pathlib import Path

from src.agent.macro import Macro, MacroStep, MacroRunner
from src.agent.safety import SafetyMonitor


class TestDesktopMacroEngine(unittest.TestCase):

    def test_parse_macro_text(self):
        """Verifies parsing plain-text macro DSL lines into MacroSteps."""
        raw_text = """
        # Sample macro script
        /click "Terminal"
        /doubleclick "File.txt"
        /type "echo hello"
        /press enter
        /wait 1.5s
        /assert "hello"
        """
        macro = Macro.from_text(raw_text, name="test_dsl")
        self.assertEqual(len(macro.steps), 6)

        self.assertEqual(macro.steps[0].command, "click")
        self.assertEqual(macro.steps[0].argument, "Terminal")

        self.assertEqual(macro.steps[1].command, "doubleclick")
        self.assertEqual(macro.steps[1].argument, "File.txt")

        self.assertEqual(macro.steps[2].command, "type")
        self.assertEqual(macro.steps[2].argument, "echo hello")

        self.assertEqual(macro.steps[3].command, "press")
        self.assertEqual(macro.steps[3].argument, "enter")

        self.assertEqual(macro.steps[4].command, "wait")
        self.assertEqual(macro.steps[4].argument, "1.5s")

        self.assertEqual(macro.steps[5].command, "assert")
        self.assertEqual(macro.steps[5].argument, "hello")

    def test_macro_serialization(self):
        """Verifies round-trip serialization between Macro object and plain text."""
        raw_text = '/click "File"\n/press ctrl+s\n'
        macro = Macro.from_text(raw_text, name="save_file")
        serialized = macro.to_text()

        self.assertIn('/click "File"', serialized)
        self.assertIn("/press ctrl+s", serialized)

    def test_duration_parsing(self):
        """Verifies parsing seconds, milliseconds, and raw floats."""
        self.assertAlmostEqual(MacroRunner._parse_duration("1.5s"), 1.5)
        self.assertAlmostEqual(MacroRunner._parse_duration("500ms"), 0.5)
        self.assertAlmostEqual(MacroRunner._parse_duration("2"), 2.0)

    def test_macro_runner_dry_run(self):
        """Verifies dry-run execution on a multi-step macro without physical side effects."""
        raw_text = """
        /wait 100ms
        /press ctrl+s
        """
        macro = Macro.from_text(raw_text, name="dry_run_test")
        runner = MacroRunner(safety_monitor=SafetyMonitor(enable_dialog=False, mouse_threshold_px=9999))

        # Run immediately in dry-run mode
        success = runner.run(macro, use_hotkey=False, dry_run=True)
        self.assertTrue(success, "Dry run on wait and press commands should succeed.")

    def test_macro_runner_failure_handling(self):
        """Verifies that non-existent targets halt execution and report failure."""
        raw_text = """
        /click "NonExistentWidget999XYZ"
        """
        macro = Macro.from_text(raw_text, name="fail_test")
        runner = MacroRunner(safety_monitor=SafetyMonitor(enable_dialog=False, mouse_threshold_px=9999))

        success = runner.run(macro, use_hotkey=False, dry_run=True)
        self.assertFalse(success, "Macro should report failure when target is missing.")

    def test_macro_formulator_from_query(self):
        """Verifies formulating a macro from a natural language query."""
        from src.agent.macro_builder import MacroFormulator
        formulator = MacroFormulator()
        macro = formulator.formulate_macro(
            query="click on ocr_engine.py, format code and save",
            name="test_formulated",
            inspect_live_screen=False,
        )
        self.assertIsNotNone(macro)
        commands = [s.command for s in macro.steps]
        self.assertIn("click", commands)
        self.assertTrue(any("ocr_engine.py" in s.argument for s in macro.steps if s.command == "click"))
        self.assertIn("press", commands)


if __name__ == "__main__":
    unittest.main()
