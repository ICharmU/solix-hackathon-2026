"""
Unit Tests for Agent CLI Action Execution.
Verifies command-line action dispatch, argument parsing, dry-run mode, and target finding.
"""

import unittest
from pathlib import Path
from src.agent.cli import execute_action


class TestAgentCLI(unittest.TestCase):

    def test_cli_dry_run_target_found(self):
        """Verifies dry-run execution on a visible element (e.g. 'File')."""
        # File is visible in VS Code menu bar
        success = execute_action(
            find_text="File",
            action="click",
            dry_run=True,
            use_hotkey=False,
        )
        self.assertTrue(success, "Should find 'File' on screen in dry-run mode.")

    def test_cli_target_not_found(self):
        """Verifies graceful handling when an element is not on screen."""
        success = execute_action(
            find_text="NonExistentTargetText99999XYZ",
            action="click",
            dry_run=True,
            use_hotkey=False,
        )
        self.assertFalse(success, "Should return False when target is not on screen.")

    def test_hotkey_timeout_handling(self):
        """Verifies hotkey timeout gracefully returns False when not pressed."""
        from src.agent.cli import wait_for_hotkey_activation
        res = wait_for_hotkey_activation(timeout_s=0.1)
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()

