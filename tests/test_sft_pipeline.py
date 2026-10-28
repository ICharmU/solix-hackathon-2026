"""
Unit Tests for SFT Dataset Preparation and Pre-Flight Validation.
"""

import json
import unittest
from pathlib import Path
import tiktoken

from src.config import BASE_DIR
from src.run_finetune import validate_jsonl_file, run_validation_and_cost_audit


class TestSFTPipeline(unittest.TestCase):
    """Test suite covering SFT dataset preparation and OpenAI pre-flight audit."""

    def setUp(self):
        self.sft_dir = BASE_DIR / "data" / "sft"
        self.train_file = self.sft_dir / "train_sft.jsonl"
        self.val_file = self.sft_dir / "val_sft.jsonl"
        self.enc = tiktoken.get_encoding("o200k_base")

    def test_jsonl_files_exist_and_non_empty(self):
        """Verifies that train and val JSONL files exist and have data."""
        self.assertTrue(self.train_file.exists(), f"Missing {self.train_file}")
        self.assertTrue(self.val_file.exists(), f"Missing {self.val_file}")
        self.assertGreater(self.train_file.stat().st_size, 5000)
        self.assertGreater(self.val_file.stat().st_size, 1000)

    def test_train_sample_count(self):
        """Ensures exactly 250 training samples and 50 validation samples."""
        train_lines = [l for l in self.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        val_lines = [l for l in self.val_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(train_lines), 250)
        self.assertEqual(len(val_lines), 50)

    def test_jsonl_schema_conformance(self):
        """Ensures valid messages schema and required tokens."""
        is_valid, sample_count, total_tokens, errors = validate_jsonl_file(self.train_file, self.enc)
        self.assertTrue(is_valid, f"Validation errors in train file: {errors[:5]}")
        self.assertEqual(sample_count, 250)
        self.assertGreater(total_tokens, 50000)

    def test_action_triple_and_lookahead_present(self):
        """Verifies (S0, A, S1) triple and lookahead buffer in training items."""
        with open(self.train_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                user_content = item["messages"][1]["content"]
                self.assertIn("[CURRENT_ACTION_OBSERVATION]", user_content)
                self.assertIn("Initial State (S0):", user_content)
                self.assertIn("Action (A):", user_content)
                self.assertIn("Outcome State (S1):", user_content)
                self.assertIn("[LOOKAHEAD_BUFFER", user_content)
                if i >= 10:
                    break

    def test_cost_audit_under_three_dollars(self):
        """Verifies that total estimated training cost with o200k_base is strictly < $3.00."""
        report = run_validation_and_cost_audit(epochs=3)
        self.assertTrue(report["is_ready_for_submission"])
        self.assertLess(report["estimated_cost_usd"], 3.00)
        self.assertEqual(report["tokenizer"], "o200k_base")


if __name__ == "__main__":
    unittest.main()

