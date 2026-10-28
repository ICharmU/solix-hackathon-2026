"""
Unit Tests for Local CUDA Neural Pipeline (Models 1 & 2 + Hybrid Tracker)
"""

import json
import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.models.device_manager import get_optimal_device
    from src.models.continuation_net import TaskContinuationNet, compute_continuation_loss
    from src.models.differentiation_net import TaskDifferentiationNet, compute_differentiation_loss
    from src.models.dataset_builder import (
        text_to_embedding,
        ContinuationDataset,
        DifferentiationDataset,
        get_dataloaders,
        OPERATOR_MAP,
        COGNITIVE_MAP
    )
    from src.document_context_engine import (
        build_project_context_for_manifest,
        reclassify_video_under_context,
        DocumentContext,
        ContextualizedManifest
    )
    from src.hybrid_tracker import TaskStack, HybridWorkflowTracker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment required")
class TestNeuralPipeline(unittest.TestCase):

    def setUp(self):
        self.device = get_optimal_device()

    def test_device_discovery(self):
        """Verifies that optimal device returns a valid torch.device."""
        self.assertIsInstance(self.device, torch.device)
        self.assertIn(self.device.type, ["cuda", "cpu"])

    def test_text_to_embedding(self):
        """Verifies pseudo-semantic deterministic embeddings."""
        v1 = text_to_embedding("VS Code schema editing", dim=128)
        v2 = text_to_embedding("VS Code schema editing", dim=128)
        v3 = text_to_embedding("AutoCAD floor plan drafting", dim=128)

        self.assertEqual(v1.shape, (128,))
        np.testing.assert_allclose(v1, v2, rtol=1e-5)
        # Verify unit norm
        self.assertAlmostEqual(float(np.linalg.norm(v1)), 1.0, places=4)
        # Verify distinct texts produce different embeddings
        self.assertFalse(np.allclose(v1, v3))

    def test_task_continuation_net_architecture(self):
        """Tests TaskContinuationNet tensor shapes, forward pass, and predict."""
        model = TaskContinuationNet(d_in=144, hidden_dim=64).to(self.device)
        batch_size = 4
        x = torch.randn(batch_size, 144, device=self.device)

        logits = model(x)
        self.assertEqual(logits.shape, (batch_size, 1))

        probs, is_shift = model.predict_continuation(x, threshold=0.45)
        self.assertEqual(probs.shape, (batch_size,))
        self.assertEqual(is_shift.shape, (batch_size,))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        # Test loss function
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.device)
        loss = compute_continuation_loss(logits, targets)
        self.assertTrue(loss.item() > 0)

    def test_task_differentiation_net_architecture(self):
        """Tests TaskDifferentiationNet cross-attention, fusion, and pointer outputs."""
        model = TaskDifferentiationNet(embed_dim=128, max_stack_depth=8, hidden_dim=64).to(self.device)
        B = 4
        x_curr = torch.randn(B, 128, device=self.device)
        x_lookahead = torch.randn(B, 128, device=self.device)
        x_prior = torch.randn(B, 128, device=self.device)
        stack_mem = torch.randn(B, 8, 128, device=self.device)
        stack_mask = torch.ones(B, 8, device=self.device)

        t_logits, p_logits = model(x_curr, x_lookahead, x_prior, stack_mem, stack_mask)
        self.assertEqual(t_logits.shape, (B, 3))
        self.assertEqual(p_logits.shape, (B, 8))

        # Test prediction helper
        preds = model.predict(x_curr, x_lookahead, x_prior, stack_mem, stack_mask)
        self.assertIn("predicted_class_id", preds)
        self.assertEqual(len(preds["predicted_class_name"]), B)
        self.assertEqual(len(preds["resumption_pointer_index"]), B)

        # Test loss function
        t_targets = torch.tensor([0, 1, 2, 2], device=self.device)
        p_targets = torch.tensor([0, 0, 3, 1], device=self.device)
        loss, ce, ptr = compute_differentiation_loss(t_logits, p_logits, t_targets, p_targets)
        self.assertTrue(loss.item() > 0)

    def test_task_stack_automaton(self):
        """Tests pushdown automaton behavior for task resumptions."""
        stack = TaskStack(max_depth=5)
        v1 = text_to_embedding("task 1", dim=128)
        v2 = text_to_embedding("task 2", dim=128)

        stack.push("seg_01", "TICKET-01", "vscode", "schema_edit", v1)
        stack.push("seg_02", "TICKET-02", "blender", "render_preview", v2)
        self.assertEqual(len(stack.stack), 2)

        # Resume seg_01
        resumed = stack.resume("seg_01")
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed["segment_id"], "seg_01")
        self.assertEqual(len(stack.stack), 1)
        self.assertEqual(stack.stack[0]["segment_id"], "seg_02")

    def test_dataset_loaders(self):
        """Verifies dataset builder builds loaders across all 3 splits."""
        loaders = get_dataloaders(batch_size=16)
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

        # Sample a batch from train
        train_c = loaders["train"]["continuation"]
        x_b, y_b = next(iter(train_c))
        self.assertEqual(x_b.shape[-1], 144)

    def test_document_context_schema(self):
        """Verifies synthetic manifests have valid upfront document context."""
        sample_manifest = {
            "synthetic_id": "test_sample_01",
            "split": "train",
            "total_duration_seconds": 120.0,
            "segments": [
                {"category": "vscode", "segment_index": 1, "duration_s": 60.0},
                {"category": "blender", "segment_index": 2, "duration_s": 60.0}
            ]
        }
        ctx = build_project_context_for_manifest(sample_manifest)
        self.assertIsInstance(ctx, DocumentContext)
        self.assertTrue(len(ctx.active_tickets) >= 2)
        self.assertIn("vscode", ctx.data_schemas)


if __name__ == "__main__":
    unittest.main()
