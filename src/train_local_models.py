"""
CUDA Training Pipeline for Task Continuation & Task Differentiation Models

Trains two separate local neural networks on NVIDIA CUDA:
1. Model 1 (TaskContinuationNet): Flow continuity vs. candidate breakpoint detection.
2. Model 2 (TaskDifferentiationNet): Macro vs. Subroutine vs. Resume classification & stack pointer resolution.
Evaluates on Validation and Test splits, saving best checkpoints and metric reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.device_manager import get_optimal_device
from src.models.continuation_net import TaskContinuationNet, compute_continuation_loss
from src.models.differentiation_net import TaskDifferentiationNet, compute_differentiation_loss
from src.models.dataset_builder import get_dataloaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("train_local_models")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = WORKSPACE_ROOT / "data" / "checkpoints"


def train_continuation_model(
    loaders: Dict[str, Any],
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
    hidden_dim: int = 256
) -> Tuple[TaskContinuationNet, Dict[str, Any]]:
    """
    Trains Model 1: TaskContinuationNet on CUDA.
    """
    logger.info(f"--- Starting Training: Model 1 (TaskContinuationNet) on {device} ---")
    train_loader = loaders["train"]["continuation"]
    val_loader = loaders["val"]["continuation"]

    model = TaskContinuationNet(d_in=144, hidden_dim=hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch).squeeze(-1)
            loss = compute_continuation_loss(logits, y_batch, pos_weight=3.5)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)

        scheduler.step()
        train_loss /= max(1, total_samples)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch).squeeze(-1)
                loss = compute_continuation_loss(logits, y_batch, pos_weight=3.5)
                val_loss += loss.item() * len(y_batch)
                val_samples += len(y_batch)

                probs = torch.sigmoid(logits)
                # target: 1 is EVENT_SHIFT, 0 is CONTINUE_FLOW
                # prob is continuation prob: shift if prob < 0.45
                predicted_shifts = (probs < 0.45).long()
                all_preds.extend(predicted_shifts.cpu().numpy().tolist())
                all_targets.extend(y_batch.long().cpu().numpy().tolist())

        val_loss /= max(1, val_samples)
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)

        accuracy = float(np.mean(preds_arr == targets_arr))
        # Metrics for EVENT_SHIFT (class 1)
        tp = float(np.sum((preds_arr == 1) & (targets_arr == 1)))
        fp = float(np.sum((preds_arr == 1) & (targets_arr == 0)))
        fn = float(np.sum((preds_arr == 0) & (targets_arr == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs or val_loss < best_val_loss:
            logger.info(
                f"[ContNet Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {accuracy * 100:.1f}% | BP Recall: {recall * 100:.1f}% | BP F1: {f1 * 100:.1f}%"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "breakpoint_precision": precision,
                "breakpoint_recall": recall,
                "breakpoint_f1": f1
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def train_differentiation_model(
    loaders: Dict[str, Any],
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden_dim: int = 64
) -> Tuple[TaskDifferentiationNet, Dict[str, Any]]:
    """
    Trains Model 2: TaskDifferentiationNet on CUDA.
    """
    logger.info(f"--- Starting Training: Model 2 (TaskDifferentiationNet) on {device} ---")
    train_loader = loaders["train"]["differentiation"]
    val_loader = loaders["val"]["differentiation"]

    model = TaskDifferentiationNet(embed_dim=128, max_stack_depth=8, hidden_dim=hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        total_samples = 0

        for x_c, x_l, x_p, sm, mask, targets, ptr_targets in train_loader:
            x_c = x_c.to(device)
            x_l = x_l.to(device)
            x_p = x_p.to(device)
            sm = sm.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            ptr_targets = ptr_targets.to(device)

            optimizer.zero_grad()
            t_logits, p_logits = model(x_c, x_l, x_p, sm, mask)
            loss, ce_l, ptr_l = compute_differentiation_loss(t_logits, p_logits, targets, ptr_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            total_samples += len(targets)

        scheduler.step()
        train_loss /= max(1, total_samples)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []
        pointer_correct = 0
        total_resume = 0

        with torch.no_grad():
            for x_c, x_l, x_p, sm, mask, targets, ptr_targets in val_loader:
                x_c = x_c.to(device)
                x_l = x_l.to(device)
                x_p = x_p.to(device)
                sm = sm.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                ptr_targets = ptr_targets.to(device)

                t_logits, p_logits = model(x_c, x_l, x_p, sm, mask)
                loss, _, _ = compute_differentiation_loss(t_logits, p_logits, targets, ptr_targets)
                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

                preds = torch.argmax(t_logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

                # Check pointer accuracy for RESUME
                resume_mask = (targets == 2)
                if resume_mask.any():
                    pred_ptrs = torch.argmax(p_logits[resume_mask], dim=-1)
                    true_ptrs = ptr_targets[resume_mask]
                    pointer_correct += int((pred_ptrs == true_ptrs).sum().item())
                    total_resume += int(resume_mask.sum().item())

        val_loss /= max(1, val_samples)
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
        ptr_acc = (pointer_correct / total_resume) if total_resume > 0 else 1.0

        if epoch % 5 == 0 or epoch == epochs or val_loss < best_val_loss:
            logger.info(
                f"[DiffNet Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {accuracy * 100:.1f}% | Pointer Acc: {ptr_acc * 100:.1f}%"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "resumption_pointer_accuracy": ptr_acc
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def evaluate_on_test_set(
    cont_model: TaskContinuationNet,
    diff_model: TaskDifferentiationNet,
    loaders: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates both trained models on the held-out test split (5 videos).
    """
    logger.info("--- Evaluating Models on Held-out Test Set (5 Videos) ---")
    cont_loader = loaders["test"]["continuation"]
    diff_loader = loaders["test"]["differentiation"]

    cont_model.eval()
    all_c_preds = []
    all_c_targets = []

    with torch.no_grad():
        for x_b, y_b in cont_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            logits = cont_model(x_b).squeeze(-1)
            probs = torch.sigmoid(logits)
            shifts = (probs < 0.45).long()
            all_c_preds.extend(shifts.cpu().numpy().tolist())
            all_c_targets.extend(y_b.long().cpu().numpy().tolist())

    c_preds_arr = np.array(all_c_preds)
    c_targets_arr = np.array(all_c_targets)
    c_acc = float(np.mean(c_preds_arr == c_targets_arr))
    tp = float(np.sum((c_preds_arr == 1) & (c_targets_arr == 1)))
    fp = float(np.sum((c_preds_arr == 1) & (c_targets_arr == 0)))
    fn = float(np.sum((c_preds_arr == 0) & (c_targets_arr == 1)))
    c_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    c_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

    diff_model.eval()
    all_d_preds = []
    all_d_targets = []
    ptr_correct = 0
    total_resume = 0

    with torch.no_grad():
        for x_c, x_l, x_p, sm, mask, targets, ptr_targets in diff_loader:
            x_c = x_c.to(device)
            x_l = x_l.to(device)
            x_p = x_p.to(device)
            sm = sm.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            ptr_targets = ptr_targets.to(device)

            t_logits, p_logits = diff_model(x_c, x_l, x_p, sm, mask)
            preds = torch.argmax(t_logits, dim=-1)
            all_d_preds.extend(preds.cpu().numpy().tolist())
            all_d_targets.extend(targets.cpu().numpy().tolist())

            resume_mask = (targets == 2)
            if resume_mask.any():
                p_preds = torch.argmax(p_logits[resume_mask], dim=-1)
                p_true = ptr_targets[resume_mask]
                ptr_correct += int((p_preds == p_true).sum().item())
                total_resume += int(resume_mask.sum().item())

    d_acc = float(np.mean(np.array(all_d_preds) == np.array(all_d_targets)))
    d_ptr_acc = (ptr_correct / total_resume) if total_resume > 0 else 1.0

    return {
        "test_continuation": {
            "total_steps": len(all_c_targets),
            "accuracy": c_acc,
            "breakpoint_precision": c_prec,
            "breakpoint_recall": c_rec,
            "breakpoint_f1": c_f1
        },
        "test_differentiation": {
            "total_breakpoints": len(all_d_targets),
            "accuracy": d_acc,
            "resumption_pointer_accuracy": d_ptr_acc
        }
    }


def run_pipeline() -> Dict[str, Any]:
    """
    Main training pipeline execution.
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_optimal_device()
    loaders = get_dataloaders(batch_size=32)

    # 1. Train Model 1
    cont_model, cont_val_metrics = train_continuation_model(loaders, device, epochs=35, lr=1e-3)
    cont_ckpt = CHECKPOINTS_DIR / "task_continuation_net.pt"
    torch.save(cont_model.state_dict(), cont_ckpt)
    logger.info(f"[SAVED] TaskContinuationNet checkpoint to {cont_ckpt}")

    # 2. Train Model 2
    diff_model, diff_val_metrics = train_differentiation_model(loaders, device, epochs=40, lr=1e-3)
    diff_ckpt = CHECKPOINTS_DIR / "task_differentiation_net.pt"
    torch.save(diff_model.state_dict(), diff_ckpt)
    logger.info(f"[SAVED] TaskDifferentiationNet checkpoint to {diff_ckpt}")

    # 3. Test Evaluation
    test_metrics = evaluate_on_test_set(cont_model, diff_model, loaders, device)

    report: Dict[str, Any] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "model_1_continuation": {
            "checkpoint": str(cont_ckpt),
            "validation_metrics": cont_val_metrics,
            "test_metrics": test_metrics["test_continuation"]
        },
        "model_2_differentiation": {
            "checkpoint": str(diff_ckpt),
            "validation_metrics": diff_val_metrics,
            "test_metrics": test_metrics["test_differentiation"]
        }
    }

    report_path = CHECKPOINTS_DIR / "neural_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"[SAVED] Evaluation Report to {report_path}")

    return report


if __name__ == "__main__":
    report = run_pipeline()
    print(json.dumps(report, indent=2))
