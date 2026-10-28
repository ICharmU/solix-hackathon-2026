"""
Hybrid Workflow Tracker: Local CUDA Neural Predictors + LLM Interpretability

Orchestrates:
1. Fast local inference on CUDA:
   - TaskContinuationNet evaluates continuous steps (flow vs. candidate breakpoint).
   - TaskDifferentiationNet arbitrates transition type (NEW_MACRO, SUBROUTINE_SHIFT, RESUME)
     and resolves task stack pointers.
2. Formal Python Task Stack maintaining active, nested, and interrupted task states.
3. LLM Interpretability Layer (gpt-4o-mini) translating mathematical transitions into
   rich, PRD-linked explanations and automation blueprints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.models.device_manager import get_optimal_device
from src.models.continuation_net import TaskContinuationNet
from src.models.differentiation_net import TaskDifferentiationNet
from src.models.dataset_builder import (
    text_to_embedding,
    get_semantic_action_embedding,
    OPERATOR_MAP,
    COGNITIVE_MAP
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid_tracker")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = WORKSPACE_ROOT / "data" / "checkpoints"


class TaskStack:
    """
    Formal pushdown automaton tracking active and interrupted task segments.
    Guarantees 100% state consistency and prevents ID hallucinations.
    """

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.stack: List[Dict[str, Any]] = []

    def push(self, segment_id: str, ticket_id: str, category: str, subroutine: str, embedding: np.ndarray) -> None:
        self.stack.append({
            "segment_id": segment_id,
            "ticket_id": ticket_id,
            "category": category,
            "subroutine": subroutine,
            "embedding": embedding
        })
        if len(self.stack) > self.max_depth:
            self.stack.pop(0)

    def resume(self, target_segment_id: str) -> Optional[Dict[str, Any]]:
        for i, item in enumerate(reversed(self.stack)):
            if item["segment_id"] == target_segment_id:
                real_idx = len(self.stack) - 1 - i
                return self.stack.pop(real_idx)
        return None

    def get_stack_tensors(self, device: torch.device, embed_dim: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        stack_arr = np.zeros((1, self.max_depth, embed_dim), dtype=np.float32)
        mask_arr = np.zeros((1, self.max_depth), dtype=np.float32)

        for i, item in enumerate(self.stack[:self.max_depth]):
            stack_arr[0, i] = item["embedding"]
            mask_arr[0, i] = 1.0

        return torch.tensor(stack_arr, device=device), torch.tensor(mask_arr, device=device)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [{k: v for k, v in item.items() if k != "embedding"} for item in self.stack]


class HybridWorkflowTracker:
    """
    Combines CUDA-accelerated neural networks with an LLM interpretability layer.
    """

    def __init__(
        self,
        cont_ckpt: Optional[Path] = None,
        diff_ckpt: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_optimal_device()
        cont_path = cont_ckpt or (CHECKPOINTS_DIR / "task_continuation_net.pt")
        diff_path = diff_ckpt or (CHECKPOINTS_DIR / "task_differentiation_net.pt")

        # Load Model 1
        self.cont_net = TaskContinuationNet(d_in=144).to(self.device)
        if cont_path.exists():
            self.cont_net.load_state_dict(torch.load(cont_path, map_location=self.device))
            self.cont_net.eval()
            logger.info(f"Loaded TaskContinuationNet from {cont_path}")

        # Load Model 2
        self.diff_net = TaskDifferentiationNet(embed_dim=128, max_stack_depth=8, hidden_dim=64).to(self.device)
        if diff_path.exists():
            self.diff_net.load_state_dict(torch.load(diff_path, map_location=self.device))
            self.diff_net.eval()
            logger.info(f"Loaded TaskDifferentiationNet from {diff_path}")

        self.stack = TaskStack(max_depth=8)
        self.class_names = ["NEW_MACRO", "SUBROUTINE_SHIFT", "RESUME"]

    def evaluate_step(
        self,
        current_action: Dict[str, Any],
        lookahead_action: Optional[Dict[str, Any]],
        document_context: Dict[str, Any],
        delta_t: float = 0.08,
        hist_vel: float = 0.10,
        app_changed: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a single workflow step using CUDA inference and task stack logic.
        """
        cat = current_action.get("category", "vscode")
        sub = current_action.get("subroutine", "unknown")
        act = current_action.get("action_a") or current_action.get("action") or {}
        seg_id = current_action.get("segment_id", "seg_01")
        ticket_id = current_action.get("governing_ticket_id", "TICKET-01")

        target_ui = act.get("target", "")
        e_curr = get_semantic_action_embedding(cat, sub, target_ui, dim=128)

        # 1. Evaluate Model 1: Task Continuation Network
        op_code = OPERATOR_MAP.get(act.get("operator", "NONE"), 5)
        cog_code = COGNITIVE_MAP.get(current_action.get("cognitive_state", "ACTIVE_INTERACTION"), 0)
        op_onehot = np.zeros(6, dtype=np.float32)
        op_onehot[op_code] = 1.0
        cog_onehot = np.zeros(3, dtype=np.float32)
        cog_onehot[cog_code] = 1.0
        flags = np.array([1.0 if app_changed else 0.0, 1.0 if app_changed else 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        x_cont = np.concatenate([e_curr, np.array([delta_t], dtype=np.float32), np.array([hist_vel], dtype=np.float32), op_onehot, cog_onehot, flags])
        x_cont_t = torch.tensor(x_cont, dtype=torch.float32, device=self.device).unsqueeze(0)

        prob_cont, is_shift = self.cont_net.predict_continuation(x_cont_t, threshold=0.45)
        p_val = float(prob_cont.item())
        shift_val = bool(is_shift.item())

        result: Dict[str, Any] = {
            "continuation_probability": round(p_val, 4),
            "is_event_shift": shift_val,
            "active_stack_depth": len(self.stack.stack)
        }

        # If continuous flow: return immediately
        if not shift_val:
            result["semantic_event"] = "CONTINUE_FLOW"
            result["transition_type"] = "NONE"
            return result

        # 2. Breakpoint Triggered: Evaluate Model 2 (Task Differentiation Network)
        result["semantic_event"] = "EVENT_SHIFT"

        # Prepare lookahead
        if lookahead_action:
            nxt_cat = lookahead_action.get("category", cat)
            nxt_act = lookahead_action.get("action_a") or lookahead_action.get("action") or {}
            nxt_sub = lookahead_action.get("subroutine", "")
            nxt_target = nxt_act.get("target", "")
            e_look = get_semantic_action_embedding(nxt_cat, nxt_sub, nxt_target, dim=128)
        else:
            e_look = e_curr.copy()

        # Prior context
        prior_text = f"{document_context.get('project_title', '')} | {document_context.get('prd_summary', '')}"
        e_prior = text_to_embedding(prior_text, dim=128)

        # Convert to CUDA tensors
        x_c_t = torch.tensor(e_curr, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_l_t = torch.tensor(e_look, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_p_t = torch.tensor(e_prior, dtype=torch.float32, device=self.device).unsqueeze(0)
        sm_t, mask_t = self.stack.get_stack_tensors(self.device)

        diff_preds = self.diff_net.predict(x_c_t, x_l_t, x_p_t, sm_t, mask_t)
        t_type = diff_preds["predicted_class_name"][0]
        t_probs = diff_preds["transition_probabilities"][0]
        ptr_idx = diff_preds["resumption_pointer_index"][0]

        result["transition_type"] = t_type
        result["transition_confidences"] = {
            "NEW_MACRO": round(t_probs[0], 4),
            "SUBROUTINE_SHIFT": round(t_probs[1], 4),
            "RESUME": round(t_probs[2], 4)
        }

        # Stack resolution
        if t_type == "RESUME" and len(self.stack.stack) > 0:
            target_idx = min(ptr_idx, len(self.stack.stack) - 1)
            resumed_item = self.stack.stack[target_idx]
            result["resumes_segment_id"] = resumed_item["segment_id"]
            result["resumes_ticket_id"] = resumed_item["ticket_id"]
            self.stack.resume(resumed_item["segment_id"])
        elif t_type == "NEW_MACRO":
            # Push current segment onto stack before switching
            self.stack.push(seg_id, ticket_id, cat, sub, e_curr)
            result["resumes_segment_id"] = None
        else:
            result["resumes_segment_id"] = None

        return result

    def generate_llm_interpretability_context(
        self,
        step_result: Dict[str, Any],
        current_action: Dict[str, Any],
        document_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Formats a clean, structured context packet that can be fed into gpt-4o-mini
        for plain-text explainability, Mermaid workflow generation, and PRD reconciliation.
        """
        ticket_id = current_action.get("governing_ticket_id", "")
        ticket_title = current_action.get("governing_ticket_title", "")

        return {
            "project_name": document_context.get("project_title"),
            "active_ticket": f"{ticket_id}: {ticket_title}",
            "neural_signals": {
                "event_shift": step_result["is_event_shift"],
                "continuation_probability": step_result["continuation_probability"],
                "predicted_transition": step_result["transition_type"],
                "transition_confidences": step_result.get("transition_confidences", {}),
                "resumes_segment": step_result.get("resumes_segment_id"),
                "active_stack_depth": step_result["active_stack_depth"]
            },
            "observed_action_triple": {
                "initial_state_s0": current_action.get("initial_state_s0", {}),
                "action_a": current_action.get("action_a", {}),
                "outcome_state_s1": current_action.get("outcome_state_s1", {}),
                "cognitive_state": current_action.get("cognitive_state", "ACTIVE_INTERACTION")
            },
            "task_stack_memory": self.stack.to_dict_list()
        }


if __name__ == "__main__":
    tracker = HybridWorkflowTracker()
    test_context = {
        "project_title": "Enterprise CAD & 3D Pipeline",
        "prd_summary": "Draft architectural walls and render 3D assets"
    }
    sample_action = {
        "segment_id": "seg_01",
        "governing_ticket_id": "TICKET-ACAD-01",
        "governing_ticket_title": "AutoCAD Wall Drafting",
        "category": "autocad",
        "subroutine": "subroutine_autocad_polyline_draw",
        "action_a": {"operator": "COMMAND_LINE", "target": "Command Prompt", "payload": "PLINE"}
    }
    res = tracker.evaluate_step(sample_action, lookahead_action=None, document_context=test_context, delta_t=0.65, app_changed=True)
    print("Hybrid Tracker Output:")
    print(json.dumps(res, indent=2))
    context_pack = tracker.generate_llm_interpretability_context(res, sample_action, test_context)
    print("\nLLM Interpretability Context Packet:")
    print(json.dumps(context_pack, indent=2))
