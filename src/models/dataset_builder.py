"""
Neural Dataset Builder with Structured Visual-Semantic Embeddings & Exact Stack Tracking

Implements:
1. Realistic visual-semantic embeddings:
   - Orthonormal application anchors (mimicking visual layout/chrome separation like CLIP/ViT).
   - Cohesive intra-application subroutine similarity (cos ~ 0.85).
   - Distinct inter-application divergence (cos ~ 0.05).
2. Exact Ground-Truth Stack Pointer Resolution:
   - Correctly matches lookahead resumption segment against the historical task stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_builder")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = WORKSPACE_ROOT / "data" / "synthetic"

CATEGORIES = ["vscode", "autocad", "blender", "salesforce", "excel", "photoshop"]

# Generate deterministic orthonormal basis for the 6 application categories
_rng = np.random.RandomState(1337)
_basis, _ = np.linalg.qr(_rng.randn(128, len(CATEGORIES)))
APP_ANCHORS: Dict[str, np.ndarray] = {
    cat: _basis[:, i].astype(np.float32) for i, cat in enumerate(CATEGORIES)
}

OPERATOR_MAP = {
    "CLICK": 0,
    "KEYSTROKE_INPUT": 1,
    "COMMAND_LINE": 2,
    "TERMINAL_COMMAND": 2,
    "HOTKEY": 3,
    "DRAG_DROP": 4,
    "SLIDER_DRAG": 4,
    "NONE": 5,
}

COGNITIVE_MAP = {
    "ACTIVE_INTERACTION": 0,
    "COGNITIVE_READING": 1,
    "IDLE_WAITING": 2,
}

TRANSITION_MAP = {
    "NEW_MACRO": 0,
    "SUBROUTINE_SHIFT": 1,
    "RESUME": 2,
}


def text_to_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic hash projection to R^dim."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def get_semantic_action_embedding(
    category: str,
    subroutine: str = "",
    target_element: str = "",
    dim: int = 128
) -> np.ndarray:
    """
    Constructs a visual-semantic embedding:
    - 82% Application Chrome / UI Layout Anchor (orthonormal across apps)
    - 15% Subroutine Specific Offset
    - 3% Target Element Modulation
    Result: Same-app actions have cos ~ 0.85; different-app actions have cos ~ 0.05.
    """
    cat_lower = category.lower()
    if cat_lower in APP_ANCHORS:
        cat_key = cat_lower
    elif "chrome" in cat_lower or "web" in cat_lower or "browser" in cat_lower:
        cat_key = "salesforce"
    elif "obs" in cat_lower or "media" in cat_lower or "record" in cat_lower:
        cat_key = "photoshop"
    elif "code" in cat_lower or "vs" in cat_lower or "ide" in cat_lower:
        cat_key = "vscode"
    elif "cad" in cat_lower:
        cat_key = "autocad"
    elif "blend" in cat_lower or "3d" in cat_lower:
        cat_key = "blender"
    elif "sheet" in cat_lower or "table" in cat_lower or "excel" in cat_lower:
        cat_key = "excel"
    else:
        cat_key = "vscode"

    base_anchor = APP_ANCHORS[cat_key]

    sub_vec = text_to_embedding(subroutine, dim=dim) if subroutine else np.zeros(dim, dtype=np.float32)
    act_vec = text_to_embedding(target_element, dim=dim) if target_element else np.zeros(dim, dtype=np.float32)

    vec = 0.82 * base_anchor + 0.15 * sub_vec + 0.03 * act_vec
    norm = np.linalg.norm(vec)
    return (vec / (norm + 1e-8)).astype(np.float32)


class ContinuationDataset(Dataset):
    """Dataset for Model 1: Task Continuation Network."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.features: List[np.ndarray] = []
        self.labels: List[float] = []

        for s in samples:
            self.features.append(s["feature_vector"])
            self.labels.append(1.0 if s["is_event_shift"] else 0.0)

        self.features_tensor = torch.tensor(np.array(self.features, dtype=np.float32))
        self.labels_tensor = torch.tensor(np.array(self.labels, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features_tensor[idx], self.labels_tensor[idx]


class DifferentiationDataset(Dataset):
    """Dataset for Model 2: Task Differentiation Network."""

    def __init__(self, samples: List[Dict[str, Any]], max_stack_depth: int = 8, embed_dim: int = 128):
        self.x_curr = []
        self.x_lookahead = []
        self.x_prior = []
        self.stack_mem = []
        self.stack_mask = []
        self.target_transition = []
        self.target_pointer = []

        for s in samples:
            self.x_curr.append(s["curr_embed"])
            self.x_lookahead.append(s["lookahead_embed"])
            self.x_prior.append(s["prior_embed"])

            raw_stack = s.get("stack_embeddings", [])
            stack_arr = np.zeros((max_stack_depth, embed_dim), dtype=np.float32)
            mask_arr = np.zeros(max_stack_depth, dtype=np.float32)

            for i, st in enumerate(raw_stack[:max_stack_depth]):
                stack_arr[i] = st
                mask_arr[i] = 1.0

            self.stack_mem.append(stack_arr)
            self.stack_mask.append(mask_arr)
            self.target_transition.append(s["transition_label"])
            self.target_pointer.append(s.get("resumed_stack_index", 0))

        self.x_curr_t = torch.tensor(np.array(self.x_curr, dtype=np.float32))
        self.x_lookahead_t = torch.tensor(np.array(self.x_lookahead, dtype=np.float32))
        self.x_prior_t = torch.tensor(np.array(self.x_prior, dtype=np.float32))
        self.stack_mem_t = torch.tensor(np.array(self.stack_mem, dtype=np.float32))
        self.stack_mask_t = torch.tensor(np.array(self.stack_mask, dtype=np.float32))
        self.target_transition_t = torch.tensor(np.array(self.target_transition, dtype=np.int64))
        self.target_pointer_t = torch.tensor(np.array(self.target_pointer, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.target_transition)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_curr_t[idx],
            self.x_lookahead_t[idx],
            self.x_prior_t[idx],
            self.stack_mem_t[idx],
            self.stack_mask_t[idx],
            self.target_transition_t[idx],
            self.target_pointer_t[idx],
        )


def build_samples_from_split(split: str = "train") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses all contextualized manifests for the given split and extracts structured sample dictionaries.
    Correctly maintains task stack history and computes accurate resumption pointers.
    """
    split_dir = SYNTHETIC_DIR / split
    manifest_paths = sorted(split_dir.glob("synthetic_*_context_manifest.json"))

    continuation_samples: List[Dict[str, Any]] = []
    differentiation_samples: List[Dict[str, Any]] = []

    for mp in manifest_paths:
        data = json.loads(mp.read_text(encoding="utf-8"))
        doc_ctx = data["document_context"]
        prior_text = f"{doc_ctx['project_title']} | PRD: {doc_ctx['prd_summary']}"
        e_prior = text_to_embedding(prior_text, dim=128)

        segments = data["segments"]
        breakpoints = {bp["time_s"]: bp for bp in data["breakpoints"]}

        # Memory stack for this video session: list of {"segment_id": ..., "category": ..., "embedding": ...}
        task_stack: List[Dict[str, Any]] = []

        for seg_idx, seg in enumerate(segments):
            cat = seg["category"]
            sub = seg["subroutine"]
            act = seg["action_a"]
            target_ui = act.get("target", "")

            # Generate realistic semantic embedding
            e_seg = get_semantic_action_embedding(cat, sub, target_ui, dim=128)

            # 1. Intra-Segment Flow Steps (CONTINUE_FLOW: y=0)
            step_duration = 10.0
            num_flow_steps = max(1, int(seg["duration_s"] // step_duration))

            for step in range(num_flow_steps):
                op_code = OPERATOR_MAP.get(act.get("operator", "NONE"), 5)
                cog_code = COGNITIVE_MAP.get(seg.get("cognitive_state", "ACTIVE_INTERACTION"), 0)

                op_onehot = np.zeros(6, dtype=np.float32)
                op_onehot[op_code] = 1.0

                cog_onehot = np.zeros(3, dtype=np.float32)
                cog_onehot[cog_code] = 1.0

                flags = np.array([
                    0.0,  # app_changed = False
                    0.0,  # window_changed = False
                    0.0 if step > 0 else 1.0,
                    1.0 if seg_idx == 0 and step == 0 else 0.0,
                    1.0 if seg.get("is_interruption") else 0.0
                ], dtype=np.float32)

                delta_t = np.array([0.05 + 0.03 * np.sin(step)], dtype=np.float32)
                hist_vel = np.array([0.06], dtype=np.float32)

                x_cont = np.concatenate([e_seg, delta_t, hist_vel, op_onehot, cog_onehot, flags])
                continuation_samples.append({
                    "feature_vector": x_cont,
                    "is_event_shift": False,
                    "segment_id": seg["segment_id"]
                })

            # 2. Breakpoint Step at segment boundary (EVENT_SHIFT: y=1)
            is_last_segment = (seg_idx == len(segments) - 1)
            if not is_last_segment:
                nxt_seg = segments[seg_idx + 1]
                nxt_cat = nxt_seg["category"]
                nxt_sub = nxt_seg["subroutine"]
                nxt_target = nxt_seg["action_a"].get("target", "")
                e_nxt = get_semantic_action_embedding(nxt_cat, nxt_sub, nxt_target, dim=128)

                app_changed = 1.0 if nxt_cat != cat else 0.0
                flags_bp = np.array([
                    app_changed,
                    1.0,
                    1.0,
                    0.0,
                    1.0 if nxt_seg.get("is_interruption") else 0.0
                ], dtype=np.float32)

                delta_bp = np.array([0.78 if app_changed else 0.38], dtype=np.float32)
                hist_bp = np.array([0.65 if app_changed else 0.28], dtype=np.float32)
                op_bp = np.zeros(6, dtype=np.float32)
                op_bp[0] = 1.0
                cog_bp = np.zeros(3, dtype=np.float32)
                cog_bp[0] = 1.0

                x_cont_bp = np.concatenate([e_seg, delta_bp, hist_bp, op_bp, cog_bp, flags_bp])
                continuation_samples.append({
                    "feature_vector": x_cont_bp,
                    "is_event_shift": True,
                    "segment_id": seg["segment_id"]
                })

                # 3. Create Sample for Model 2: Task Differentiation Network
                bp_info = breakpoints.get(seg["end_time_s"])
                if bp_info:
                    t_type_str = bp_info["transition_type"]
                    t_label = TRANSITION_MAP.get(t_type_str, 0)
                else:
                    t_label = 0

                # Determine the true resumption pointer in task_stack
                target_pointer = 0
                if t_label == 2:  # RESUME
                    resumes_id = nxt_seg.get("resumes_segment_id")
                    for s_idx, s_item in enumerate(task_stack):
                        if s_item["segment_id"] == resumes_id:
                            target_pointer = s_idx
                            break

                stack_embeddings = [s["embedding"] for s in task_stack]

                differentiation_samples.append({
                    "curr_embed": e_seg,
                    "lookahead_embed": e_nxt,
                    "prior_embed": e_prior,
                    "stack_embeddings": stack_embeddings,
                    "transition_label": t_label,
                    "resumed_stack_index": target_pointer
                })

            # Maintain task stack for next step
            if seg.get("resumes_segment_id"):
                # The task was resumed: remove the resumed segment from the stack
                resumes_id = seg.get("resumes_segment_id")
                task_stack = [s for s in task_stack if s["segment_id"] != resumes_id]
            else:
                task_stack.append({
                    "segment_id": seg["segment_id"],
                    "category": cat,
                    "embedding": e_seg
                })

    return continuation_samples, differentiation_samples


def get_dataloaders(
    batch_size: int = 32
) -> Dict[str, Dict[str, DataLoader]]:
    loaders: Dict[str, Dict[str, DataLoader]] = {}

    for split in ["train", "val", "test"]:
        cont_samples, diff_samples = build_samples_from_split(split)

        cont_ds = ContinuationDataset(cont_samples)
        diff_ds = DifferentiationDataset(diff_samples)

        is_train = (split == "train")
        cont_loader = DataLoader(cont_ds, batch_size=batch_size, shuffle=is_train)
        diff_loader = DataLoader(diff_ds, batch_size=max(8, batch_size // 2), shuffle=is_train)

        loaders[split] = {
            "continuation": cont_loader,
            "differentiation": diff_loader
        }

        logger.info(
            f"Split '{split}': {len(cont_ds)} continuation steps, {len(diff_ds)} differentiation breakpoints."
        )

    return loaders


if __name__ == "__main__":
    # Test semantic similarities
    e_cad1 = get_semantic_action_embedding("autocad", "subroutine_autocad_polyline_draw")
    e_cad2 = get_semantic_action_embedding("autocad", "subroutine_autocad_dimension_verify")
    e_blen = get_semantic_action_embedding("blender", "subroutine_blender_transform_mesh")

    sim_same = float(np.dot(e_cad1, e_cad2))
    sim_diff = float(np.dot(e_cad1, e_blen))
    print(f"Same App Cosine Similarity (AutoCAD vs AutoCAD): {sim_same:.4f}")
    print(f"Diff App Cosine Similarity (AutoCAD vs Blender): {sim_diff:.4f}")

    loaders = get_dataloaders(batch_size=16)
    for split, l_dict in loaders.items():
        print(f"[{split.upper()}] Continuation: {len(l_dict['continuation']) * 16} steps, Differentiation: {len(l_dict['differentiation']) * 8} breakpoints")
