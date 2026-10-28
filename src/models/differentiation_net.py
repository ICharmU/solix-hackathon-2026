"""
Task Differentiation Network (Model 2) with Inductive Geometric Cross-Attention

Classifies transition types (NEW_MACRO, SUBROUTINE_SHIFT, RESUME) and resolves
exact stack resumption pointers using scaled cross-attention between lookahead
actions and historical task stack memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class TaskDifferentiationNet(nn.Module):
    """
    Inductive Cross-Attention Network for Task Differentiation & Stack Pointer Resolution.
    Leverages visual-semantic cosine geometries:
    - Intra-application continuation / subroutine shift (cos ~ 0.95)
    - Stack memory recurrence / task resumption (cos ~ 0.95 with stack item)
    - Inter-application macro switches (cos ~ 0.01)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        max_stack_depth: int = 8,
        hidden_dim: int = 64,
        temperature: float = 0.15,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_stack_depth = max_stack_depth
        self.hidden_dim = hidden_dim
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # Geometric feature dimension:
        # [s_curr_look (1), s_max_stack (1), s_curr_prior (1), s_look_prior (1)] -> 4 dims
        # Full input to fusion: embed_dim * 3 + 4 = 128 * 3 + 4 = 388
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Learnable projection for stack attention
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)

        # Transition Classification Head
        # Inputs: h_fused (hidden_dim) + geometric signals (4) -> hidden_dim + 4
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(
        self,
        x_curr: torch.Tensor,
        x_lookahead: torch.Tensor,
        x_prior: torch.Tensor,
        stack_memory: torch.Tensor,
        stack_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x_curr: (B, embed_dim) - Current action embedding
            x_lookahead: (B, embed_dim) - Future lookahead action embedding
            x_prior: (B, embed_dim) - Context prior / project goal embedding
            stack_memory: (B, max_stack_depth, embed_dim) - Historical task stack centroids
            stack_mask: (B, max_stack_depth) - 1 for valid stack items, 0 for padding
        Returns:
            transition_logits: (B, 3) - Logits for [NEW_MACRO, SUBROUTINE_SHIFT, RESUME]
            pointer_logits: (B, max_stack_depth) - Attention logits over stack memory
        """
        # 1. Compute Direct Geometric Cosine Affinities
        s_curr_look = (x_curr * x_lookahead).sum(dim=-1, keepdim=True)
        s_curr_prior = (x_curr * x_prior).sum(dim=-1, keepdim=True)
        s_look_prior = (x_lookahead * x_prior).sum(dim=-1, keepdim=True)

        # Cosine similarity between lookahead and each stack slot
        # stack_memory is (B, K, D), x_lookahead is (B, D)
        # raw dot product: (B, K)
        stack_sims = torch.bmm(x_lookahead.unsqueeze(1), stack_memory.transpose(1, 2)).squeeze(1)

        # Mask invalid stack slots
        masked_stack_sims = stack_sims.masked_fill(~stack_mask.bool(), -1.0)
        has_items = stack_mask.any(dim=-1, keepdim=True).float()
        s_max_stack, _ = masked_stack_sims.max(dim=-1, keepdim=True)
        s_max_stack = s_max_stack * has_items

        geom_features = torch.cat([s_curr_look, s_max_stack, s_curr_prior, s_look_prior], dim=-1)

        # 2. Resumption Pointer Cross-Attention
        # Combine projected query-key attention with raw cosine affinity
        q = self.q_proj(x_lookahead).unsqueeze(1)  # (B, 1, H)
        k = self.k_proj(stack_memory)               # (B, K, H)
        proj_attn = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)

        # Final pointer logits scale directly with stack similarity
        temp = torch.clamp(self.temperature, min=0.05, max=1.0)
        pointer_logits = (stack_sims / temp) + 0.2 * proj_attn
        pointer_logits = pointer_logits.masked_fill(~stack_mask.bool(), -1e9)

        # 3. Transition Classification Head
        fused_in = torch.cat([x_curr, x_lookahead, x_prior, geom_features], dim=-1)
        h_fused = self.fusion(fused_in)

        head_in = torch.cat([h_fused, geom_features], dim=-1)
        transition_logits = self.classifier_head(head_in)

        return transition_logits, pointer_logits

    def predict(
        self,
        x_curr: torch.Tensor,
        x_lookahead: torch.Tensor,
        x_prior: torch.Tensor,
        stack_memory: torch.Tensor,
        stack_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            t_logits, p_logits = self.forward(x_curr, x_lookahead, x_prior, stack_memory, stack_mask)
            t_probs = F.softmax(t_logits, dim=-1)
            pred_classes = torch.argmax(t_probs, dim=-1)

            p_probs = F.softmax(p_logits, dim=-1)
            pred_pointers = torch.argmax(p_probs, dim=-1)

        class_names = ["NEW_MACRO", "SUBROUTINE_SHIFT", "RESUME"]
        return {
            "predicted_class_id": pred_classes.cpu().tolist(),
            "predicted_class_name": [class_names[c] for c in pred_classes.cpu().tolist()],
            "transition_probabilities": t_probs.cpu().tolist(),
            "resumption_pointer_index": pred_pointers.cpu().tolist(),
            "pointer_probabilities": p_probs.cpu().tolist()
        }


def compute_differentiation_loss(
    transition_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    target_transitions: torch.Tensor,
    target_pointers: torch.Tensor,
    pointer_loss_weight: float = 1.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for transition classification and resumption pointer matching.
    """
    ce_loss = F.cross_entropy(transition_logits, target_transitions, label_smoothing=0.02)

    resume_mask = (target_transitions == 2)
    if resume_mask.any():
        ptr_loss = F.cross_entropy(pointer_logits[resume_mask], target_pointers[resume_mask])
    else:
        ptr_loss = torch.tensor(0.0, device=transition_logits.device)

    total_loss = ce_loss + pointer_loss_weight * ptr_loss
    return total_loss, ce_loss, ptr_loss
