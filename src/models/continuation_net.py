"""
Task Continuation Network (Model 1)

PyTorch neural classifier predicting whether a continuous workflow step is continuing
an active subroutine (CONTINUE_FLOW) or triggering a candidate event boundary (EVENT_SHIFT).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class TaskContinuationNet(nn.Module):
    """
    Lightweight residual MLP predicting workflow continuity vs. candidate breakpoint.
    Input dimension: d_in = 144 (128-dim latent embedding + temporal deltas + categorical flags).
    """

    def __init__(self, d_in: int = 144, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Residual Block
        self.res_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_norm1 = nn.LayerNorm(hidden_dim)
        self.res_silu = nn.SiLU()
        self.res_dropout = nn.Dropout(dropout)
        self.res_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.res_norm2 = nn.LayerNorm(hidden_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, d_in)
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        h = self.input_layer(x)
        res = h
        h = self.res_linear1(h)
        h = self.res_norm1(h)
        h = self.res_silu(h)
        h = self.res_dropout(h)
        h = self.res_linear2(h)
        h = self.res_norm2(h)
        h = self.res_silu(h + res)

        logits = self.head(h)
        return logits

    def predict_continuation(self, x: torch.Tensor, threshold: float = 0.40) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            continuation_prob: Tensor of shape (batch_size,), values in [0, 1]
            is_event_shift: Bool tensor of shape (batch_size,), True if prob < threshold
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            # Logit represents continue probability (positive logit = high continue)
            prob = torch.sigmoid(logits)
            is_shift = prob < threshold
        return prob, is_shift


def compute_continuation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 3.5
) -> torch.Tensor:
    """
    Weighted binary cross entropy loss.
    targets: 1 for EVENT_SHIFT, 0 for CONTINUE_FLOW.
    Note: if logits represent P(continue), then target for continuation is (1 - targets).
    """
    # Ensure both logits and targets have matching shape (1D or 2D)
    logits_flat = logits.view(-1)
    cont_targets = (1.0 - targets.float()).view(-1)
    weight = torch.tensor([pos_weight], device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits_flat, cont_targets, pos_weight=weight)
    return loss
