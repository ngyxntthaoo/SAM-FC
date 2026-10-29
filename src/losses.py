"""losses.py — Multi-task and logic-consistency loss for SAM-FC.

L_total = L_verdict + lambda_conflict * L_conflict + lambda_consistency * L_consistency

Where:
- L_verdict: CrossEntropy with Refuted class weighting (w=1.20) and label smoothing (0.10)
- L_conflict: CrossEntropy over 5-class fine-grained conflict slots
- L_consistency: Logic constraint penalizing contradictory predictions:
  (1/B) * sum( p_verdict(Supported) * (1 - p_slot(NONE)) )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMFCLoss(nn.Module):
    """Multi-task objective with logic consistency regularization."""

    def __init__(
        self,
        refuted_weight: float = 1.20,
        label_smoothing: float = 0.10,
        lambda_conflict: float = 0.35,
        lambda_consistency: float = 0.20,
        supported_id: int = 0,
        none_slot_id: int = 0,
    ) -> None:
        super().__init__()
        weights = torch.tensor([1.0, refuted_weight, 1.0], dtype=torch.float)
        self.criterion_verdict = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        self.criterion_conflict = nn.CrossEntropyLoss()

        self.lambda_conflict = lambda_conflict
        self.lambda_consistency = lambda_consistency
        self.supported_id = supported_id
        self.none_slot_id = none_slot_id

    def forward(
        self,
        logits_verdict: torch.Tensor,
        verdict_targets: torch.Tensor,
        logits_conflict: Optional[torch.Tensor] = None,
        conflict_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss and individual loss components."""
        if self.criterion_verdict.weight is not None and self.criterion_verdict.weight.device != logits_verdict.device:
            self.criterion_verdict.weight = self.criterion_verdict.weight.to(logits_verdict.device)

        loss_verdict = self.criterion_verdict(logits_verdict, verdict_targets)
        loss_conflict = torch.tensor(0.0, device=logits_verdict.device)
        loss_consistency = torch.tensor(0.0, device=logits_verdict.device)

        # Auxiliary conflict loss and logic consistency constraint
        if logits_conflict is not None and conflict_targets is not None and self.lambda_conflict > 0:
            loss_conflict = self.criterion_conflict(logits_conflict, conflict_targets)

            if self.lambda_consistency > 0:
                p_verdict = F.softmax(logits_verdict, dim=-1)
                p_slot = F.softmax(logits_conflict, dim=-1)
                p_supported = p_verdict[:, self.supported_id]
                p_has_conflict = 1.0 - p_slot[:, self.none_slot_id]
                loss_consistency = torch.mean(p_supported * p_has_conflict)

        total_loss = loss_verdict + (self.lambda_conflict * loss_conflict) + (self.lambda_consistency * loss_consistency)

        components = {
            "loss_total": float(total_loss.item()),
            "loss_verdict": float(loss_verdict.item()),
            "loss_conflict": float(loss_conflict.item()),
            "loss_consistency": float(loss_consistency.item()),
        }
        return total_loss, components
