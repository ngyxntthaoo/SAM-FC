"""models.py — SAM-FC (Slot-Attributed Multi-Task Learning for Fact-Checking).

Architecture:
1. Contextual Encoder (NLI-adapted XLM-RoBERTa / PLM)
2. Pair Pooling: [h_cls; h_claim; h_evidence; |h_claim - h_evidence|; h_claim * h_evidence]
3. Auxiliary Conflict-Slot Head: predicts fine-grained conflict types on Refuted claims
4. Conflict-Gated Modulation: g = sigmoid(W_g * h_conflict + b_g), h_fused = h_pair + g * h_shift
5. Final Verdict Classification Head: outputs probabilities over [Supported, Refuted, NEI]
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SAMFCModel(nn.Module):
    """SAM-FC: Multi-task fact verification model with conflict-gated modulation."""

    def __init__(
        self,
        backbone_name_or_path: str = "xlm-roberta-large",
        num_verdict_labels: int = 3,
        num_conflict_labels: int = 5,
        dropout: float = 0.1,
        pair_pooling: bool = True,
        use_conflict_head: bool = True,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path
        self.config = AutoConfig.from_pretrained(backbone_name_or_path)
        self.encoder = AutoModel.from_pretrained(backbone_name_or_path, config=self.config)
        hidden_size = self.config.hidden_size

        self.pair_pooling = pair_pooling
        self.use_conflict_head = use_conflict_head
        self.use_gate = use_gate
        self.dropout = nn.Dropout(dropout)

        # 1. Pair Pooling projection layer
        if self.pair_pooling:
            self.pair_proj = nn.Sequential(
                nn.Linear(5 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # 2. Auxiliary Conflict-Slot Head
        if self.use_conflict_head:
            self.conflict_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.conflict_head = nn.Linear(hidden_size // 2, num_conflict_labels)

            # 3. Conflict Gate & Shift Vector
            if self.use_gate:
                self.conflict_gate = nn.Sequential(
                    nn.Linear(hidden_size // 2, hidden_size),
                    nn.Sigmoid(),
                )
                self.conflict_shift_proj = nn.Linear(hidden_size // 2, hidden_size)

        # 4. Verdict Classification Head
        self.verdict_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_verdict_labels),
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden[:, 0, :]
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: Optional[torch.Tensor] = None,
        evidence_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through SAM-FC.
        
        Returns:
            logits_verdict: [Batch, num_verdict_labels]
            logits_conflict: [Batch, num_conflict_labels] or None (if single-task)
            gate: [Batch, hidden_size] or None (if gating disabled)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        cls_rep = hidden[:, 0, :]

        # Compute pair representation
        if self.pair_pooling and claim_mask is not None and evidence_mask is not None:
            claim_rep = self._masked_mean(hidden, claim_mask)
            evidence_rep = self._masked_mean(hidden, evidence_mask)
            pair = torch.cat(
                [cls_rep, claim_rep, evidence_rep, (claim_rep - evidence_rep).abs(), claim_rep * evidence_rep],
                dim=-1,
            )
            rep = self.pair_proj(self.dropout(pair))
        else:
            rep = self.dropout(cls_rep)

        logits_conflict = None
        gate = None
        fused_rep = rep

        # Multi-task conflict-slot extraction & modulation
        if self.use_conflict_head:
            conflict_feat = self.conflict_proj(rep)
            logits_conflict = self.conflict_head(conflict_feat)

            if self.use_gate:
                gate = self.conflict_gate(conflict_feat)
                shift = self.conflict_shift_proj(conflict_feat)
                fused_rep = rep + (gate * shift)

        logits_verdict = self.verdict_head(fused_rep)
        return logits_verdict, logits_conflict, gate
