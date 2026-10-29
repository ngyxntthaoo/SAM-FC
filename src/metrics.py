"""metrics.py — Evaluation metrics for fact verification and conflict slot attribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_metrics(
    y_true_verdict: Sequence[int],
    y_pred_verdict: Sequence[int],
    y_true_conflict: Optional[Sequence[int]] = None,
    y_pred_conflict: Optional[Sequence[int]] = None,
    verdict_names: Sequence[str] = ("Supported", "Refuted", "NEI"),
) -> Dict[str, Any]:
    """Compute Macro-F1, Accuracy, Per-class F1, and Slot Accuracy on Refuted claims."""
    acc = accuracy_score(y_true_verdict, y_pred_verdict) * 100
    macro_f1 = f1_score(y_true_verdict, y_pred_verdict, average="macro") * 100
    per_class = f1_score(y_true_verdict, y_pred_verdict, average=None) * 100

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "per_class_f1": {
            name: round(per_class[i], 2) for i, name in enumerate(verdict_names)
        },
    }

    # Slot Accuracy on Refuted subset
    if y_true_conflict is not None and y_pred_conflict is not None:
        y_true_v = np.array(y_true_verdict)
        y_true_c = np.array(y_true_conflict)
        y_pred_c = np.array(y_pred_conflict)

        # Refuted is label 1
        refuted_mask = (y_true_v == 1)
        if np.any(refuted_mask):
            slot_acc = (y_true_c[refuted_mask] == y_pred_c[refuted_mask]).mean() * 100
            metrics["conflict_slot_acc"] = round(float(slot_acc), 2)

        # Inconsistency Rate (ICR): Predicted Supported (0) while slot head predicts conflict (!= 0)
        y_pred_v = np.array(y_pred_verdict)
        inconsistent = (y_pred_v == 0) & (y_pred_c != 0)
        metrics["inconsistency_rate"] = round(float(inconsistent.mean() * 100), 2)

    return metrics
