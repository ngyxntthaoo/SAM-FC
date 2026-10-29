"""plot_confusion_matrix.py — Generate Figure 4: Cross-Domain Confusion Matrices.

Usage:
    python scripts/plot_confusion_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "predictions"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    print("\nPlotting Figure 4 (Cross-Domain Confusion Matrices)...")
    vfc_file = OUTPUTS_DIR / "vifactcheck_test_predictions_ensemble.json"
    wiki_file = OUTPUTS_DIR / "viwikifc_test_predictions_ensemble.json"

    with open(vfc_file, "r", encoding="utf-8") as f:
        vfc_preds = json.load(f)
    with open(wiki_file, "r", encoding="utf-8") as f:
        wiki_preds = json.load(f)

    labels = ["Supported", "Refuted", "NEI"]

    vfc_gold = [p["gold_verdict"] for p in vfc_preds]
    vfc_pred = [p["pred_verdict"] for p in vfc_preds]
    cm_vfc = confusion_matrix(vfc_gold, vfc_pred, labels=labels)

    wiki_gold = [p["gold_verdict"] for p in wiki_preds]
    wiki_pred = [p["pred_verdict"] for p in wiki_preds]
    cm_wiki = confusion_matrix(wiki_gold, wiki_pred, labels=labels)

    recall_vfc = cm_vfc.astype("float") / cm_vfc.sum(axis=1, keepdims=True) * 100
    recall_wiki = cm_wiki.astype("float") / cm_wiki.sum(axis=1, keepdims=True) * 100

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)
    cms = [cm_vfc, cm_wiki]
    recalls = [recall_vfc, recall_wiki]
    titles = ["(a) ViFactCheck", "(b) ViWikiFC"]

    for idx, (ax, cm, rec, title) in enumerate(zip(axes, cms, recalls, titles)):
        im = ax.imshow(rec, cmap="Blues", vmin=0, vmax=100, aspect="equal")

        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)

        ax.tick_params(axis="x", which="major", labelsize=12, pad=6)
        ax.tick_params(axis="y", which="major", labelsize=12, pad=6)

        ax.set_xlabel("Predicted Verdict", fontsize=13, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Verdict", fontsize=13, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

        for i in range(3):
            for j in range(3):
                val = rec[i, j]
                count = cm[i, j]
                text_color = "white" if val > 50 else "black"
                ax.text(
                    j, i, f"{count}\n({val:.1f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=12,
                    linespacing=1.3,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.32, bottom=0.15, top=0.88)
    cbar_ax = fig.add_axes([0.905, 0.17, 0.016, 0.69])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recall (%)", fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_visible(False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "cross_domain_confusion_matrix.png"
    out_pdf = FIG_DIR / "cross_domain_confusion_matrix.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure 4 saved to:\n  - {out_png}\n  - {out_pdf}\n")


if __name__ == "__main__":
    main()
