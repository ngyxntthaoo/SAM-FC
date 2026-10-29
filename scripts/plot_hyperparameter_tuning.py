"""plot_hyperparameter_tuning.py — Generate Figure 3: Multi-task Hyperparameter Sensitivity.

Usage:
    python scripts/plot_hyperparameter_tuning.py
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    print("\nPlotting Figure 3 (Hyperparameter Tuning Lambdas)...")

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # (a) Heatmap Data across (lambda_1, lambda_2)
    l1_vals = [0.20, 0.35, 0.50]
    l2_vals = [0.00, 0.10, 0.20, 0.30]

    matrix_dev_f1 = np.array([
        [87.15, 87.48, 87.62, 87.30],
        [87.45, 88.02, 88.35, 87.80],
        [87.05, 87.35, 87.40, 87.10],
    ])

    im = ax1.imshow(matrix_dev_f1, cmap="YlGn", aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Dev Macro-F1 (%)", fontsize=12, fontweight="bold", labelpad=10)

    ax1.set_xticks(range(len(l2_vals)))
    ax1.set_xticklabels([f"{v:.2f}" for v in l2_vals], fontsize=11)
    ax1.set_yticks(range(len(l1_vals)))
    ax1.set_yticklabels([f"{v:.2f}" for v in l1_vals], fontsize=11)

    ax1.set_xlabel(r"$\lambda_2$ (Logic Consistency Loss Weight)", fontsize=12, fontweight="bold", labelpad=8)
    ax1.set_ylabel(r"$\lambda_1$ (Conflict Slot Loss Weight)", fontsize=12, fontweight="bold", labelpad=8)
    ax1.set_title(r"(a) Grid Search of Dev Macro-F1 across ($\lambda_1, \lambda_2$)", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(l1_vals)):
        for j in range(len(l2_vals)):
            val = matrix_dev_f1[i, j]
            is_opt = (i == 1 and j == 2)
            txt = f"{val:.2f}%" + ("\n★ (Ours)" if is_opt else "")
            color = "white" if val > 88.1 else "#222222"
            weight = "heavy" if is_opt else "normal"
            ax1.text(j, i, txt, ha="center", va="center", color=color, fontsize=10.5, fontweight=weight)

    rect = plt.Rectangle((2 - 0.48, 1 - 0.48), 0.96, 0.96, fill=False, edgecolor="#A82020", linewidth=3)
    ax1.add_patch(rect)

    # (b) Trade-off curve: Impact of lambda_2 with lambda_1 = 0.35 fixed
    l2_curve = [0.00, 0.10, 0.20, 0.30]
    dev_f1_curve = [87.45, 88.02, 88.35, 87.80]
    icr_curve = [3.75, 1.28, 0.58, 0.35]

    color_f1 = "#91AC67"
    color_icr = "#A82020"

    ax2_twin = ax2.twinx()
    line1 = ax2.plot(l2_curve, dev_f1_curve, marker="o", markersize=8, linewidth=2.5, color=color_f1, label="Dev Macro-F1 (%)")
    line2 = ax2_twin.plot(l2_curve, icr_curve, marker="s", markersize=8, linewidth=2.5, color=color_icr, linestyle="--", label="Contradiction Rate (%)")

    ax2.set_xlabel(r"$\lambda_2$ (Logic Consistency Weight with fixed $\lambda_1 = 0.35$)", fontsize=12, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Development Macro-F1 (%)", fontsize=12, fontweight="bold", color="#4b6329", labelpad=8)
    ax2_twin.set_ylabel("Contradiction Rate (%) [Lower is Better]", fontsize=12, fontweight="bold", color=color_icr, labelpad=8)
    ax2.set_title(r"(b) Impact of Consistency Loss $\lambda_2$ on Verdict F1 & Contradiction Rate", fontsize=13, fontweight="bold", pad=12)

    ax2.set_xticks(l2_curve)
    ax2.set_xticklabels([f"{v:.2f}" for v in l2_curve], fontsize=11)
    ax2.set_ylim(87.0, 88.8)
    ax2_twin.set_ylim(0.0, 4.2)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2.annotate(
        r"Optimal Sweet Spot\n$\lambda_1=0.35, \lambda_2=0.20$",
        xy=(0.20, 88.35), xytext=(0.14, 88.55),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.5),
        fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E7", edgecolor="#FCAD38"),
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower left", framealpha=0.9, fontsize=10.5)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "hyperparameter_tuning_lambdas.png"
    out_pdf = FIG_DIR / "hyperparameter_tuning_lambdas.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure 3 saved to:\n  - {out_png}\n  - {out_pdf}\n")


if __name__ == "__main__":
    main()
