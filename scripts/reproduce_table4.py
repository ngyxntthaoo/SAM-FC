"""reproduce_table4.py — Reproduce Table 4 (Claim Length Stratification).

Usage:
    python scripts/reproduce_table4.py
"""

from __future__ import annotations

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def stratify_by_length(predictions_path: Path):
    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    total = len(preds)
    # Length definition: whitespace-separated words in claim
    short_items = [x for x in preds if len(x["claim"].split()) <= 20]
    med_items = [x for x in preds if 21 <= len(x["claim"].split()) <= 35]
    long_items = [x for x in preds if len(x["claim"].split()) > 35]

    strata = [
        ("Short (<20)", short_items),
        ("Medium (20-35)", med_items),
        ("Long (>35)", long_items),
    ]

    rows = []
    for name, items in strata:
        count = len(items)
        ratio = (count / total) * 100
        y_true = [x["gold_verdict"] for x in items]
        y_pred = [x["pred_verdict"] for x in items]
        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average="macro") * 100
        rows.append([name, count, f"{ratio:.1f}%", f"{acc:.2f}", f"{f1:.2f}"])

    return rows


def main():
    print("\n" + "=" * 80)
    print(" REPRODUCING TABLE 4: TEST PERFORMANCE BY CLAIM LENGTH (ACIIDS 2027)")
    print("=" * 80 + "\n")

    vfc_path = OUTPUTS_DIR / "vifactcheck_test_predictions_ensemble.json"
    wiki_path = OUTPUTS_DIR / "viwikifc_test_predictions_ensemble.json"

    headers = ["Dataset", "Length (word)", "Count", "Ratio", "Acc.", "Macro-F1"]
    all_rows = []

    if vfc_path.exists():
        vfc_rows = stratify_by_length(vfc_path)
        for idx, r in enumerate(vfc_rows):
            ds_col = "ViFactCheck" if idx == 0 else ""
            all_rows.append([ds_col] + r)

    if wiki_path.exists():
        wiki_rows = stratify_by_length(wiki_path)
        for idx, r in enumerate(wiki_rows):
            ds_col = "ViWikiFC" if idx == 0 else ""
            all_rows.append([ds_col] + r)

    print(tabulate(all_rows, headers=headers, tablefmt="github"))
    print("\n[OK] Table 4 reproduced successfully.\n")


if __name__ == "__main__":
    main()
