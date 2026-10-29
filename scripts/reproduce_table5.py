"""reproduce_table5.py — Reproduce Table 5 (Cross-Domain Conflict Breakdown on Refuted Claims).

Usage:
    python scripts/reproduce_table5.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def analyze_conflict_breakdown(predictions_path: Path):
    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Filter refuted gold samples
    refuted_items = [x for x in preds if str(x.get("gold_verdict", "")).strip().capitalize() == "Refuted"]
    conflict_order = ["NUMERIC", "NEGATION", "ENTITY", "TEMPORAL"]
    display_names = {"NUMERIC": "Numeric", "NEGATION": "Negation", "ENTITY": "Entity", "TEMPORAL": "Temporal"}

    rows = []
    for ctype in conflict_order:
        items = [x for x in refuted_items if str(x.get("gold_conflict", "")).upper() == ctype]
        count = len(items)
        if count == 0:
            continue

        verdict_correct = sum(1 for x in items if str(x.get("pred_verdict", "")).strip().capitalize() == "Refuted")
        verdict_recall = (verdict_correct / count) * 100

        slot_correct = sum(1 for x in items if x.get("is_conflict_correct", False))
        slot_acc = (slot_correct / count) * 100

        # Determine dominant error
        errors = Counter()
        for x in items:
            pred_v = str(x.get("pred_verdict", "")).strip().capitalize()
            if pred_v != "Refuted":
                errors[f"{pred_v}"] += 1
            elif not x.get("is_conflict_correct", False):
                pred_c = str(x.get("pred_conflict", "")).capitalize()
                errors[f"slot:{pred_c}"] += 1

        top_err = errors.most_common(1)
        if top_err:
            err_type, err_cnt = top_err[0]
            if err_type.startswith("slot:"):
                main_error = f"{err_cnt} assigned to {err_type.split(':')[1]}"
            else:
                main_error = f"{err_cnt} predicted as {err_type}"
        else:
            main_error = "None"

        rows.append([
            display_names.get(ctype, ctype),
            count,
            f"{verdict_recall:.1f}%",
            f"{slot_acc:.1f}%",
            main_error,
        ])

    return len(refuted_items), rows


def main():
    print("\n" + "=" * 90)
    print(" REPRODUCING TABLE 5: CROSS-DOMAIN CONFLICT-SLOT ATTRIBUTION (ACIIDS 2027)")
    print("=" * 90 + "\n")

    vfc_path = OUTPUTS_DIR / "vifactcheck_test_predictions_ensemble.json"
    wiki_path = OUTPUTS_DIR / "viwikifc_test_predictions_ensemble.json"

    headers = ["Dataset", "Type", "Count", "Verdict Recall", "Slot Acc.", "Main Error"]
    all_rows = []

    if vfc_path.exists():
        n_vfc, vfc_rows = analyze_conflict_breakdown(vfc_path)
        for idx, r in enumerate(vfc_rows):
            ds_col = f"ViFactCheck (n={n_vfc})" if idx == 0 else ""
            all_rows.append([ds_col] + r)

    if wiki_path.exists():
        n_wiki, wiki_rows = analyze_conflict_breakdown(wiki_path)
        for idx, r in enumerate(wiki_rows):
            ds_col = f"ViWikiFC (n={n_wiki})" if idx == 0 else ""
            all_rows.append([ds_col] + r)

    print(tabulate(all_rows, headers=headers, tablefmt="github"))
    print("\n[OK] Table 5 reproduced successfully.\n")


if __name__ == "__main__":
    main()
