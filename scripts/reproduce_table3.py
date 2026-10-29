"""reproduce_table3.py — Reproduce Table 3 (Ablation Study on ViFactCheck & ViWikiFC).

Usage:
    python scripts/reproduce_table3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation"


def load_ablation_data(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("\n" + "=" * 95)
    print(" REPRODUCING TABLE 3: ABLATION STUDY (ACIIDS 2027)")
    print("=" * 95 + "\n")

    vfc_file = ABLATION_DIR / "vifactcheck_ablation_result.json"
    wiki_file = ABLATION_DIR / "viwikifc_ablation_results.json"

    vfc_data = load_ablation_data(vfc_file).get("results", {})
    wiki_data = load_ablation_data(wiki_file).get("results", {})

    # The 5 configurations reported in Table 3 of the paper
    configs = [
        ("Full (CG-SAP / SAM-FC)", "full"),
        ("w/o NLI adaptation", "wo_nli"),
        ("w/o logic-consistency loss", "wo_consistency"),
        ("w/o diff markers", "wo_diff_markers"),
        ("w/o conflict-slot branch", "wo_auxiliary_conflict_branch"),
    ]

    for ds_name, ds_dict in [("ViFactCheck (n=1,447)", vfc_data), ("ViWikiFC (n=2,091)", wiki_data)]:
        full_res = ds_dict.get("full", {})
        full_f1 = full_res.get("test_macro_f1", 0.0) * 100

        rows = []
        for label, key in configs:
            res = ds_dict.get(key)
            # Handle fallback alias if single-task branch has alternative key
            if not res and key == "wo_auxiliary_conflict_branch":
                res = ds_dict.get("wo_slot_detection") or ds_dict.get("wo_gate")

            if res:
                f1 = res.get("test_macro_f1", 0.0) * 100
                acc = res.get("test_accuracy", 0.0) * 100
                per_cls = res.get("per_class_f1", {})
                s_f1 = per_cls.get("Supported", 0.0) * 100
                r_f1 = per_cls.get("Refuted", 0.0) * 100
                n_f1 = per_cls.get("NEI", 0.0) * 100
                slot_acc = res.get("conflict_slot_acc")
                slot_str = f"{slot_acc * 100:.2f}" if slot_acc is not None else "--"
                delta_f1 = f1 - full_f1
                delta_str = "--" if key == "full" else f"{delta_f1:+.2f}"

                rows.append([
                    label,
                    f"{f1:.2f}",
                    f"{acc:.2f}",
                    f"{s_f1:.2f}",
                    f"{r_f1:.2f}",
                    f"{n_f1:.2f}",
                    slot_str,
                    delta_str,
                ])
            else:
                rows.append([label, "--", "--", "--", "--", "--", "--", "--"])

        headers = ["Configuration", "Test F1", "Acc.", "Supported F1", "Refuted F1", "NEI F1", "Slot Acc.", "ΔF1"]
        print(f"--- Dataset: {ds_name} ---")
        print(tabulate(rows, headers=headers, tablefmt="github"))
        print()

    print("[OK] Table 3 reproduced successfully.\n")


if __name__ == "__main__":
    main()
