"""run_ablation.py — Run ablation experiments for Table 3.

Configurations:
    1. full: Full model (SAM-FC)
    2. wo_nli: w/o NLI adaptation (raw encoder)
    3. wo_consistency: w/o logic consistency loss (lambda_consistency = 0.0)
    4. wo_diff_markers: w/o diff markers in claim
    5. wo_auxiliary_conflict_branch: w/o conflict slot branch (single-task)

Usage:
    python scripts/run_ablation.py --dataset vifactcheck --config wo_consistency
    python scripts/run_ablation.py --dataset viwikifc --config all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.dataset import create_dataloader, load_dataset
from src.diff_marker import resize_with_mean_init, setup_diff_tokenizer
from src.losses import SAMFCLoss
from src.models import SAMFCModel
from src.trainer import SAMFCTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ABLATION_SETTINGS = {
    "full": {
        "use_nli": True,
        "use_diff_markers": True,
        "use_conflict_head": True,
        "use_gate": True,
        "lambda_conflict": 0.35,
        "lambda_consistency": 0.20,
    },
    "wo_nli": {
        "use_nli": False,
        "use_diff_markers": True,
        "use_conflict_head": True,
        "use_gate": True,
        "lambda_conflict": 0.35,
        "lambda_consistency": 0.20,
    },
    "wo_consistency": {
        "use_nli": True,
        "use_diff_markers": True,
        "use_conflict_head": True,
        "use_gate": True,
        "lambda_conflict": 0.35,
        "lambda_consistency": 0.0,
    },
    "wo_diff_markers": {
        "use_nli": True,
        "use_diff_markers": False,
        "use_conflict_head": True,
        "use_gate": True,
        "lambda_conflict": 0.35,
        "lambda_consistency": 0.20,
    },
    "wo_auxiliary_conflict_branch": {
        "use_nli": True,
        "use_diff_markers": True,
        "use_conflict_head": False,
        "use_gate": False,
        "lambda_conflict": 0.0,
        "lambda_consistency": 0.0,
    },
}


def run_single_config(config_name: str, args):
    cfg = ABLATION_SETTINGS[config_name]
    print("\n" + "=" * 60)
    print(f"RUNNING ABLATION: {config_name}")
    print(json.dumps(cfg, indent=2))
    print("=" * 60)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix = "vifactcheck" if args.dataset == "vifactcheck" else "factcheckwiki"
    train_path = DATA_DIR / f"train-{prefix}.csv"
    dev_path = DATA_DIR / f"dev-{prefix}.csv"
    test_path = DATA_DIR / f"test-{prefix}.csv"

    train_samples = load_dataset(train_path, dataset_type=args.dataset, limit=args.limit)
    dev_samples = load_dataset(dev_path, dataset_type=args.dataset, limit=args.limit)
    test_samples = load_dataset(test_path, dataset_type=args.dataset, limit=args.limit)

    backbone_name = args.backbone if not cfg["use_nli"] else args.nli_checkpoint or args.backbone
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if cfg["use_diff_markers"]:
        setup_diff_tokenizer(tokenizer)

    train_loader = create_dataloader(
        train_samples, tokenizer, batch_size=args.batch_size, shuffle=True, use_diff_markers=cfg["use_diff_markers"]
    )
    dev_loader = create_dataloader(
        dev_samples, tokenizer, batch_size=args.batch_size * 2, shuffle=False, use_diff_markers=cfg["use_diff_markers"]
    )
    test_loader = create_dataloader(
        test_samples, tokenizer, batch_size=args.batch_size * 2, shuffle=False, use_diff_markers=cfg["use_diff_markers"]
    )

    model = SAMFCModel(
        backbone_name_or_path=backbone_name,
        num_verdict_labels=3,
        num_conflict_labels=5,
        pair_pooling=True,
        use_conflict_head=cfg["use_conflict_head"],
        use_gate=cfg["use_gate"],
    )
    if cfg["use_diff_markers"]:
        resize_with_mean_init(model.encoder, len(tokenizer))

    loss_fn = SAMFCLoss(
        refuted_weight=1.20,
        label_smoothing=0.10,
        lambda_conflict=cfg["lambda_conflict"],
        lambda_consistency=cfg["lambda_consistency"],
    )

    trainer = SAMFCTrainer(model=model, loss_fn=loss_fn, device=device)
    trainer.train(
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=args.epochs,
        head_warmup_epochs=1,
    )

    metrics, _, _ = trainer.evaluate(test_loader)
    print(f"\nResult for {config_name}:")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run SAM-FC Ablations")
    parser.add_argument("--dataset", choices=["vifactcheck", "viwikifc"], default="vifactcheck")
    parser.add_argument("--config", choices=list(ABLATION_SETTINGS.keys()) + ["all"], default="wo_consistency")
    parser.add_argument("--backbone", default="xlm-roberta-large")
    parser.add_argument("--nli_checkpoint", default=None, help="Path to pre-adapted NLI backbone")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs_to_run = list(ABLATION_SETTINGS.keys()) if args.config == "all" else [args.config]
    results = {}
    for cfg_name in configs_to_run:
        results[cfg_name] = run_single_config(cfg_name, args)

    print("\n" + "=" * 60)
    print("ALL ABLATION RUNS COMPLETED:")
    print(json.dumps(results, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
