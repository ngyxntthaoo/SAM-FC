"""train_main.py — Train SAM-FC model on ViFactCheck or ViWikiFC.

Usage:
    python scripts/train_main.py --dataset vifactcheck --epochs 8 --batch_size 4
    python scripts/train_main.py --dataset viwikifc --epochs 8 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train SAM-FC Fact Verification Model")
    parser.add_argument("--dataset", choices=["vifactcheck", "viwikifc"], default="vifactcheck", help="Dataset name")
    parser.add_argument("--backbone", default="xlm-roberta-large", help="Encoder backbone or NLI checkpoint path")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr_backbone", type=float, default=5e-6, help="Backbone learning rate")
    parser.add_argument("--lr_heads", type=float, default=1e-4, help="Classification heads learning rate")
    parser.add_argument("--lambda_conflict", type=float, default=0.35, help="Auxiliary conflict loss weight")
    parser.add_argument("--lambda_consistency", type=float, default=0.20, help="Logic-consistency loss weight")
    parser.add_argument("--refuted_weight", type=float, default=1.20, help="Class weight for Refuted verdict")
    parser.add_argument("--label_smoothing", type=float, default=0.10, help="Label smoothing for verdict loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit for debugging")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset} | Backbone: {args.backbone}")

    # File mapping
    prefix = "vifactcheck" if args.dataset == "vifactcheck" else "factcheckwiki"
    train_path = DATA_DIR / f"train-{prefix}.csv"
    dev_path = DATA_DIR / f"dev-{prefix}.csv"
    test_path = DATA_DIR / f"test-{prefix}.csv"

    print(f"Loading data from {DATA_DIR}...")
    train_samples = load_dataset(train_path, dataset_type=args.dataset, limit=args.limit)
    dev_samples = load_dataset(dev_path, dataset_type=args.dataset, limit=args.limit)
    test_samples = load_dataset(test_path, dataset_type=args.dataset, limit=args.limit)
    print(f"Loaded: train={len(train_samples)}, dev={len(dev_samples)}, test={len(test_samples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    setup_diff_tokenizer(tokenizer)

    train_loader = create_dataloader(train_samples, tokenizer, batch_size=args.batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_samples, tokenizer, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = create_dataloader(test_samples, tokenizer, batch_size=args.batch_size * 2, shuffle=False)

    print("Initializing SAM-FC architecture...")
    model = SAMFCModel(
        backbone_name_or_path=args.backbone,
        num_verdict_labels=3,
        num_conflict_labels=5,
        pair_pooling=True,
        use_conflict_head=True,
        use_gate=True,
    )
    resize_with_mean_init(model.encoder, len(tokenizer))

    loss_fn = SAMFCLoss(
        refuted_weight=args.refuted_weight,
        label_smoothing=args.label_smoothing,
        lambda_conflict=args.lambda_conflict,
        lambda_consistency=args.lambda_consistency,
    )

    save_dir = OUTPUT_DIR / "checkpoints" / f"sam_fc_{args.dataset}_seed{args.seed}"
    trainer = SAMFCTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
    )

    print("Starting training...")
    train_res = trainer.train(
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=args.epochs,
        head_warmup_epochs=1,
        grad_accum_steps=args.grad_accum,
        save_dir=str(save_dir),
    )

    print(f"\nTraining complete. Best Epoch: {train_res['best_epoch']} | Best Dev F1: {train_res['best_dev_f1']:.2f}%")
    print("Evaluating on Test Set...")
    test_metrics, _, _ = trainer.evaluate(test_loader)
    print("\n" + "=" * 50)
    print("TEST EVALUATION RESULTS:")
    print(json.dumps(test_metrics, indent=2))
    print("=" * 50)


if __name__ == "__main__":
    main()
