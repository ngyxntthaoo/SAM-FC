"""trainer.py — Training, validation, and evaluation pipeline for SAM-FC.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.losses import SAMFCLoss
from src.metrics import compute_metrics
from src.models import SAMFCModel


class SAMFCTrainer:
    """Trainer supporting Head Warmup, Differential LR, and multi-task evaluation."""

    def __init__(
        self,
        model: SAMFCModel,
        loss_fn: SAMFCLoss,
        device: torch.device,
        lr_backbone: float = 5e-6,
        lr_heads: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def _setup_optimizer(self, freeze_backbone: bool = False) -> torch.optim.Optimizer:
        if freeze_backbone:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            head_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder") and p.requires_grad]
            return torch.optim.AdamW(head_params, lr=self.lr_heads, weight_decay=self.weight_decay)

        for p in self.model.encoder.parameters():
            p.requires_grad = True

        encoder_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder")]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder")]

        param_groups = [
            {"params": encoder_params, "lr": self.lr_backbone, "weight_decay": self.weight_decay},
            {"params": head_params, "lr": self.lr_heads, "weight_decay": self.weight_decay},
        ]
        return torch.optim.AdamW(param_groups)

    def train(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int = 8,
        head_warmup_epochs: int = 1,
        grad_accum_steps: int = 2,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the model across epochs with warmup and evaluation."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        best_dev_f1 = -1.0
        best_epoch = -1
        best_state = None

        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for epoch in range(1, num_epochs + 1):
            is_warmup = epoch <= head_warmup_epochs
            optimizer = self._setup_optimizer(freeze_backbone=is_warmup)
            total_steps = (len(train_loader) // grad_accum_steps) * 1
            scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)

            self.model.train()
            running_loss = 0.0
            t0 = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                claim_mask = batch["claim_mask"].to(self.device)
                evidence_mask = batch["evidence_mask"].to(self.device)
                label_v = batch["label_verdict"].to(self.device)
                label_c = batch["label_conflict"].to(self.device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits_v, logits_c, _ = self.model(input_ids, attention_mask, claim_mask, evidence_mask)
                    loss, _ = self.loss_fn(logits_v, label_v, logits_c, label_c)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
                running_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            train_loss = running_loss / len(train_loader)
            dev_metrics, _, _ = self.evaluate(dev_loader)
            dev_f1 = dev_metrics["macro_f1"]
            elapsed = time.time() - t0

            warmup_tag = " (Warmup)" if is_warmup else ""
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d}{warmup_tag} | "
                f"Loss: {train_loss:.4f} | Dev F1: {dev_f1:.2f}% | "
                f"Dev Acc: {dev_metrics['accuracy']:.2f}% | Time: {elapsed:.0f}s"
            )

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if save_dir:
                    torch.save(best_state, os.path.join(save_dir, "best_model.pt"))

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_epoch": best_epoch, "best_dev_f1": best_dev_f1}

    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluate model and return metrics, predictions, and prediction probabilities."""
        self.model.eval()
        all_pred_v: List[int] = []
        all_true_v: List[int] = []
        all_pred_c: List[int] = []
        all_true_c: List[int] = []
        all_probs_v: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                claim_mask = batch["claim_mask"].to(self.device)
                evidence_mask = batch["evidence_mask"].to(self.device)

                logits_v, logits_c, _ = self.model(input_ids, attention_mask, claim_mask, evidence_mask)
                probs_v = torch.softmax(logits_v, dim=-1).cpu().numpy()
                preds_v = np.argmax(probs_v, axis=-1)

                all_probs_v.append(probs_v)
                all_pred_v.extend(preds_v.tolist())
                all_true_v.extend(batch["label_verdict"].numpy().tolist())

                if logits_c is not None:
                    probs_c = torch.softmax(logits_c, dim=-1).cpu().numpy()
                    preds_c = np.argmax(probs_c, axis=-1)
                    all_pred_c.extend(preds_c.tolist())
                    all_true_c.extend(batch["label_conflict"].numpy().tolist())

        y_true_c = all_true_c if all_true_c else None
        y_pred_c = all_pred_c if all_pred_c else None

        metrics = compute_metrics(all_true_v, all_pred_v, y_true_c, y_pred_c)
        probs_v_concat = np.concatenate(all_probs_v, axis=0)
        return metrics, np.array(all_pred_v), probs_v_concat
