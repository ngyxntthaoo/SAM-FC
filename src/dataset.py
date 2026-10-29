"""dataset.py — Dataset loaders and tokenization for ViFactCheck and ViWikiFC.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.diff_marker import encode_pair
from src.slot_detector import (
    CONFLICT_MAP,
    VERDICT_MAP,
    detect_conflict_slot,
    normalize_vietnamese_text,
)


@dataclass
class MultiTaskSample:
    sample_id: str
    claim: str
    evidence: str
    verdict: str           # "Supported" | "Refuted" | "NEI"
    conflict_slot: str     # "NONE" | "TEMPORAL" | "NUMERIC" | "NEGATION" | "ENTITY"
    label_verdict: int     # 0 | 1 | 2
    label_conflict: int    # 0..4


def parse_verdict(raw_val: Any) -> str:
    """Normalize raw CSV label into standard verdict string."""
    v = str(raw_val).strip().lower().replace(" ", "_")
    if v in ("0", "0.0", "supported", "support", "supports"):
        return "Supported"
    if v in ("1", "1.0", "refuted", "refute", "refutes"):
        return "Refuted"
    if v in ("2", "2.0", "nei", "not_enough_information", "not_enough_info"):
        return "NEI"
    raise ValueError(f"Unknown verdict label: {raw_val}")


def load_dataset(
    file_path: str | Path,
    dataset_type: str = "vifactcheck",
    limit: Optional[int] = None,
) -> List[MultiTaskSample]:
    """Load and weakly label samples from CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)

    is_wiki = "wiki" in dataset_type.lower() or "claim" in df.columns
    claim_col = "claim" if is_wiki else "Statement"
    evidence_col = "evidence" if is_wiki else "Evidence"
    label_col = "gold_label" if is_wiki else "labels"
    id_col = "pairID" if is_wiki else "index"

    samples: List[MultiTaskSample] = []
    for idx, row in df.iterrows():
        claim_text = normalize_vietnamese_text(str(row.get(claim_col, "")))
        evidence_text = normalize_vietnamese_text(str(row.get(evidence_col, "")))
        if not claim_text or not evidence_text or claim_text == "nan" or evidence_text == "nan":
            continue

        raw_label = row.get(label_col, "")
        try:
            verdict = parse_verdict(raw_label)
        except ValueError:
            continue

        conflict_slot = detect_conflict_slot(claim_text, evidence_text, verdict)
        sample_id = str(row.get(id_col, f"sample_{idx}"))

        samples.append(
            MultiTaskSample(
                sample_id=sample_id,
                claim=claim_text,
                evidence=evidence_text,
                verdict=verdict,
                conflict_slot=conflict_slot,
                label_verdict=VERDICT_MAP[verdict],
                label_conflict=CONFLICT_MAP[conflict_slot],
            )
        )
    return samples


class FactCheckingDataset(Dataset):
    """PyTorch Dataset with on-the-fly diff marking and tokenization."""

    def __init__(
        self,
        samples: Sequence[MultiTaskSample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        use_diff_markers: bool = True,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_diff_markers = use_diff_markers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        item = encode_pair(
            s.claim,
            s.evidence,
            tokenizer=self.tokenizer,
            max_len=self.max_length,
            use_diff_markers=self.use_diff_markers,
        )
        item["sample_id"] = s.sample_id
        item["label_verdict"] = torch.tensor(s.label_verdict, dtype=torch.long)
        item["label_conflict"] = torch.tensor(s.label_conflict, dtype=torch.long)
        return item


def create_dataloader(
    samples: Sequence[MultiTaskSample],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    shuffle: bool = False,
    max_length: int = 256,
    use_diff_markers: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    dataset = FactCheckingDataset(
        samples,
        tokenizer=tokenizer,
        max_length=max_length,
        use_diff_markers=use_diff_markers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
