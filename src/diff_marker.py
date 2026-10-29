"""diff_marker.py — Claim-evidence difference marking (<diff> tokens) and tokenization.

Words in the claim that do not appear in the evidence are wrapped with <diff> ... </diff> tags
to directly indicate to the encoder where the claim departs from the evidence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

DIFF_OPEN = "<diff>"
DIFF_CLOSE = "</diff>"
SPECIAL_DIFF_TOKENS = [DIFF_OPEN, DIFF_CLOSE]

_EDGE_PUNCT = re.compile(r"^\W+|\W+$")


def norm_word(word: str) -> str:
    """Normalize word by stripping edge punctuation and lowercasing."""
    return _EDGE_PUNCT.sub("", word.lower())


def mark_claim_diff(claim: str, evidence: str) -> str:
    """Wrap claim words absent from the evidence in <diff> ... </diff> tags.
    
    Operates at whitespace-separated syllable/word level. Contiguous novel words
    are grouped into a single <diff> ... </diff> span.
    """
    evidence_words: Set[str] = {norm_word(w) for w in evidence.split()}
    out: List[str] = []
    span_open = False

    for word in claim.split():
        nw = norm_word(word)
        novel = bool(nw) and nw not in evidence_words
        if novel and not span_open:
            out.append(DIFF_OPEN)
            span_open = True
        elif not novel and span_open:
            out.append(DIFF_CLOSE)
            span_open = False
        out.append(word)

    if span_open:
        out.append(DIFF_CLOSE)

    return " ".join(out)


def setup_diff_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Register <diff> and </diff> special tokens with the tokenizer if not already present."""
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_DIFF_TOKENS})
    return tokenizer


def resize_with_mean_init(model: PreTrainedModel, vocab_size: int) -> None:
    """Resize token embeddings with mean initialization for novel special tokens."""
    old_size = model.get_input_embeddings().weight.shape[0]
    if old_size == vocab_size:
        return
    model.resize_token_embeddings(vocab_size)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[old_size:] = emb[:old_size].mean(dim=0, keepdim=True)


def encode_pair(
    claim: str,
    evidence: str,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 256,
    use_diff_markers: bool = True,
) -> Dict[str, torch.Tensor]:
    """Encode a claim-evidence pair with segment masks for pair pooling."""
    processed_claim = mark_claim_diff(claim, evidence) if use_diff_markers else claim
    enc = tokenizer(processed_claim, evidence, truncation=True, padding="max_length", max_length=max_len)
    seq_ids = enc.sequence_ids()

    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "claim_mask": torch.tensor([1 if s == 0 else 0 for s in seq_ids], dtype=torch.long),
        "evidence_mask": torch.tensor([1 if s == 1 else 0 for s in seq_ids], dtype=torch.long),
    }
