"""slot_detector.py — Deterministic rule-based conflict-slot detector for weak supervision.

Assigns conflict-slot labels to refuted claim-evidence pairs based on a prioritized rule cascade:
1. Non-refuted claims -> NONE
2. Temporal conflicts (mismatched 4-digit years) -> TEMPORAL
3. Numerical discrepancies (mismatched numeric quantities) -> NUMERIC
4. Polarity reversals (presence of negation markers) -> NEGATION
5. Named entity differences (mismatched capitalized multi-word phrases) -> ENTITY
6. Fallback for refuted claims -> ENTITY
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set

VERDICT_MAP = {"Supported": 0, "Refuted": 1, "NEI": 2}
ID2VERDICT = {0: "Supported", 1: "Refuted", 2: "NEI"}

CONFLICT_MAP = {
    "NONE": 0,
    "TEMPORAL": 1,
    "NUMERIC": 2,
    "NEGATION": 3,
    "ENTITY": 4,
}
ID2CONFLICT = {v: k for k, v in CONFLICT_MAP.items()}

NEGATION_WORDS: List[str] = [
    "không", "chưa", "chẳng", "không phải", "chưa từng", "sai sự thật",
    "bác bỏ", "không có", "chưa hề", "không thể",
]


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese Unicode to NFC and clean extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_years(text: str) -> List[int]:
    """Extract 4-digit years between 1600 and 2099."""
    matches = re.findall(r"\b((?:16|17|18|19|20)\d{2})\b", str(text))
    return sorted({int(y) for y in matches})


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values, excluding 4-digit calendar years."""
    raw_matches = re.findall(r"\b\d+(?:[.,]\d+)?\b", str(text))
    nums: Set[float] = set()
    for raw in raw_matches:
        try:
            val = float(raw.replace(",", "."))
            if 1600 <= val <= 2099 and val.is_integer():
                continue
            nums.add(val)
        except ValueError:
            continue
    return sorted(nums)


def contains_negation(text: str) -> bool:
    """Check if the text contains Vietnamese negation indicators."""
    lower = text.lower()
    for w in NEGATION_WORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", lower):
            return True
    return False


def extract_potential_entities(text: str) -> List[str]:
    """Extract capitalized multi-word sequences as named entity candidates."""
    pattern = (
        r"\b[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
        r"[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]*"
        r"(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
        r"[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]*)+\b"
    )
    found = re.findall(pattern, text)
    return list(dict.fromkeys(found))


def detect_conflict_slot(claim: str, evidence: str, verdict: str) -> str:
    """Identify the fine-grained conflict slot between claim and evidence."""
    v_norm = verdict.strip().capitalize()
    if v_norm != "Refuted":
        return "NONE"

    # 1. Temporal conflict
    c_years = extract_years(claim)
    e_years = extract_years(evidence)
    if c_years and e_years and set(c_years) != set(e_years):
        return "TEMPORAL"

    # 2. Numeric conflict
    c_nums = extract_numbers(claim)
    e_nums = extract_numbers(evidence)
    if c_nums and e_nums and set(c_nums) != set(e_nums):
        return "NUMERIC"

    # 3. Negation polarity conflict
    if contains_negation(claim) != contains_negation(evidence):
        return "NEGATION"

    # 4. Entity conflict
    c_ents = extract_potential_entities(claim)
    e_ents = extract_potential_entities(evidence)
    if c_ents and e_ents and set(c_ents) != set(e_ents):
        return "ENTITY"

    return "ENTITY"
