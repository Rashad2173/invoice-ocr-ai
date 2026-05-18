import re
from typing import List, Optional, Tuple

from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    text = (text or "").upper()
    for t, e in {
        "Ğ": "G", "Ü": "U", "Ş": "S", "İ": "I", "Ö": "O", "Ç": "C",
        "ğ": "G", "ü": "U", "ş": "S", "ı": "I", "ö": "O", "ç": "C"
    }.items():
        text = text.replace(t, e)
    text = re.sub(r"[^A-Z0-9 \-\./:%₺]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def join_split_digits(text: str) -> str:
    return re.sub(r"(?<=[\dA-Z])\s+(?=[\dA-Z\-])", "", (text or ""), flags=re.IGNORECASE)


def is_phone(v: str) -> bool:
    d = re.sub(r"\D", "", v or "")
    if d.startswith(("444", "0212", "0216", "0850", "0800")):
        return True
    if d.startswith("0") and 10 <= len(d) <= 11:
        return True
    return False


def is_date_token(v: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[./\-]\d{1,2}([./\-]\d{2,4})?", v.strip()))


def is_date_value(v: str) -> bool:
    return bool(re.search(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b", str(v or "")))


def digit_count(v: str) -> int:
    return len(re.findall(r"\d", v or ""))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def label_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    scores = [
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ]
    return max(scores)


def best_label_match(text: str, label_variants: List[str], min_score: float = 80.0) -> Tuple[Optional[str], float]:
    best_label = None
    best_score = 0.0
    norm_text = normalize_text(text)
    for lbl in label_variants:
        sc = label_similarity(norm_text, lbl)
        if sc > best_score:
            best_score = sc
            best_label = lbl
    if best_score >= min_score:
        return best_label, best_score
    return None, best_score


def normalize_id_text(text: str) -> str:
    text = join_split_digits(text or "")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def has_alpha(v: str) -> bool:
    return any(c.isalpha() for c in (v or ""))
