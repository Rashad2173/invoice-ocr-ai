import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from normalization import (
    best_label_match,
    clamp01,
    digit_count,
    has_alpha,
    is_date_token,
    is_date_value,
    is_phone,
    join_split_digits,
    label_similarity,
    normalize_id_text,
    normalize_text,
)
from ocr_engine import OCRItem


def spatial_proximity_score(label_item, value_item, max_y_dist: float = 220.0, col_tol: float = 80.0) -> float:
    dy = max(0.0, value_item.y1 - label_item.y2)
    dx = abs(value_item.cx - label_item.cx)
    y_score = 1.0 - min(dy / max_y_dist, 1.0)
    x_score = 1.0 - min(dx / col_tol, 1.0)
    return clamp01(0.6 * y_score + 0.4 * x_score)


def combined_field_score(
    label_score_100: float,
    label_ocr_conf: float,
    value_ocr_conf: float,
    spatial_score: float,
    value_length_score: float = 1.0,
) -> float:
    return clamp01(
        0.35 * (label_score_100 / 100.0) +
        0.15 * clamp01(label_ocr_conf) +
        0.25 * clamp01(value_ocr_conf) +
        0.15 * clamp01(spatial_score) +
        0.10 * clamp01(value_length_score)
    )

def find_column_value_below(
    items: List[OCRItem],
    label_item: OCRItem,
    col_tol: float = 40.0,
    max_y_dist: float = 200.0,
    skip_rows: int = 0
) -> List[OCRItem]:
    cands = [
        it for it in items
        if it.y1 > label_item.y2 - 5
        and it.y1 <= label_item.y2 + max_y_dist
        and it.cx >= label_item.cx - col_tol
        and it.cx <= label_item.cx + col_tol
    ]
    cands.sort(key=lambda x: x.y1)
    if skip_rows > 0 and len(cands) > skip_rows:
        cands = cands[skip_rows:]
    return cands


def find_label_items_fuzzy(
    items: List[OCRItem],
    label_variants: List[str],
    min_label_score: float = 82.0,
    min_ocr_conf: float = 0.30,
) -> List[Tuple[OCRItem, str, float]]:
    matches = []
    for it in items:
        if it.confidence < min_ocr_conf:
            continue
        matched, score = best_label_match(it.text, label_variants, min_score=min_label_score)
        if matched:
            matches.append((it, matched, score))
    matches.sort(key=lambda x: (x[2], x[0].confidence), reverse=True)
    return matches


def find_line_matches_fuzzy(lines: List[str], label_variants: List[str], min_label_score: float = 82.0) -> List[Tuple[int, str, float]]:
    hits = []
    for i, line in enumerate(lines):
        matched, score = best_label_match(line, label_variants, min_score=min_label_score)
        if matched:
            hits.append((i, matched, score))
    hits.sort(key=lambda x: x[2], reverse=True)
    return hits

def numeric_body(v: str) -> str:
    return re.sub(r"\D", "", v or "")


def contains_other_id(a: Optional[str], b: Optional[str], min_len: int = 6) -> bool:
    da = numeric_body(a)
    db = numeric_body(b)
    if not da or not db:
        return False
    if len(da) < min_len or len(db) < min_len:
        return False
    return da in db or db in da


def nearest_line_index_for_item(item, lines):
    txt = normalize_text(getattr(item, "text", "") or "")
    if not txt:
        return None
    for i, line in enumerate(lines):
        ln = normalize_text(line)
        if txt == ln or txt in ln or ln in txt:
            return i
    return None


def within_line_gap(label_item, value_item, lines, max_gap=6):
    li = nearest_line_index_for_item(label_item, lines)
    vi = nearest_line_index_for_item(value_item, lines)
    if li is None or vi is None:
        return True
    return (vi - li) <= max_gap and vi >= li


def within_one_line_above(label_item, value_item, lines):
    li = nearest_line_index_for_item(label_item, lines)
    vi = nearest_line_index_for_item(value_item, lines)
    if li is None or vi is None:
        return True
    return 0 < (li - vi) <= 1


@dataclass
class FieldCandidate:
    field: str
    value: str
    score: float
    source: str = ""
    label: str = ""
    context: str = ""

    @property
    def digits(self) -> str:
        return numeric_body(self.value)


FIELD_MIN_SCORE = {
    "sozlesme_no": 0.56,
    "tesisat_no": 0.54,
    "musteri_no": 0.58,
    "fatura_no": 0.62,
    "belge_no": 0.60,
}

FIELD_PRIORITY = {
    "sozlesme_no": 5,
    "tesisat_no": 4,
    "musteri_no": 3,
    "fatura_no": 2,
    "belge_no": 1,
}


def values_conflict(a: Optional[str], b: Optional[str], min_len: int = 6) -> bool:
    da = numeric_body(a)
    db = numeric_body(b)
    if not da or not db:
        return False
    if len(da) < min_len or len(db) < min_len:
        return False
    return da == db or da in db or db in da


def add_field_candidate(
    candidates: List[FieldCandidate],
    field: str,
    value: Optional[str],
    score: float,
    source: str,
    label: str = "",
    context: str = "",
    min_score: float = 0.40,
):
    if not value:
        return
    value = normalize_id_text(value)
    if not value:
        return
    score = clamp01(score)
    if score < min_score:
        return
    candidates.append(FieldCandidate(
        field=field,
        value=value,
        score=score,
        source=source,
        label=label,
        context=context,
    ))


def dedupe_candidates(candidates: List[FieldCandidate]) -> List[FieldCandidate]:
    grouped: Dict[Tuple[str, str], FieldCandidate] = {}
    source_map: Dict[Tuple[str, str], set] = {}

    for c in candidates:
        key = (c.field, c.value)
        if key not in grouped:
            grouped[key] = c
            source_map[key] = {c.source}
        else:
            source_map[key].add(c.source)
            if c.score > grouped[key].score:
                grouped[key] = c

    result = []
    for key, cand in grouped.items():
        sources = source_map[key]
        boosted = FieldCandidate(
            field=cand.field,
            value=cand.value,
            score=clamp01(cand.score + (0.06 if len(sources) >= 2 else 0.0)),
            source="+".join(sorted(sources)),
            label=cand.label,
            context=cand.context,
        )
        result.append(boosted)

    result.sort(
        key=lambda c: (
            c.score,
            has_alpha(c.value),
            digit_count(c.value),
            len(c.value),
            FIELD_PRIORITY.get(c.field, 0),
        ),
        reverse=True
    )
    return result


def best_candidate(candidates: List[FieldCandidate], field: str, min_score: Optional[float] = None) -> Tuple[Optional[str], float]:
    min_score = FIELD_MIN_SCORE.get(field, 0.55) if min_score is None else min_score
    pool = [c for c in dedupe_candidates(candidates) if c.field == field]
    if not pool:
        return None, 0.0
    c = pool[0]
    if c.score < min_score:
        return None, 0.0
    return c.value, c.score


def resolve_field_candidates(candidate_map: Dict[str, List[FieldCandidate]]) -> Tuple[
    Optional[str], float,
    Optional[str], float,
    Optional[str], float,
    Optional[str], float,
    Optional[str], float,
]:
    """
    Her alan için aday listesi alır.
    Aynı / iç içe geçen değerleri tek alana verir.
    Skoru yüksek olan alan değeri alır, diğer alan kendi ikinci / üçüncü adayına düşer.
    """
    fields = ["sozlesme_no", "tesisat_no", "musteri_no", "fatura_no", "belge_no"]

    flattened: List[FieldCandidate] = []
    for f in fields:
        flattened.extend(dedupe_candidates(candidate_map.get(f, [])))

    flattened.sort(
        key=lambda c: (
            c.score,
            has_alpha(c.value),
            digit_count(c.value),
            FIELD_PRIORITY.get(c.field, 0),
        ),
        reverse=True
    )

    selected: Dict[str, FieldCandidate] = {}
    used_values: List[str] = []

    for cand in flattened:
        if cand.field in selected:
            continue

        min_sc = FIELD_MIN_SCORE.get(cand.field, 0.55)
        if cand.score < min_sc:
            continue

        conflict = any(values_conflict(cand.value, used, min_len=6) for used in used_values)
        if conflict:
            continue

        selected[cand.field] = cand
        used_values.append(cand.value)

    def get(field: str) -> Tuple[Optional[str], float]:
        c = selected.get(field)
        if not c:
            return None, 0.0
        return c.value, c.score

    soz, soz_sc = get("sozlesme_no")
    tes, tes_sc = get("tesisat_no")
    mus, mus_sc = get("musteri_no")
    fat, fat_sc = get("fatura_no")
    bel, bel_sc = get("belge_no")
    return soz, soz_sc, tes, tes_sc, mus, mus_sc, fat, fat_sc, bel, bel_sc


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK CONTEXT FİLTRELERİ
# ─────────────────────────────────────────────────────────────────────────────

COMMON_BAD_ID_CONTEXT = [
    "E-ARSIV FATURA", "DOGAL GAZ FATURASI", "SU FATURASI", "ELEKTRIK FATURASI",
    "FATURA TARIHI", "FATURA TUTARI", "FATURA GUN SAYISI", "FATURA DONEMI",
    "SAYAC", "OKUMA", "ILK OKUMA", "SON OKUMA", "OKUYUCU",
    "SERI NO", "SERI NUMARASI", "ENDEKS", "ISARET", "DURUM",
    "KWH", "M3", "SM3", "KDV", "MATRAH", "BIRIM FIYAT", "TUKETIM",
]


def fuzzy_has_bad_context(text: str, bad_list: List[str] = COMMON_BAD_ID_CONTEXT, threshold: int = 88) -> bool:
    n = normalize_text(text)
    if not n:
        return False
    if any(b in n for b in bad_list):
        return True
    for b in bad_list:
        if fuzz.partial_ratio(n, normalize_text(b)) >= threshold:
            return True
    return False


def same_row_neighbors_text(item: OCRItem, items: List[OCRItem], y_tol: float = 28.0) -> str:
    neighbors = [
        it.text for it in items
        if it is not item and abs(it.cy - item.cy) <= y_tol
    ]
    return " ".join(neighbors)


def value_has_bad_neighbor(item: OCRItem, items: List[OCRItem], extra_bad: Optional[List[str]] = None) -> bool:
    ctx = same_row_neighbors_text(item, items)
    bad = COMMON_BAD_ID_CONTEXT + (extra_bad or [])
    return fuzzy_has_bad_context(ctx, bad_list=bad, threshold=88)


# ─────────────────────────────────────────────────────────────────────────────
# SÖZLEŞME NO
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_ID_RE = re.compile(r"\b\d{6,20}\b")
ALPHANUM_ID_RE = re.compile(r"\b[A-Z]{1,4}\d[A-Z0-9\-]{4,24}\b", re.IGNORECASE)
LONG_ALPHA_RE = re.compile(r"\b[A-Z]\d{10,20}\b", re.IGNORECASE)
SAYAC_SERI_RE = re.compile(r"^\d{8,20}$")
ID_BLACKLIST_RE = re.compile(
    r"\b(FATURA|NUMARASI|BELGE|TARIH|SAAT|VRS|BILGILERI|TUTAR|TOPLAM|"
    r"ELEKTRIK|PERAKENDE|DAGITIM|ISTANBUL|ANADOLU|SATIS|BARBAROS|"
    r"MAH|ATASEHIR|KOZYATAGI|MERSIS|TICARET|ENDEKS|GUNDUZ|PUANT|GECE)\b",
    re.IGNORECASE
)


def extract_best_numeric_id(text: str, min_digits: int = 6) -> Optional[str]:
    text = join_split_digits(text)
    cands = []
    for m in NUMERIC_ID_RE.finditer(text):
        v = m.group()
        if len(v) < min_digits or is_phone(v) or is_date_token(v):
            continue
        cands.append(v)
    return max(cands, key=len) if cands else None


def extract_best_alphanum_id(text: str) -> Optional[str]:
    text = join_split_digits(text)
    cands = []
    for m in ALPHANUM_ID_RE.finditer(text):
        v = m.group()
        if is_phone(v) or digit_count(v) < 5:
            continue
        cands.append(v)
    return max(cands, key=len) if cands else None


def extract_best_long_alpha_id(text: str) -> Optional[str]:
    text = join_split_digits(text)
    cands = []
    for m in LONG_ALPHA_RE.finditer(text):
        v = m.group()
        if ID_BLACKLIST_RE.search(v):
            continue
        cands.append(v)
    return max(cands, key=len) if cands else None


def extract_any_id(text: str, prefer_alphanum: bool = False, min_digits: int = 6) -> Optional[str]:
    if prefer_alphanum:
        v = extract_best_alphanum_id(text)
        if v:
            return v
        v = extract_best_long_alpha_id(text)
        if v:
            return v
    return extract_best_numeric_id(text, min_digits=min_digits)


def looks_like_belge_id(v: str) -> bool:
    v = normalize_id_text(v)
    if not v:
        return False
    if is_date_value(v):
        return False
    if ID_BLACKLIST_RE.search(v):
        return False

    digits = re.sub(r"\D", "", v)
    if len(digits) < 7:
        return False

    if re.fullmatch(r"[A-Z]{1,3}\d{8,20}", v, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z]\d{10,20}", v, re.IGNORECASE):
        return True
    if digits.isdigit() and len(digits) >= 14:
        return True

    return False


def looks_like_fatura_id(v: str) -> bool:
    v = normalize_id_text(v)
    if not v:
        return False
    if ID_BLACKLIST_RE.search(v):
        return False

    digits = re.sub(r"\D", "", v)
    if len(digits) < 8:
        return False
    if is_phone(digits):
        return False

    if re.fullmatch(r"[A-Z]{1,3}\d{8,20}", v, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{8,20}[A-Z]?", v, re.IGNORECASE):
        return True

    return False

AD_PATTERN = re.compile(
    r"www\.|\.istanbul|\.com|\.net|\.org|https?://"
    r"|dijital\s+fatura|kent\s+orman|sosyal\s+belediye"
    r"|milyar\s+tl|ne\s+mutlu\s+istanbul|bir\s+kent\s+orman"
    r"|kurtarabilirsiniz",
    re.IGNORECASE
)

FIRMA_CORROBORATION: Dict[str, List[str]] = {
    "ISKI": ["SU FATURASI", "KANALIZASYON", "SU VE KANAL", "SOZLESME NUMARASI", "SU BIRIM", "ATIK SU", "ABONE BILGILERI", "FATURA NUMARASI"],
    "IGDAS": ["DOGAL GAZ FATURASI", "GAZ FATURASI", "SOZLESME HESABI", "TUKETIM NOKTASI", "DOGALGAZ", "GAZ DAGITIM", "E-ARSIV FATURA"],
    "ENERJISA": ["ELEKTRIK", "KWH", "TUKETIM", "SOZLESME HESAP NO", "ENDEKS", "PUANT", "GUNDUZ", "GECE", "MUSTERI NO", "TEKIL KOD"],
    "BEDAS": ["ELEKTRIK", "KWH", "ENDEKS", "DAGITIM", "BOGAZ"],
    "AYEDAS": ["ELEKTRIK", "KWH", "ENDEKS", "ANADOLU", "DAGITIM"],
    "BUSKI": ["SU FATURASI", "SU VE KANAL", "BURSA", "KANALIZASYON"],
}


def corroboration_score(key: str, full_norm: str) -> float:
    kws = FIRMA_CORROBORATION.get(key, [])
    if not kws:
        return 0.5
    return sum(1 for kw in kws if kw in full_norm) / len(kws)


def detect_firma(full_text: str, firmalar: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    clean = [l for l in lines if not AD_PATTERN.search(l)]
    search_txt = " ".join(clean[:30]).upper()
    search_nrm = normalize_text(search_txt)
    full_nrm = normalize_text(full_text)

    best_can: Optional[str] = None
    best_typ: Optional[str] = None
    best_sc: float = 0.0

    for firma in firmalar:
        canonical = firma["canonical_name"]
        ftype = firma["type"]
        for alias in [canonical] + firma.get("aliases", []):
            au = alias.upper()
            an = normalize_text(alias)

            if au in search_txt:
                key = an.split()[0] if an else ""
                sc = 1.0 + corroboration_score(key, full_nrm) + len(au) / 1000.0
                if sc > best_sc:
                    best_sc, best_can, best_typ = sc, canonical, ftype
                break

            if an and len(an) >= 4 and an in search_nrm:
                key = an.split()[0] if an else ""
                sc = 0.8 + corroboration_score(key, full_nrm) + len(an) / 1000.0
                if sc > best_sc:
                    best_sc, best_can, best_typ = sc, canonical, ftype
                break

    if best_can is None:
        FALLBACK = [
            ("ISKI", "İSKİ İSTANBUL SU VE KANALİZASYON İDARESİ", "su"),
            ("IGDAS", "İGDAŞ İSTANBUL GAZ DAĞITIM SANAYİ VE TİCARET A.Ş.", "dogalgaz"),
            ("ENERJISA", "ENERJİSA İSTANBUL ANADOLU YAKASI ELEKTRİK PERAKENDE SATIŞ A.Ş.", "elektrik"),
            ("BEDAS", "BEDAŞ BOĞAZİÇİ ELEKTRİK DAĞITIM A.Ş.", "elektrik"),
            ("AYEDAS", "AYEDAŞ ANADOLU YAKASI ELEKTRİK DAĞITIM A.Ş.", "elektrik"),
            ("BUSKI", "BUSKİ BURSA SU VE KANALİZASYON İDARESİ", "su"),
        ]
        bc = 0.0
        for key, canon, ftype in FALLBACK:
            cs = corroboration_score(key, full_nrm)
            if cs > bc and cs >= 0.25:
                bc, best_can, best_typ = cs, canon, ftype

    return best_can, best_typ

SOZLESME_LABELS = [
    "SOZLESME HESABI",
    "SOZLESME HESAP NO",
    "SOZLESME NUMARASI",
    "SOZLESME NO",
    "HESAP NO",
]

TESISAT_LABELS = [
    "TEKIL KOD",
    "TEKI KOD",
    "TEK KOD",
    "TEKILKOD",
    "TESISAT NO",
    "TEKIL KOD TESISAT",
    "TEKIL KOD TESISAT NO",
    "TUKETIM NOKTASI",
    "TUKETIM NOKTASI NO",
]

MUSTERI_LABELS = [
    "MUSTERI NO",
    "MUSTERI NUMARASI",
    "MUSTERI KODU",
    "ABONE NO",
    "ABONE NUMARASI",
]

FATURA_LABELS = [
    "FATURA NUMARASI",
    "FATURA NO",
    "FTR NO",
]

BELGE_LABELS = [
    "BELGE NO",
    "BELGE NUMARASI",
    "BLG NO",
]

SON_ODEME_LABELS = [
    "SON ODEME TARIHI",
    "SON ODEME TARIH",
    "SON ODEME TAR",
    "ODEME SON TARIHI",
    "SON TARIHI",
    "SON ODEME",
]

SOZLESME_PATS = [
    r"S[ÖO]ZLE[SŞ]ME\s*HESAB[I]?",
    r"SOZLESME\s*HESABI",
    r"S[ÖO]ZLE[SŞ]ME\s*HESAP\s*NO",
    r"S[ÖO]ZLE[SŞ]ME\s*NUMARASI",
    r"SOZLESME\s*NUMARASI",
    r"SOZIESME\s*NUMARASI",
    r"S\w{5,8}\s*NUMARASI",
    r"S[ÖO]ZLE[SŞ]ME\s*NO(?!\s*SIRA)",
    r"SOZLESME\s*NO",
    r"SOZIESME\s*NO",
    r"HESAP\s*NO(?!\s*SIRA)",
]


def extract_sozlesme_candidate(text: str) -> Optional[str]:
    text = normalize_id_text(text)
    pats = [
        r"\b[A-Z]{1,4}-\d{5,}-\d{1,4}\b",
        r"\b[A-Z]{1,4}-\d{6,20}\b",
        r"\b[A-Z]{1,4}\d{6,20}\b",
    ]
    cands = []
    for pat in pats:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = normalize_id_text(m.group())
            if digit_count(v) >= 6 and not is_phone(v):
                cands.append(v)

    v = extract_best_alphanum_id(text)
    if v:
        v = normalize_id_text(v)
        if digit_count(v) >= 6 and not is_phone(v):
            cands.append(v)

    if cands:
        cands = sorted(set(cands), key=lambda x: (has_alpha(x), len(x), digit_count(x)), reverse=True)
        return cands[0]

    v = extract_best_numeric_id(text, min_digits=6)
    if v and not is_phone(v):
        return v
    return None


def extract_sozlesme_candidates(lines: List[str], items: List[OCRItem]) -> List[FieldCandidate]:
    field = "sozlesme_no"
    candidates: List[FieldCandidate] = []

    label_hits = find_label_items_fuzzy(items, SOZLESME_LABELS, min_label_score=80.0, min_ocr_conf=0.25)

    for li, matched_label, label_score in label_hits:
        # Aynı satır sağ taraf
        same_row = [
            it for it in items
            if it is not li and abs(it.cy - li.cy) <= 24 and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:4]:
            if value_has_bad_neighbor(bit, items):
                continue
            val = extract_sozlesme_candidate(bit.joined)
            if not val:
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=55.0, col_tol=320.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 12.0))
            score = clamp01(score + 0.07 + (0.05 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_same_row", matched_label, f"{li.text} {bit.text}")

        # Alt / yakın kolon
        for bit in find_column_value_below(items, li, col_tol=100.0, max_y_dist=230.0)[:5]:
            if value_has_bad_neighbor(bit, items):
                continue
            if not within_line_gap(li, bit, lines, max_gap=5):
                continue
            val = extract_sozlesme_candidate(bit.joined)
            if not val:
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=230.0, col_tol=100.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 12.0))
            score = clamp01(score + (0.05 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_below", matched_label, f"{li.text} {bit.text}")

    line_hits = find_line_matches_fuzzy(lines, SOZLESME_LABELS, min_label_score=80.0)
    for i, matched_label, line_score in line_hits:
        cur = join_split_digits(lines[i])
        val = extract_sozlesme_candidate(cur)
        if val:
            score = clamp01(0.58 + 0.32 * (line_score / 100.0) + (0.04 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "line_same", matched_label, lines[i])

        for j in range(i + 1, min(i + 7, len(lines))):
            val = extract_sozlesme_candidate(join_split_digits(lines[j]))
            if not val:
                continue
            score = clamp01(0.43 + 0.42 * (line_score / 100.0) - ((j - i - 1) * 0.03) + (0.04 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "line_below", matched_label, f"{lines[i]} {lines[j]}")

    # Regex fallback
    for li in items:
        if not any(re.search(pat, li.text, re.IGNORECASE) for pat in SOZLESME_PATS):
            continue
        for bit in find_column_value_below(items, li, col_tol=100.0, max_y_dist=230.0)[:5]:
            val = extract_sozlesme_candidate(bit.joined)
            if val:
                score = clamp01(0.62 + min(0.20, digit_count(val) / 80.0) + (0.05 if has_alpha(val) else 0.0))
                add_field_candidate(candidates, field, val, score, "regex_item_below", "SOZLESME_REGEX", f"{li.text} {bit.text}")

    for i, line in enumerate(lines):
        lj = join_split_digits(line)
        if not any(re.search(pat, lj, re.IGNORECASE) for pat in SOZLESME_PATS):
            continue
        for j in range(i, min(i + 7, len(lines))):
            val = extract_sozlesme_candidate(join_split_digits(lines[j]))
            if val:
                score = clamp01(0.62 - ((j - i) * 0.04) + (0.05 if has_alpha(val) else 0.0))
                add_field_candidate(candidates, field, val, score, "regex_line", "SOZLESME_REGEX", f"{line} {lines[j]}")

    return dedupe_candidates(candidates)


def extract_sozlesme_no(lines: List[str], items: List[OCRItem]) -> Tuple[Optional[str], float]:
    return best_candidate(extract_sozlesme_candidates(lines, items), "sozlesme_no")


# ─────────────────────────────────────────────────────────────────────────────
# TESİSAT NO / TEKİL KOD
# ─────────────────────────────────────────────────────────────────────────────

TESISAT_PATS = [
    r"TEK[İI]L\s*KOD\s*/\s*TES[İI]SAT",
    r"TEK[İI]L\s*KOD",
    r"TEK[İI]?\s*KOD",
    r"TEKI\s*KOD",
    r"TEKIL?\s*KOD",
    r"TEK.\s*KOD",
    r"TEKIL\s*KOD",
    r"TES[İI]SAT\s*NO(?!\s*[A-Z]{3,})",
    r"TESISAT\s*NO",
    r"T[ÜU]KET[İI]M\s*NOKTASI",
    r"TUKETIM\s*NOKTASI",
    r"TUK\s*NOKTA",
]


def validate_tesisat_candidate(val: str) -> bool:
    digits = numeric_body(val)
    if not digits or len(digits) < 6:
        return False
    if is_phone(digits):
        return False
    return True


def extract_tesisat_candidates(lines: List[str], items: List[OCRItem]) -> List[FieldCandidate]:
    field = "tesisat_no"
    candidates: List[FieldCandidate] = []

    label_hits = find_label_items_fuzzy(items, TESISAT_LABELS, min_label_score=74.0, min_ocr_conf=0.20)
    for li, matched_label, label_score in label_hits:
        same_row = [
            it for it in items
            if it is not li and abs(it.cy - li.cy) <= 24 and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:4]:
            val = extract_best_numeric_id(bit.joined, min_digits=6)
            if not val or not validate_tesisat_candidate(val):
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=55.0, col_tol=300.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 10.0))
            add_field_candidate(candidates, field, val, clamp01(score + 0.06), "item_same_row", matched_label, f"{li.text} {bit.text}")

        for bit in find_column_value_below(items, li, col_tol=115.0, max_y_dist=240.0)[:5]:
            if not within_line_gap(li, bit, lines, max_gap=5):
                continue
            val = extract_best_numeric_id(bit.joined, min_digits=6)
            if not val or not validate_tesisat_candidate(val):
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=240.0, col_tol=115.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 10.0))
            add_field_candidate(candidates, field, val, score, "item_below", matched_label, f"{li.text} {bit.text}")

    line_hits = find_line_matches_fuzzy(lines, TESISAT_LABELS, min_label_score=74.0)
    for i, matched_label, line_score in line_hits:
        val = extract_best_numeric_id(join_split_digits(lines[i]), min_digits=6)
        if val and validate_tesisat_candidate(val):
            add_field_candidate(candidates, field, val, clamp01(0.56 + 0.30 * (line_score / 100.0)), "line_same", matched_label, lines[i])

        for j in range(i + 1, min(i + 7, len(lines))):
            val = extract_best_numeric_id(join_split_digits(lines[j]), min_digits=6)
            if val and validate_tesisat_candidate(val):
                score = clamp01(0.42 + 0.42 * (line_score / 100.0) - ((j - i - 1) * 0.03))
                add_field_candidate(candidates, field, val, score, "line_below", matched_label, f"{lines[i]} {lines[j]}")

    for li in items:
        if not any(re.search(pat, li.text, re.IGNORECASE) for pat in TESISAT_PATS):
            continue
        for bit in find_column_value_below(items, li, col_tol=115.0, max_y_dist=240.0)[:5]:
            val = extract_best_numeric_id(bit.joined, min_digits=6)
            if val and validate_tesisat_candidate(val):
                add_field_candidate(candidates, field, val, min(1.0, 0.58 + digit_count(val) / 30.0), "regex_item_below", "TESISAT_REGEX", f"{li.text} {bit.text}")

    return dedupe_candidates(candidates)


def extract_tesisat_no(lines: List[str], items: List[OCRItem]) -> Tuple[Optional[str], float]:
    return best_candidate(extract_tesisat_candidates(lines, items), "tesisat_no")


# ─────────────────────────────────────────────────────────────────────────────
# MÜŞTERİ / ABONE NO
# ─────────────────────────────────────────────────────────────────────────────

MUSTERI_PATS = [
    r"M[ÜU][SŞ]TER[İI]\s*NO(?!\s*[A-Z]{2,})",
    r"MUSTERI\s*NO",
    r"M[ÜU][SŞ]TER[İI]\s*NUMARASI",
    r"MUSTERI\s*NUMARASI",
    r"M[ÜU][SŞ]TER[İI]\s*KODU",
    r"MUSTERI\s*KODU",
    r"ABONE\s*NO(?!\s*[A-Z]{2,})",
    r"ABONE\s*NUMARASI",
]


def validate_musteri_candidate(val: str) -> bool:
    clean = numeric_body(val)
    if not clean.isdigit():
        return False
    if len(clean) < 8:
        return False
    if is_phone(clean):
        return False
    if clean.startswith("0") and len(clean) <= 8:
        return False
    return True


def extract_musteri_candidates(lines: List[str], items: List[OCRItem]) -> List[FieldCandidate]:
    field = "musteri_no"
    candidates: List[FieldCandidate] = []

    label_hits = find_label_items_fuzzy(items, MUSTERI_LABELS, min_label_score=88.0, min_ocr_conf=0.30)
    regex_label_items = []

    for li, matched_label, label_score in label_hits:
        same_row = [
            it for it in items
            if it is not li and abs(it.cy - li.cy) <= 24 and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:4]:
            val = extract_best_numeric_id(bit.joined, min_digits=8)
            if not val or not validate_musteri_candidate(val):
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=55.0, col_tol=300.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 9.0))
            add_field_candidate(candidates, field, val, clamp01(score + 0.06), "item_same_row", matched_label, f"{li.text} {bit.text}")

        for bit in find_column_value_below(items, li, col_tol=85.0, max_y_dist=190.0)[:5]:
            if not within_line_gap(li, bit, lines, max_gap=5):
                continue
            val = extract_best_numeric_id(bit.joined, min_digits=8)
            if not val or not validate_musteri_candidate(val):
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=190.0, col_tol=85.0)
            score = combined_field_score(label_score, li.confidence, bit.confidence, spatial, min(1.0, digit_count(val) / 9.0))
            add_field_candidate(candidates, field, val, score, "item_below", matched_label, f"{li.text} {bit.text}")

    for li in items:
        if any(re.search(pat, li.text, re.IGNORECASE) for pat in MUSTERI_PATS):
            regex_label_items.append(li)

    for li in regex_label_items:
        for bit in find_column_value_below(items, li, col_tol=85.0, max_y_dist=190.0)[:5]:
            val = extract_best_numeric_id(bit.joined, min_digits=8)
            if val and validate_musteri_candidate(val):
                add_field_candidate(candidates, field, val, min(1.0, 0.62 + digit_count(val) / 40.0), "regex_item_below", "MUSTERI_REGEX", f"{li.text} {bit.text}")

    for i, line in enumerate(lines):
        lj = join_split_digits(line)
        matched, score = best_label_match(line, MUSTERI_LABELS, min_score=90.0)
        if matched:
            val = extract_best_numeric_id(lj, min_digits=8)
            if val and validate_musteri_candidate(val):
                add_field_candidate(candidates, field, val, clamp01(0.58 + 0.32 * (score / 100.0)), "line_same", matched, line)

            for j in range(i + 1, min(i + 5, len(lines))):
                val = extract_best_numeric_id(join_split_digits(lines[j]), min_digits=8)
                if val and validate_musteri_candidate(val):
                    add_field_candidate(candidates, field, val, clamp01(0.42 + 0.40 * (score / 100.0) - ((j - i - 1) * 0.03)), "line_below", matched, f"{line} {lines[j]}")

        if any(re.search(pat, lj, re.IGNORECASE) for pat in MUSTERI_PATS):
            for j in range(i, min(i + 5, len(lines))):
                val = extract_best_numeric_id(join_split_digits(lines[j]), min_digits=8)
                if val and validate_musteri_candidate(val):
                    add_field_candidate(candidates, field, val, clamp01(0.64 - ((j - i) * 0.04)), "regex_line", "MUSTERI_REGEX", f"{line} {lines[j]}")

    return dedupe_candidates(candidates)


def extract_musteri_no(lines: List[str], items: List[OCRItem]) -> Tuple[Optional[str], float]:
    return best_candidate(extract_musteri_candidates(lines, items), "musteri_no")


# ─────────────────────────────────────────────────────────────────────────────
# FATURA NO
# ─────────────────────────────────────────────────────────────────────────────

FATURA_LABEL_VARIANTS = [
    "FATURA NUMARASI",
    "FATURA NO",
    "FTR NO",
    "FALURA NUMARASI",
    "FALUA NUMARASI",
    "FATURA NURNARASI",
    "FALUA NURNARASI",
]

FATURA_BAD_CONTEXT = COMMON_BAD_ID_CONTEXT + [
    "BELGE TARIHI", "BELGE NO", "BELGE NUMARASI",
    "MERSIS", "VKN", "VERGI", "TICARET SICIL",
    "TESISAT", "SOZLESME", "MUSTERI", "ABONE", "TEKIL KOD",
]


def validate_fatura_value(val: str, sozlesme_no: Optional[str] = None) -> bool:
    val = normalize_id_text(val)
    if not val or is_date_value(val):
        return False
    if ID_BLACKLIST_RE.search(val):
        return False
    if fuzzy_has_bad_context(val, FATURA_BAD_CONTEXT, threshold=90):
        return False

    digits = numeric_body(val)
    if len(digits) < 8:
        return False
    if is_phone(digits):
        return False
    if re.match(r"^s?No\d", val, re.IGNORECASE):
        return False

    if sozlesme_no and values_conflict(val, sozlesme_no, min_len=6):
        return False

    return True


def extract_fatura_id(text: str, sozlesme_no: Optional[str] = None) -> Optional[str]:
    text = re.sub(
        r"\b(FATURA|FALURA|FALUA)\s*(NO|NUMARASI|NURNARASI)\b|\bFTR\s*NO\b",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )
    text = normalize_id_text(text)
    cands = []

    patterns = [
        r"\b[A-Z]{1,3}\d{8,20}[A-Z]?\b",
        r"\b\d{8,20}[A-Z]?\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = normalize_id_text(m.group())
            if validate_fatura_value(v, sozlesme_no=sozlesme_no):
                cands.append(v)

    if not cands:
        return None

    return sorted(
        set(cands),
        key=lambda x: (has_alpha(x), digit_count(x), len(x)),
        reverse=True
    )[0]


def fatura_value_has_bad_neighbor(item: OCRItem, items: List[OCRItem]) -> bool:
    neighbors = []
    for it in items:
        if it is item or abs(it.cy - item.cy) > 28.0:
            continue
        matched, score = best_label_match(it.text, FATURA_LABEL_VARIANTS, min_score=92.0)
        if matched and score >= 92.0:
            continue
        neighbors.append(it.text)
    return fuzzy_has_bad_context(" ".join(neighbors), bad_list=FATURA_BAD_CONTEXT, threshold=88)


def extract_fatura_candidates(lines: List[str], items: List[OCRItem], sozlesme_no: Optional[str] = None) -> List[FieldCandidate]:
    field = "fatura_no"
    candidates: List[FieldCandidate] = []

    label_hits = find_label_items_fuzzy(
        items,
        FATURA_LABEL_VARIANTS,
        min_label_score=82.0,
        min_ocr_conf=0.30
    )

    for li, matched_label, label_score in label_hits:
        if label_score < 92.0 and fuzzy_has_bad_context(li.text, FATURA_BAD_CONTEXT, threshold=88):
            continue

        # 1) Aynı satır sağ taraf — en güçlü aday
        same_row = [
            it for it in items
            if it is not li
            and abs(it.cy - li.cy) <= 24
            and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:4]:
            if fuzzy_has_bad_context(bit.text, FATURA_BAD_CONTEXT, threshold=90):
                continue
            if fatura_value_has_bad_neighbor(bit, items):
                continue
            if not within_line_gap(li, bit, lines, max_gap=2):
                continue

            val = extract_fatura_id(bit.joined, sozlesme_no=sozlesme_no)
            if not val:
                continue

            spatial = spatial_proximity_score(li, bit, max_y_dist=55.0, col_tol=320.0)
            score = combined_field_score(
                label_score, li.confidence, bit.confidence, spatial,
                value_length_score=min(1.0, digit_count(val) / 12.0)
            )
            score = clamp01(score + 0.08 + (0.06 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_same_row", matched_label, f"{li.text} {bit.text}")

        # 2) Alt satırlar / yakın kolon — ikinci öncelik
        below = find_column_value_below(
            items,
            li,
            col_tol=150.0,
            max_y_dist=185.0,
            skip_rows=0
        )[:5]

        for bit in below:
            if fuzzy_has_bad_context(bit.text, FATURA_BAD_CONTEXT, threshold=90):
                continue
            if fatura_value_has_bad_neighbor(bit, items):
                continue
            if not within_line_gap(li, bit, lines, max_gap=4):
                continue

            val = extract_fatura_id(bit.joined, sozlesme_no=sozlesme_no)
            if not val:
                continue

            spatial = spatial_proximity_score(li, bit, max_y_dist=185.0, col_tol=150.0)
            score = combined_field_score(
                label_score, li.confidence, bit.confidence, spatial,
                value_length_score=min(1.0, digit_count(val) / 12.0)
            )
            score = clamp01(score + (0.06 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_below", matched_label, f"{li.text} {bit.text}")

        # 3) SADECE EN SON: 1 satır üst — eğik fatura toleransı
        # Geniş üst arama yok. Sadece çok yakın ve aynı kolonda/yakında ise aday olur.
        above = [
            it for it in items
            if it is not li
            and it.y2 < li.y1
            and (li.y1 - it.y2) <= 90
            and (
                abs(it.cx - li.cx) <= 260
                or (it.x1 <= li.x2 + 180 and it.x2 >= li.x1 - 180)
            )
        ]
        above.sort(key=lambda x: x.y2, reverse=True)

        for bit in above[:4]:
            if not within_one_line_above(li, bit, lines):
                li_idx = nearest_line_index_for_item(li, lines)
                bit_idx = nearest_line_index_for_item(bit, lines)
                if li_idx is not None and bit_idx is not None and not (0 < (li_idx - bit_idx) <= 2):
                    continue
            if fuzzy_has_bad_context(bit.text, FATURA_BAD_CONTEXT, threshold=90):
                continue
            if fatura_value_has_bad_neighbor(bit, items):
                continue

            val = extract_fatura_id(bit.joined, sozlesme_no=sozlesme_no)
            if not val:
                continue

            vertical_gap = li.y1 - bit.y2
            x_gap = abs(bit.cx - li.cx)
            y_score = 1.0 - min(vertical_gap / 90.0, 1.0)
            x_score = 1.0 - min(x_gap / 260.0, 1.0)
            spatial = clamp01(0.55 * y_score + 0.45 * x_score)

            score = combined_field_score(
                label_score, li.confidence, bit.confidence, spatial,
                value_length_score=min(1.0, digit_count(val) / 12.0)
            )

            # Üstten geldiği için ceza. Böylece sadece gerçekten güçlü ise seçilir.
            score = clamp01(score - 0.08 + (0.06 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_above_one_line_tilt", matched_label, f"{bit.text} {li.text}")

    # Line bazlıda üst satıra bakmıyoruz.
    # Çünkü line bazlıda koordinat yok; eğiklik toleransı sadece item bazlı üst aramada var.
    line_hits = find_line_matches_fuzzy(lines, FATURA_LABEL_VARIANTS, min_label_score=82.0)
    for i, matched_label, line_score in line_hits:
        if line_score < 92.0 and fuzzy_has_bad_context(lines[i], FATURA_BAD_CONTEXT, threshold=88):
            continue

        val = extract_fatura_id(lines[i], sozlesme_no=sozlesme_no)
        if val:
            score = clamp01(0.78 + 0.12 * (line_score / 100.0) + (0.04 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "line_same", matched_label, lines[i])

        for j in range(i + 1, min(i + 5, len(lines))):
            if fuzzy_has_bad_context(lines[j], FATURA_BAD_CONTEXT, threshold=90):
                continue
            val = extract_fatura_id(lines[j], sozlesme_no=sozlesme_no)
            if not val:
                continue
            distance_penalty = (j - i - 1) * 0.04
            score = clamp01(0.70 + 0.12 * (line_score / 100.0) - distance_penalty + (0.04 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "line_below", matched_label, f"{lines[i]} {lines[j]}")

    return dedupe_candidates(candidates)


def extract_fatura_no(lines, items, sozlesme_no=None):
    return best_candidate(extract_fatura_candidates(lines, items, sozlesme_no=sozlesme_no), "fatura_no")


# ─────────────────────────────────────────────────────────────────────────────
# BELGE NO
# ─────────────────────────────────────────────────────────────────────────────

BELGE_FORBIDDEN_CONTEXT = [
    "MERSIS", "MERSİS", "VKN", "VERGI", "VERGİ",
    "TIC", "TİC", "SICIL", "SİCİL", "FAX", "FAKS",
    "WEB", "INTERNET", "MUSTERI NO", "MÜŞTERI NO",
    "HESAP NO", "TESISAT NO", "TESİSAT NO",
    "SOZLESME", "SÖZLEŞME", "ABONE NO",
    "SAYAC", "OKUMA", "ENDEKS", "KWH", "M3", "SM3",
]

BELGE_HINT_CONTEXT = [
    "BELGE NO", "BELGE NUMARASI",
    "FATURA NO", "FATURA NUMARASI",
    "E-ARSIV", "E-ARŞIV",
    "EVRAK NO", "SIRA NO", "SIRA NUMARASI",
    "SERI NO", "SERI NUMARASI",
]


def has_forbidden_belge_context(text: str) -> bool:
    return fuzzy_has_bad_context(text, BELGE_FORBIDDEN_CONTEXT, threshold=88)


def has_belge_hint(text: str) -> bool:
    n = normalize_text(text)
    if any(x in n for x in BELGE_HINT_CONTEXT):
        return True
    return any(fuzz.partial_ratio(n, normalize_text(x)) >= 88 for x in BELGE_HINT_CONTEXT)


def validate_belge_candidate(val: str, local_context: str = "") -> bool:
    val = normalize_id_text(val)
    if not val:
        return False
    if has_forbidden_belge_context(local_context):
        return False
    if not looks_like_belge_id(val):
        return False

    digits = numeric_body(val)
    if not digits:
        return False
    if is_phone(digits):
        return False
    if digits.isdigit() and len(digits) == 10 and not has_alpha(val):
        return False
    if len(digits) < 8:
        return False
    return True


def extract_belge_id(text: str, local_context: str = "") -> Optional[str]:
    text = normalize_id_text(text)
    cands = []

    strong_patterns = [
        r"\b[A-Z]{1,3}\d{8,20}[A-Z]?\b",
        r"\b[A-Z]\d{10,20}\b",
    ]
    numeric_patterns = [
        r"\b\d{14,20}\b",
    ]

    for pat in strong_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = normalize_id_text(m.group())
            if validate_belge_candidate(v, local_context=local_context):
                cands.append(v)

    if cands:
        return sorted(set(cands), key=lambda x: (has_alpha(x), len(x), digit_count(x)), reverse=True)[0]

    if not has_forbidden_belge_context(local_context):
        for pat in numeric_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                v = normalize_id_text(m.group())
                if validate_belge_candidate(v, local_context=local_context):
                    cands.append(v)

    if cands:
        return sorted(set(cands), key=lambda x: (len(x), digit_count(x)), reverse=True)[0]

    for fn in (extract_best_long_alpha_id, extract_best_alphanum_id):
        v = fn(text)
        if v:
            v = normalize_id_text(v)
            if validate_belge_candidate(v, local_context=local_context):
                return v

    if not has_forbidden_belge_context(local_context):
        v = extract_best_numeric_id(text, min_digits=14)
        if v:
            v = normalize_id_text(v)
            if validate_belge_candidate(v, local_context=local_context):
                return v

    return None


def extract_belge_candidates(lines, items) -> List[FieldCandidate]:
    field = "belge_no"
    candidates: List[FieldCandidate] = []

    label_hits = find_label_items_fuzzy(
        items,
        BELGE_LABELS,
        min_label_score=78.0,
        min_ocr_conf=0.22
    )

    for li, matched_label, label_score in label_hits:
        if has_forbidden_belge_context(li.text):
            continue

        same_row = [
            it for it in items
            if it is not li
            and abs(it.cy - li.cy) <= 22
            and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:4]:
            local_context = f"{li.text} {bit.text}"
            if has_forbidden_belge_context(local_context):
                continue
            val = extract_belge_id(bit.joined, local_context=local_context)
            if not val:
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=45.0, col_tol=270.0)
            score = combined_field_score(
                label_score, li.confidence, bit.confidence, spatial,
                value_length_score=min(1.0, digit_count(val) / 14.0)
            )
            score = clamp01(score + 0.08 + (0.08 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_same_row", matched_label, local_context)

        below_cands = find_column_value_below(items, li, col_tol=150.0, max_y_dist=145.0)[:5]
        for bit in below_cands:
            local_context = f"{li.text} {bit.text}"
            if has_forbidden_belge_context(local_context):
                continue
            if not within_line_gap(li, bit, lines, max_gap=4):
                continue
            val = extract_belge_id(bit.joined, local_context=local_context)
            if not val:
                continue
            spatial = spatial_proximity_score(li, bit, max_y_dist=145.0, col_tol=150.0)
            score = combined_field_score(
                label_score, li.confidence, bit.confidence, spatial,
                value_length_score=min(1.0, digit_count(val) / 14.0)
            )
            score = clamp01(score + (0.08 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "item_below", matched_label, local_context)

    line_hits = find_line_matches_fuzzy(lines, BELGE_LABELS, min_label_score=78.0)
    for i, matched_label, line_score in line_hits:
        line_ctx = lines[i]
        if has_forbidden_belge_context(line_ctx):
            continue

        val = extract_belge_id(lines[i], local_context=lines[i])
        if val:
            score = clamp01(0.64 + 0.26 * (line_score / 100.0) + (0.06 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "line_same", matched_label, lines[i])

        for j in range(i + 1, min(i + 5, len(lines))):
            local_context = f"{lines[i]} {lines[j]}"
            if has_forbidden_belge_context(local_context):
                continue
            val = extract_belge_id(lines[j], local_context=local_context)
            if val:
                score = clamp01(0.50 + 0.30 * (line_score / 100.0) - ((j - i - 1) * 0.04) + (0.06 if has_alpha(val) else 0.0))
                add_field_candidate(candidates, field, val, score, "line_below", matched_label, local_context)

    # Kontrollü fallback: ilk 18 satırda sadece belge/fatura/e-arşiv hint varsa
    for i, line in enumerate(lines[:18]):
        window = " ".join(lines[max(0, i - 2): min(len(lines), i + 3)])
        if has_forbidden_belge_context(window):
            continue
        if not has_belge_hint(window):
            continue
        val = extract_belge_id(line, local_context=window)
        if val:
            score = clamp01(0.58 + (0.08 if has_alpha(val) else 0.0))
            add_field_candidate(candidates, field, val, score, "hint_window_fallback", "BELGE_HINT", window)

    return dedupe_candidates(candidates)


def extract_belge_no(lines, items):
    return best_candidate(extract_belge_candidates(lines, items), "belge_no")


# ─────────────────────────────────────────────────────────────────────────────
# Amount / Money Extraction Helpers
# ─────────────────────────────────────────────────────────────────────────────
CURRENCY_RE = re.compile(r"(₺|\bTL\b|\bTRY\b|TL|TRY)", re.IGNORECASE)
MONEY_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[.\s]\d{3})*|\d+)(?:[.,]\d{1,2})?\s*(?:TL|TRY|₺)?",
    re.IGNORECASE
)
DATE_RE = re.compile(r"\b\d{1,2}[./\-]\d{1,2}([./\-]\d{2,4})?\b")
NOT_MONEY_RE = re.compile(
    r"(\bkwh\b|\bkw\b|\bm3\b|\bsm3\b|\bendeks\b|\bsayac\b|"
    r"\btuketim\b|\bbirim\b|\bfiyat\b|\bvergi\b|%|‰|"
    r"\bfon\b|\bkdv\b|\bmatrah\b|\bdevlet\s*destegi\b|"
    r"\bgunduz\b|\bpuant\b|\bgece\b|\bokuma\b|\bfark\b)",
    re.IGNORECASE
)


def looks_like_index_value(token: str) -> bool:
    """Sayaç/endeks değeri gibi görünen 12345.678 formatlarını para sanmayı engeller."""
    t = str(token or "").strip().replace(" ", "")
    return bool(re.fullmatch(r"\d{3,7}\.\d{3}", t))

def strip_dates_from_text(text: str) -> str:
    return re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", " ", text or "")

def parse_turkish_money(token: str) -> Optional[float]:
    original = str(token or "")
    # YENİ: TL rakama yapışıksa ayır
    original = re.sub(r"(?<=\d)(TL|TRY|₺)", r" \1", original, flags=re.IGNORECASE)
    
    t = re.sub(r"(TL|TRY|₺)", "", original, flags=re.IGNORECASE).strip()
    t = re.sub(r"[^0-9\.,]", "", t).replace(" ", "")
    
    if not t or len(t) < 2:
        return None
    if looks_like_index_value(t):
        return None
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    elif "." in t:
        parts = t.split(".")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            pass
        elif all(len(p) == 3 for p in parts[1:]):
            t = "".join(parts)
        else:
            return None
    try:
        v = float(t)
    except ValueError:
        return None
    if v < 1 or v > 100000:
        return None
    return v

def extract_tutar(lines: List[str], items: List[OCRItem]) -> Tuple[Optional[str], float]:
    """
    Güçlendirilmiş ödenecek tutar çıkarımı.

    Öncelik sırası:
    1) "ÖDENECEK + TUTAR" window içinde para değeri
    2) Fuzzy label item + aynı satır / alt satır
    3) Exact/fuzzy line label + aynı/alt satırlar
    4) Kontrollü fallback: para birimi bulunan, KDV/matrah/endeks olmayan değerler
    """
    STRONG_LABELS = [
        "ODENECEK TUTAR",
        "TOPLAM ODENECEK TUTAR",
        "TOPLAM ODENECEK",
        "ODENECEK",
    ]

    WEAK_LABELS = [
        "GENEL TOPLAM",
        "FATURA TUTARI",
        "FATURA TUTAR",
        "TOPLAM TUTAR",
        "TOPLAM BORC",
        "BORC TUTARI",
    ]

    BAD_MONEY_CONTEXT = [
        "KDV", "MATRAH", "VERGI", "FON", "BEDEL", "BIRIM FIYAT",
        "KWH", "KW", "M3", "SM3", "ENDEKS", "SAYAC", "OKUMA",
        "GUNDUZ", "PUANT", "GECE", "TUKETIM", "FARK",
        "DEVLET DESTEGI", "YUVARLAMA", "GECIKME",
    ]

    def has_bad_money_context(text: str) -> bool:
        n = normalize_text(text)
        if any(b in n for b in BAD_MONEY_CONTEXT):
            return True
        if NOT_MONEY_RE.search(text or ""):
            return True
        return False

    def extract_money_values(text: str) -> List[Tuple[str, float]]:
        # Tarihleri önce temizle — 2026 gibi yıl parçalarını para sanmayı önler
        clean = strip_dates_from_text(text)
        vals = []
        for m in MONEY_RE.finditer(clean):
            tok = m.group().strip()
            if DATE_RE.fullmatch(tok):
                continue
            if looks_like_index_value(tok):
                continue
            v = parse_turkish_money(tok)
            if v is None:
                continue
            if not (5 <= v <= 100000):
                continue
            vals.append((tok, v))
        return vals

    def label_strength(text: str) -> float:
        n = normalize_text(text)
        if "ODENECEK" in n and "TUTAR" in n:
            return 1.00
        if "TOPLAM" in n and "ODENECEK" in n:
            return 0.96
        if "ODENECEK" in n:
            return 0.90
        if "GENEL" in n and "TOPLAM" in n:
            return 0.78
        if "FATURA" in n and "TUTAR" in n:
            return 0.74
        if "TOPLAM" in n and "TUTAR" in n:
            return 0.70

        best = 0.0
        for lbl in STRONG_LABELS:
            best = max(best, label_similarity(n, lbl) / 100.0)
        if best >= 0.78:
            return max(best, 0.82)

        for lbl in WEAK_LABELS:
            best = max(best, (label_similarity(n, lbl) / 100.0) * 0.82)
        return best

    money_candidates: List[Tuple[float, float, str, str]] = []

    def add_money_candidate(value: float, score: float, source: str, context: str):
        if value is None:
            return
        if not (5 <= value <= 100000):
            return
        money_candidates.append((clamp01(score), value, source, context))

    # 1) Üst bölgede "ÖDENECEK + TUTAR" window araması
    top_n = min(len(lines), 40)
    for i in range(top_n):
        window_lines = lines[i:min(i + 5, len(lines))]
        window_text = " ".join(window_lines)
        window_norm = normalize_text(window_text)

        strong_window = (
            ("ODENECEK" in window_norm and "TUTAR" in window_norm)
            or ("TOPLAM" in window_norm and "ODENECEK" in window_norm)
        )
        if not strong_window:
            continue

        for j in range(i, min(i + 7, len(lines))):
            if has_bad_money_context(lines[j]) and "ODENECEK" not in normalize_text(lines[j]):
                continue

            vals = extract_money_values(lines[j])
            for _, v in vals:
                has_currency = bool(CURRENCY_RE.search(lines[j]))
                distance_penalty = max(0, j - i) * 0.025
                score = 0.96 - distance_penalty + (0.02 if has_currency else 0.0)
                add_money_candidate(v, score, "strong_odenecek_window", f"{window_text} || {lines[j]}")

    # 2) OCR item bazlı: label item + aynı satır / alt satır
    label_item_hits = []
    for it in items:
        if it.confidence < 0.22:
            continue
        st = label_strength(it.text)
        if st >= 0.70:
            label_item_hits.append((it, st))

    label_item_hits.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)

    for li, st in label_item_hits[:12]:
        # Aynı satır sağ taraf
        same_row = [
            it for it in items
            if it is not li
            and abs(it.cy - li.cy) <= 26
            and it.cx > li.cx
        ]
        same_row.sort(key=lambda x: x.cx)

        for bit in same_row[:5]:
            row_context = f"{li.text} {bit.text} {same_row_neighbors_text(bit, items)}"
            if has_bad_money_context(row_context) and st < 0.90:
                continue

            vals = extract_money_values(bit.text)
            for _, v in vals:
                spatial = spatial_proximity_score(li, bit, max_y_dist=60.0, col_tol=340.0)
                has_currency = bool(CURRENCY_RE.search(bit.text))
                score = 0.45 + 0.35 * st + 0.12 * bit.confidence + 0.08 * spatial
                score += 0.03 if has_currency else 0.0
                add_money_candidate(v, score, "item_same_row", row_context)

        # Alt / yakın kolon
        below = find_column_value_below(items, li, col_tol=180.0, max_y_dist=180.0)[:7]
        for bit in below:
            row_context = f"{li.text} {bit.text} {same_row_neighbors_text(bit, items)}"
            if has_bad_money_context(row_context) and st < 0.90:
                continue

            vals = extract_money_values(bit.text)
            for _, v in vals:
                spatial = spatial_proximity_score(li, bit, max_y_dist=180.0, col_tol=180.0)
                has_currency = bool(CURRENCY_RE.search(bit.text))
                score = 0.40 + 0.34 * st + 0.12 * bit.confidence + 0.08 * spatial
                score += 0.03 if has_currency else 0.0
                add_money_candidate(v, score, "item_below", row_context)

    # 3) Line bazlı label
    for i, line in enumerate(lines):
        st = label_strength(line)
        if st < 0.68:
            continue

        if has_bad_money_context(line) and st < 0.90:
            continue

        # Aynı satır
        for _, v in extract_money_values(line):
            has_currency = bool(CURRENCY_RE.search(line))
            score = 0.50 + 0.40 * st + (0.03 if has_currency else 0.0)
            add_money_candidate(v, score, "line_same", line)

        # Sonraki satırlar
        for j in range(i + 1, min(i + 5, len(lines))):
            ctx = f"{line} {lines[j]}"
            if has_bad_money_context(ctx) and st < 0.90:
                continue

            vals = extract_money_values(lines[j])
            for _, v in vals:
                has_currency = bool(CURRENCY_RE.search(lines[j]))
                distance_penalty = (j - i - 1) * 0.04
                score = 0.46 + 0.36 * st - distance_penalty + (0.03 if has_currency else 0.0)
                add_money_candidate(v, score, "line_below", ctx)

    # 4) Kontrollü fallback
    # Para birimi olan değerler, fatura üst/orta bölümünde ise ve kötü context yoksa değerlendirilir.
    total = len(lines)
    for idx, line in enumerate(lines):
        pos = idx / max(1, total - 1)
        if pos > 0.70:
            continue

        ln = normalize_text(line)
        if "TARIH" in ln or has_bad_money_context(line):
            continue

        vals = extract_money_values(line)
        if not vals:
            continue

        has_currency = bool(CURRENCY_RE.search(line))
        if not has_currency:
            # Para birimi yoksa fallbackte çok daha dikkatli davran.
            if not ("ODENECEK" in ln or "TOPLAM" in ln or "TUTAR" in ln):
                continue

        local_strength = label_strength(line)
        for _, v in vals:
            score = 0.42
            score += 0.12 if has_currency else 0.0
            score += 0.12 if pos < 0.45 else 0.04
            score += 0.18 * local_strength
            score += 0.04 if 10 <= v <= 50000 else 0.0
            add_money_candidate(v, score, "controlled_fallback", line)

    if not money_candidates:
        return None, 0.0

    # Aynı tutar birden fazla stratejide çıktıysa boost ver.
    grouped: Dict[float, Tuple[float, set, str]] = {}
    for score, value, source, context in money_candidates:
        key = round(value, 2)
        if key not in grouped:
            grouped[key] = (score, {source}, context)
        else:
            old_score, sources, old_ctx = grouped[key]
            sources.add(source)
            grouped[key] = (max(old_score, score), sources, old_ctx)

    final_pool = []
    for value, (score, sources, context) in grouped.items():
        if len(sources) >= 2:
            score = clamp01(score + 0.06)
        final_pool.append((score, value, sources, context))

    # Skor ana kriter. Skor eşitse label kaynaklı / para birimli adaylar zaten daha yüksek olur.
    final_pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_value, _, _ = final_pool[0]

    if best_score < 0.58:
        return None, 0.0

    return f"{best_value:.2f} TL", best_score


def extract_son_odeme_tarihi(lines: List[str], items: List[OCRItem] = None) -> Tuple[Optional[str], float]:
    DATE_FULL_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")

    BAD_CONTEXTS = [
        "BELGE TARIHI",
        "FATURA TARIHI",
        "FATURA DONEMI",
        "DUZENLEME SAATI",
        "SONRAKI OKUMA DONEMI",
        "OKUMA BILGILERI",
        "ILK OKUMA",
        "SON OKUMA",
        "OKUMA TARIHLERI",
        "OKUMA GUN",
        "ENDEKS",
        "GUNDUZ",
        "PUANT",
        "GECE",
    ]

    SON_ODEME_EXACT = [
        "SON ODEME TARIHI",
        "SON ODEME TARIH",
        "SON ODEME TAR",
        "SON ODEME",
        "ODEME SON TARIHI",
    ]

    def norm(s: str) -> str:
        return normalize_text(s)

    def valid_payment_date(date_str: str) -> bool:
        parts = re.split(r"[./]", date_str)
        if len(parts) != 3:
            return False
        try:
            dd, mm, yy = int(parts[0]), int(parts[1]), int(parts[2])
            if yy < 100:
                yy += 2000
            return 1 <= dd <= 31 and 1 <= mm <= 12 and 2020 <= yy <= 2100
        except:
            return False

    def line_has_bad_context(line: str) -> bool:
        n = norm(line)
        return any(b in n for b in BAD_CONTEXTS)

    def extract_date_from_text(text: str) -> Optional[str]:
        m = DATE_FULL_RE.search(text or "")
        if not m:
            return None
        d = m.group()
        return d if valid_payment_date(d) else None

    def looks_like_son_odeme_label(text: str) -> bool:
        n = norm(text)
        if any(lbl == n or lbl in n for lbl in SON_ODEME_EXACT):
            return True

        # parçalı yakalama
        has_son = "SON" in n
        has_odeme = "ODEME" in n
        has_tarih = "TARIH" in n or "TAR" in n
        if has_son and has_odeme:
            return True
        if has_odeme and has_tarih:
            return True
        return False

    # 1) EN GÜÇLÜ: üst blokta "SON ODEME" etrafında window araması
    top_n = min(len(lines), 35)
    for i in range(top_n):
        cur = norm(lines[i])

        if line_has_bad_context(lines[i]):
            continue

        window = " ".join(norm(lines[j]) for j in range(i, min(i + 4, top_n)))

        # örn: ODENECEK / Son Odeme / TUTAR / 12.12.2025
        if ("SON ODEME" in window) or ("ODEME SON TARIHI" in window):
            for j in range(i, min(i + 6, len(lines))):
                if line_has_bad_context(lines[j]):
                    continue
                d = extract_date_from_text(lines[j])
                if d:
                    return d, 0.96

        # parçalı label: SON + ODEME ayrı satırlarda
        if "SON" in window and "ODEME" in window:
            for j in range(i, min(i + 6, len(lines))):
                if line_has_bad_context(lines[j]):
                    continue
                d = extract_date_from_text(lines[j])
                if d:
                    return d, 0.93

    # 2) Exact/fuzzy line matching
    line_hits = find_line_matches_fuzzy(lines[:35], SON_ODEME_LABELS, min_label_score=72.0)
    for i, _, line_score in line_hits:
        if line_has_bad_context(lines[i]):
            continue

        d = extract_date_from_text(lines[i])
        if d:
            return d, clamp01(0.65 + 0.25 * (line_score / 100.0))

        for j in range(i + 1, min(i + 6, len(lines))):
            if line_has_bad_context(lines[j]):
                continue
            d = extract_date_from_text(lines[j])
            if d:
                return d, clamp01(0.54 + 0.28 * (line_score / 100.0))

    # 3) Heuristic line scan: label parçalanmış olabilir
    for i in range(top_n):
        if line_has_bad_context(lines[i]):
            continue

        local = " ".join(norm(lines[j]) for j in range(i, min(i + 3, top_n)))
        if not looks_like_son_odeme_label(local):
            continue

        for j in range(i, min(i + 6, len(lines))):
            if line_has_bad_context(lines[j]):
                continue
            d = extract_date_from_text(lines[j])
            if d:
                return d, 0.88

    # 4) OCR item bazlı arama
    if items:
        label_candidates = []
        for it in items:
            if it.confidence < 0.20:
                continue
            if looks_like_son_odeme_label(it.text) and not line_has_bad_context(it.text):
                label_candidates.append(it)

        label_candidates.sort(key=lambda x: (x.y1, -x.confidence))

        for li in label_candidates:
            # aynı kolon / yakın alt bölge
            cands = find_column_value_below(items, li, col_tol=180.0, max_y_dist=170.0, skip_rows=0)[:6]
            best_date, best_score = None, 0.0

            for bit in cands:
                if line_has_bad_context(bit.text):
                    continue

                d = extract_date_from_text(bit.text)
                if not d:
                    continue

                spatial = spatial_proximity_score(li, bit, max_y_dist=170.0, col_tol=180.0)
                score = combined_field_score(
                    100.0,
                    li.confidence,
                    bit.confidence,
                    spatial,
                    value_length_score=1.0
                )
                if score > best_score:
                    best_date, best_score = d, score

            if best_date:
                return best_date, best_score

    # 5) Son fallback: üst bölgede ODENECEK + date ilişkisi
    for i in range(top_n):
        block = " ".join(norm(lines[j]) for j in range(i, min(i + 5, top_n)))
        if "ODENECEK" in block and "TUTAR" in block:
            for j in range(i, min(i + 7, len(lines))):
                if line_has_bad_context(lines[j]):
                    continue
                d = extract_date_from_text(lines[j])
                if d:
                    return d, 0.72

    return None, 0.0
