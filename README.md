# Turkish Invoice OCR (invoice-ocr-ai)

A FastAPI service that turns a photographed or scanned Turkish utility invoice into clean, structured data — from raw image to a validated JSON record in PostgreSQL, covering **52 utility providers nationwide**.

## Overview

Turkish utility invoices (electricity, natural gas, water) come from dozens of regional providers with no shared layout. This project handles the full path from an unconstrained phone photo to structured data: it locates and straightens the document, runs OCR, resolves the extracted text into the correct fields even when candidates overlap or conflict, and persists everything for later use or auditing.

## Pipeline

**1. Document scanning & correction** (`document_scanner.py`)
- Multi-stage corner detection: a primary Otsu-threshold contour pass, backed by four progressively more aggressive fallback strategies (adaptive threshold, Canny edges, bilateral+threshold with solidity/aspect filtering, and Hough-line intersections clustered with k-means) — so the pipeline still finds the invoice edges on cluttered backgrounds or uneven lighting.
- EXIF-safe rotation, an aspect-ratio-aware four-point perspective transform, automatic portrait orientation, and deskewing.
- A weighted confidence score (contour area 50% / rectangularity 35% / aspect ratio 15%) drives real-time capture feedback — tilt and framing warnings sent straight back to the client.
- Produces 5 image variants per scan (warped, grayscale, adaptive-threshold, CLAHE-enhanced, normalized-original) so OCR and the UI preview can each use whichever works best.

**2. OCR** (`ocr_engine.py`)
- PaddleOCR (English detection/recognition models + angle classifier) with per-line confidence filtering, plus a dedicated text-normalization module for artifacts common in Turkish OCR output (e.g. digits split across tokens).

**3. Field extraction & resolution** (`field_extractors.py`, `invoice_analyzer.py`)
- Provider identification against a 52-company reference table spanning electricity, natural gas, and water/sewerage utilities across Turkey (`firmalar.json`).
- Each target field (`sozlesme_no`, `tesisat_no`, `musteri_no`, `fatura_no`, `belge_no`, `tutar`, `son_odeme_tarihi`) independently generates a list of OCR-derived candidates.
- A resolver reconciles candidates *across* fields: if two fields would claim the same value, it goes to the stronger match and the losing field automatically falls back to its next-best candidate.
- Optional per-field confidence scores and candidate counts for debugging.

**4. Persistence** (`database.py`, `models.py`)
- PostgreSQL via SQLAlchemy — stores the original image, the full OCR result (JSONB), and timing metrics for every upload.
- The schema already provisions parallel `vlm_*` columns for a planned GPT-4o vision cross-validation pass (implemented, currently disabled in `api.py`).

## Extracted fields

```
firma_ismi                         # provider name
fatura_turu                        # elektrik / dogalgaz / su
abone_bilgileri.sozlesme_no        # contract number
abone_bilgileri.tesisat_no         # installation number
abone_bilgileri.musteri_no         # customer number
abone_bilgileri.fatura_no          # invoice number
abone_bilgileri.belge_no           # document number
odeme.tutar                        # amount due
odeme.son_odeme_tarihi             # due date
```

## API

- `POST /scan` — scan + perspective correction only, no OCR. Returns confidence, detected corners, tilt/capture warnings, and the selected preview variant.
- `POST /analyze` — full pipeline: scan → OCR → field extraction → DB write. Returns the structured fields, timing, and a preview image.
- A static web UI is served at `/` and mounted under `/ui`.

## Tech stack

`Python` · `FastAPI` · `PaddleOCR` · `OpenCV` · `scikit-image` · `PostgreSQL` (SQLAlchemy + JSONB) · `Docker`

## Coverage

52 providers nationwide — 19 electricity distributors (BEDAŞ, AYEDAŞ, ENERJİSA, BAŞKENT EDAŞ, ...), 14 natural gas distributors (İGDAŞ, Bursagaz, Başkentgaz, ...), 19 water/sewerage administrations (İSKİ, ASKİ, İZSU, ...).

## Setup

```bash
git clone https://github.com/Rashad2173/invoice-ocr-ai.git
cd invoice-ocr-ai
pip install -r requirements.txt
```

Set `DATABASE_URL` for your Postgres instance (falls back to a local default if unset), and point the PaddleOCR model directory in `api.py` to where your model files live.

```bash
uvicorn api:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

> **Note:** a few paths (`C:\paddle_tmp`, `C:\paddle_uploads`, `C:\paddle_models`) are currently hardcoded for local Windows development — worth externalizing via environment variables before deploying elsewhere.

## License

MIT (adjust if different)
