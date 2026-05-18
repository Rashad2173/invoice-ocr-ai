import json
import uuid
import asyncio
from pathlib import Path
import io
import time
import threading
import base64
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import gc
import uvicorn
import httpx

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import cv2
import numpy as np

from database import SessionLocal, engine, Base
from models import Document
from document_scanner import DocumentScanner, ScanResult
from invoice_analyzer import InvoiceAnalyzer


BASE_DIR = Path(__file__).resolve().parent

TMP_DIR = Path(r"C:\paddle_tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(r"C:\paddle_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



# def build_vlm_prompt(ocr_hint_data: Optional[dict] = None) -> str:
#     ocr_hint_text = ""
#     if ocr_hint_data:
#         ocr_hint_text = (
#             "\nOCR İPUCU (yanlış olabilir, görsel daha önceliklidir):\n"
#             f"{json.dumps(ocr_hint_data, ensure_ascii=False)}\n"
#         )

#     return f"""Sen Türk faturalarından yapılandırılmış veri çıkarmaya uzmanlaşmış bir Vision Language Model'sin.
# Görevin, verilen fatura görselini analiz edip ilgili bilgileri aşağıdaki JSON formatında çıkarmaktır.
# Fatura elektrik, doğalgaz veya su faturası olabilir.
# Sadece görselde açıkça görülen bilgileri doldur, tahmin etme veya uydurma.
# Sadece JSON çıktısı ver, açıklama veya yorum ekleme.

# SCHEMA:
# {{
#     "firma_ismi": "faturayı düzenleyen kurum adı (aşağıdaki listeden seç)",
#     "fatura_turu": "elektrik, dogalgaz veya su",
#     "abone_bilgileri": {{
#         "sozlesme_no": "sözleşme veya hesap numarası",
#         "tesisat_no": "tesisat veya tüketim noktası numarası",
#         "musteri_no": "müşteri numarası",
#         "fatura_no": "fatura numarası (yalnızca tam olarak Fatura No etiketindeki değer)",
#         "belge_no": "belge numarası (yalnızca tam olarak Belge No etiketindeki değer)"
#     }},
#     "odeme": {{
#         "tutar": "sayısal değer",
#         "son_odeme_tarihi": "son ödeme tarihi"
#     }}
# }}

# KURUM LİSTESİ (firma_ismi bu listeden seçilmeli):
# {kurum_listesi}

# KURALLAR:
# - musteri_no için yalnızca tam olarak "Müşteri No" etiketinin altındaki/yanındaki değeri al
# - belge_no için yalnızca tam olarak "Belge No" etiketi görünüyorsa o değeri al, yoksa null
# - fatura_no için yalnızca tam olarak "Fatura No" etiketinin yanındaki değeri al
# - "Fatura Sıra No" değerini fatura_no alanına yazma
# - Fatura Sıra No, Sayaç Seri No gibi alanları hiçbir alana yazma
# - 444, 0212, 0216, 0850 ile başlayan numaralar telefondur, hiçbir alana yazma
# - sozlesme_no en az 8 haneli olmalı
# - tutar için sadece "ÖDENECEK TUTAR" alanındaki rakamı al
# - tutar alanına sadece sayısal değer yaz, virgül veya TL yazma
# - Türkçe format dönüşümü: 380,00 → 380.0 | 1.250,75 → 1250.75
# - 380,00 için 3800 veya 38000 yazma, doğrusu 380.0
# - Sadece mevcut olan alanları doldur, bilgi yoksa null yaz
# - Sadece JSON döndür, markdown code block kullanma
# {ocr_hint_text}
# """

# async def _run_vlm(image_bytes: bytes, ocr_hint_data: Optional[dict] = None):
#     vlm_url = os.getenv("VLM_ENDPOINT_URL")
#     if not vlm_url:
#         raise RuntimeError("VLM_ENDPOINT_URL tanımlı değil")

#     prompt = build_vlm_prompt(ocr_hint_data)

#     model_name = os.getenv("VLM_MODEL_NAME", "")
#     api_key = os.getenv("VLM_API_KEY", "")

#     files = {
#         "file": ("invoice.jpg", image_bytes, "image/jpeg")
#     }

#     data = {
#         "prompt": prompt,
#     }

#     if model_name:
#         data["model"] = model_name

#     headers = {}
#     if api_key:
#         headers["Authorization"] = f"Bearer {api_key}"

#     async with httpx.AsyncClient(timeout=180.0) as client:
#         response = await client.post(
#             vlm_url,
#             data=data,
#             files=files,
#             headers=headers,
#         )

#     if response.status_code != 200:
#         raise RuntimeError(f"VLM endpoint error {response.status_code}: {response.text}")

#     result = response.json()
#     return result


# @app.post("/vlm")
# async def vlm_extract(
#     file: UploadFile = File(...),
#     ocr_json: str = Form(None),
#     document_id: Optional[int] = Form(None),
# ):
#     data = await file.read()
#     if not data:
#         return JSONResponse(status_code=400, content={"error": "Boş dosya (0 byte)."})

#     ocr_hint_data = None
#     if ocr_json:
#         try:
#             ocr_hint_data = json.loads(ocr_json)
#         except Exception:
#             pass

#     try:
#         vlm_result = await _run_vlm(data, ocr_hint_data)

#         if document_id is not None:
#             db = SessionLocal()
#             try:
#                 db_doc = db.query(Document).filter(Document.id == document_id).first()
#                 if db_doc:
#                     db_doc.vlm_firma_adi   = vlm_result.get("firma_ismi") or vlm_result.get("firma_adi")
#                     db_doc.vlm_sozlesme_no = (
#                         (vlm_result.get("abone_bilgileri") or {}).get("sozlesme_no")
#                         or vlm_result.get("sozlesme_no")
#                     )
#                     db_doc.vlm_tutar = (
#                         (vlm_result.get("odeme") or {}).get("tutar")
#                         or vlm_result.get("tutar")
#                     )
#                     db_doc.vlm_fatura_turu = vlm_result.get("fatura_turu")
#                     db_doc.vlm_raw_json = vlm_result
#                     db.commit()
#             finally:
#                 db.close()

#             vlm_result["document_id"] = document_id

#         return vlm_result

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"VLM error: {e}"})


# ------------------------------------------------------------------
# Geçerli varyant isimleri
# ------------------------------------------------------------------
VALID_VARIANTS = {
    "warped",
    "grayscale",
    "threshold",
    "enhanced",
    "original_normalized",
}

# ------------------------------------------------------------------
# TEK NOKTADAN KONTROL
# Sadece bunları değiştirmen yeterli
# ------------------------------------------------------------------
DEFAULT_OCR_VARIANT = "grayscale"
DEFAULT_UI_VARIANT = "grayscale"

scanner = DocumentScanner(
    min_area_ratio=0.08,
    max_area_ratio=0.995,
    min_aspect_ratio=1.0,
    max_aspect_ratio=6.0,
    detection_width=1280,
    ocr_min_width=1600,
    long_doc_min_height=2600,
)

analyzer: Optional[InvoiceAnalyzer] = None
_analyzer_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Invoice OCR + Scanner Demo")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "web"), html=True), name="web")
from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "web" / "index.html"))


# ------------------------------------------------------------------
# OCR MODEL
# ------------------------------------------------------------------
def get_analyzer() -> InvoiceAnalyzer:
    global analyzer
    if analyzer is None:
        with _analyzer_lock:
            if analyzer is None:
                print("OCR modeli yükleniyor...")
                analyzer = InvoiceAnalyzer(
                    firmalar_path="firmalar.json",
                    model_dir=r"C:\paddle_models"
                )
                print("OCR modeli yüklendi ✅")
    return analyzer


def warmup_analyzer() -> None:
    try:
        local_analyzer = get_analyzer()
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "warmup.jpg"
        cv2.imwrite(str(tmp), dummy)
        try:
            local_analyzer.analyze(str(tmp))
            print("OCR modeli hazır ✅")
        finally:
            if tmp.exists():
                tmp.unlink()
    except Exception as e:
        print(f"OCR warmup uyarı: {e}")


@app.on_event("startup")
async def startup_event():
    get_analyzer()
    _executor.submit(warmup_analyzer)


# ------------------------------------------------------------------
# OCR RUNNER
# ------------------------------------------------------------------
def _run_ocr(image_path, original_data):
    local_analyzer = get_analyzer()
    norm_path = None
    try:
        result = local_analyzer.analyze(str(image_path), return_enhanced=True)
        return result
    except Exception:
        img = None
        try:
            img = Image.open(io.BytesIO(original_data))
            img = ImageOps.exif_transpose(img).convert("RGB")
            norm_path = TMP_DIR / f"{uuid.uuid4().hex}.jpg"
            img.save(norm_path, format="JPEG", quality=98, subsampling=0)
            img.close()
            img = None
            return local_analyzer.analyze(str(norm_path), return_enhanced=True)
        finally:
            if img:
                img.close()
            if norm_path and norm_path.exists():
                try:
                    norm_path.unlink()
                except Exception:
                    pass
    finally:
        gc.collect()
        try:
            if image_path and Path(image_path).exists():
                Path(image_path).unlink()
        except Exception:
            pass


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def calculate_tilt_angle_from_corners(corners) -> Optional[float]:
    try:
        if not corners or len(corners) != 4:
            return None
        tl, tr, br, bl = corners
        dx = float(tr[0] - tl[0])
        dy = float(tr[1] - tl[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return round(math.degrees(math.atan2(dy, dx)), 2)
    except Exception:
        return None


def build_capture_warning(scan_success, scan_confidence, tilt_angle):
    tilt_warning = None
    capture_warning = None

    if tilt_angle is not None:
        a = abs(float(tilt_angle))
        if a < 10:
            tilt_warning = None
        elif a < 18:
            tilt_warning = "Belgeyi biraz daha düz tutun."
        else:
            tilt_warning = "Belgeyi tam karşıdan çekin."

    if scan_success is False:
        capture_warning = (
            "Belge kenarları net algılanamadı. "
            "Lütfen belgeyi kadraja alıp tekrar deneyin."
        )
    elif scan_confidence is not None and scan_confidence < 35:
        capture_warning = (
            f"Belge algılama güveni düşük ({scan_confidence:.1f}%). "
            "Belgeyi biraz daha yaklaştırın veya ışığı iyileştirin."
        )
    elif scan_confidence is not None and scan_confidence < 50:
        capture_warning = "Belge algılandı, biraz daha sabit ve karşıdan tutun."
    elif tilt_warning:
        capture_warning = tilt_warning

    return tilt_warning, capture_warning


def variant_to_b64(result: ScanResult, variant_name: str) -> Optional[str]:
    img_bytes = scanner.variant_to_bytes(result, variant_name=variant_name)
    if img_bytes is None:
        return None
    return base64.b64encode(img_bytes).decode("utf-8")


def resolve_variant(requested: Optional[str], default_variant: str) -> str:
    v = (requested or default_variant or "").strip().lower()
    if v not in VALID_VARIANTS:
        return default_variant
    return v


# ------------------------------------------------------------------
# /scan — sadece tarama, OCR yok
# SADECE TEK UI VARYANTI DÖNER
# ------------------------------------------------------------------
@app.post("/scan")
async def scan_document_endpoint(
    file: UploadFile = File(...),
    enhance: bool = Form(True),
    ui_variant: str = Form(DEFAULT_UI_VARIANT),
):
    try:
        image_bytes = await file.read()
        result = scanner.scan_from_bytes(image_bytes, enhance=enhance)

        tilt_angle = calculate_tilt_angle_from_corners(result.corners)
        tilt_warning, capture_warning = build_capture_warning(
            scan_success=result.success,
            scan_confidence=round(result.confidence, 2),
            tilt_angle=tilt_angle,
        )

        selected_ui_variant = resolve_variant(ui_variant, DEFAULT_UI_VARIANT)
        scanned_image_b64 = None

        if enhance:
            scanned_image_b64 = variant_to_b64(result, selected_ui_variant)

        return JSONResponse(content={
            "success": result.success,
            "confidence": round(result.confidence, 2),
            "corners": result.corners,
            "message": result.message,
            "tilt_angle": tilt_angle,
            "tilt_warning": tilt_warning,
            "capture_warning": capture_warning,
            "ui_variant": selected_ui_variant,
            "scanned_image": scanned_image_b64,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Scan error: {str(e)}"})


# ------------------------------------------------------------------
# /analyze — tarama + OCR
# SADECE TEK OCR VARYANTI OCR'A GİDER
# SADECE TEK UI VARYANTI ARAYÜZE GİDER
# ------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    auto_scan: bool = Form(True),
    ocr_variant: str = Form(DEFAULT_OCR_VARIANT),
    ui_variant: str = Form(DEFAULT_UI_VARIANT),
):
    ext = (Path(file.filename).suffix or "").lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Desteklenmeyen dosya tipi. jpg/jpeg/png/webp yükleyin."},
        )

    selected_ocr_variant = resolve_variant(ocr_variant, DEFAULT_OCR_VARIANT)
    selected_ui_variant = resolve_variant(ui_variant, DEFAULT_UI_VARIANT)

    raw_path = TMP_DIR / f"{uuid.uuid4().hex}{ext}"

    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Boş dosya (0 byte)."})

        original_data = data

        # OCR'a gidecek bytes
        ocr_input_bytes = original_data
        ocr_source = "original"

        # Arayüze gidecek tek varyant
        preview_image_b64 = None
        preview_source = None

        scan_meta = None
        tilt_angle = None
        tilt_warning = None
        capture_warning = None

        if auto_scan:
            scan_result = scanner.scan_from_bytes(original_data, enhance=True)

            tilt_angle = calculate_tilt_angle_from_corners(scan_result.corners)
            tilt_warning, capture_warning = build_capture_warning(
                scan_success=scan_result.success,
                scan_confidence=round(scan_result.confidence, 2),
                tilt_angle=tilt_angle,
            )

            scan_meta = {
                "scan_success": scan_result.success,
                "scan_confidence": round(scan_result.confidence, 2),
                "scan_message": scan_result.message,
                "scan_corners": scan_result.corners,
            }

            # OCR için sadece seçilen varyant
            selected_bytes = scanner.variant_to_bytes(scan_result, variant_name=selected_ocr_variant)
            if selected_bytes is not None:
                ocr_input_bytes = selected_bytes
                ocr_source = selected_ocr_variant
            else:
                fallback_bytes = scanner.variant_to_bytes(scan_result, variant_name=DEFAULT_OCR_VARIANT)
                if fallback_bytes is not None:
                    ocr_input_bytes = fallback_bytes
                    ocr_source = f"{DEFAULT_OCR_VARIANT}_fallback"
                else:
                    ocr_input_bytes = original_data
                    ocr_source = "original_fallback"

            # UI için sadece seçilen varyant
            preview_image_b64 = variant_to_b64(scan_result, selected_ui_variant)
            if preview_image_b64 is not None:
                preview_source = selected_ui_variant
            else:
                preview_image_b64 = variant_to_b64(scan_result, DEFAULT_UI_VARIANT)
                preview_source = DEFAULT_UI_VARIANT if preview_image_b64 else None

        # OCR input'u diske yaz
        raw_path.write_bytes(ocr_input_bytes)

        # DB için orijinal dosyayı kaydet
        upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        upload_path.write_bytes(original_data)

        db = SessionLocal()
        try:
            document = Document(
                file_name=file.filename,
                file_path=str(upload_path),
                image_data=original_data,
                client_ip=request.client.host,
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            doc_id = document.id
            print(f"DOCUMENT SAVED: id={doc_id} file={file.filename} ocr_source={ocr_source}")
        finally:
            db.close()

        loop = asyncio.get_event_loop()
        ocr_started_at = datetime.utcnow()
        t0 = time.perf_counter()

        ocr_result = await loop.run_in_executor(
            _executor,
            _run_ocr,
            raw_path,
            ocr_input_bytes,
        )

        t1 = time.perf_counter()
        ocr_finished_at = datetime.utcnow()
        ocr_duration_seconds = round(t1 - t0, 3)

        if isinstance(ocr_result, Exception):
            print(f"OCR ERROR: {ocr_result}")
            ocr_result = {}

        db = SessionLocal()
        try:
            db_doc = db.query(Document).filter(Document.id == doc_id).first()
            if db_doc:
                ocr_result_to_save = dict(ocr_result)
                ocr_result_to_save.pop("enhanced_image_b64", None)
                ocr_result_to_save.pop("preview_image_b64", None)

                db_doc.ocr_firma_adi = (
                    ocr_result.get("firma_ismi") or ocr_result.get("firma_adi")
                )
                db_doc.ocr_sozlesme_no = (
                    (ocr_result.get("abone_bilgileri") or {}).get("sozlesme_no")
                    or ocr_result.get("sozlesme_no")
                    or (ocr_result.get("abone_bilgileri") or {}).get("musteri_no")
                    or ocr_result.get("musteri_no")
                )
                db_doc.ocr_tutar = (
                    (ocr_result.get("odeme") or {}).get("tutar")
                    or ocr_result.get("tutar")
                )
                db_doc.ocr_fatura_turu = ocr_result.get("fatura_turu")
                db_doc.ocr_raw_json = ocr_result_to_save
                db_doc.ocr_started_at = ocr_started_at
                db_doc.ocr_finished_at = ocr_finished_at
                db_doc.ocr_duration_seconds = ocr_duration_seconds
                db.commit()
                print(f"DB UPDATED: id={doc_id}")
        finally:
            db.close()

        response = {
            **ocr_result,
            "document_id": doc_id,
            "ocr_duration_seconds": ocr_duration_seconds,
            "ocr_source": ocr_source,
            "ui_source": preview_source,
            "scan_meta": scan_meta,
            "tilt_angle": tilt_angle,
            "tilt_warning": tilt_warning,
            "capture_warning": capture_warning,
            "preview_image_b64": preview_image_b64,
            # geriye uyumluluk için:
            "enhanced_image_b64": preview_image_b64,
        }
        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"API error: {e}"})
    finally:
        try:
            if raw_path.exists():
                raw_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
