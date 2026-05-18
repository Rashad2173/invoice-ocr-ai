import base64
import gc
import json
import logging
from typing import Any, Dict, List, Optional

import cv2

from field_extractors import (
    detect_firma,
    extract_belge_candidates,
    extract_fatura_candidates,
    extract_musteri_candidates,
    extract_son_odeme_tarihi,
    extract_sozlesme_candidates,
    extract_tesisat_candidates,
    extract_tutar,
    resolve_field_candidates,
)
from ocr_engine import OCREngine, OCRItem


class InvoiceAnalyzer:
    def __init__(self, firmalar_path: str = "firmalar.json", model_dir: str = r"C:\paddle_models"):
        self.ocr = OCREngine(model_dir=model_dir)
        self.firmalar: List[Dict] = []

        try:
            with open(firmalar_path, encoding="utf-8") as f:
                self.firmalar = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Firmalar dosyası bulunamadı: {firmalar_path}")
        except json.JSONDecodeError as e:
            logging.error(f"Firmalar JSON parse hatası: {e}")

    def analyze(
        self,
        image_path: str,
        print_ocr_text: bool = False,
        include_scores: bool = False,
        return_enhanced: bool = False,
    ) -> Dict[str, Any]:
        image = None
        full_text = ""
        items: List[OCRItem] = []
        lines: List[str] = []
        enhanced_b64: Optional[str] = None

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Görsel okunamadı: {image_path}")

            if return_enhanced:
                try:
                    from document_scanner import DocumentScanner
                    scanner = DocumentScanner()
                    enh = scanner.enhance_for_ocr(image)
                    if len(enh.shape) == 2:
                        enh = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
                    ok, buf = cv2.imencode(".jpg", enh, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        enhanced_b64 = base64.b64encode(buf).decode("utf-8")
                    del enh, buf
                except ImportError:
                    enhanced_b64 = None
                except Exception as e:
                    logging.debug(f"Enhanced image oluşturulamadı: {e}")
                    enhanced_b64 = None

            full_text, items = self.ocr.run(image)
            if items is None:
                items = []

            if print_ocr_text:
                print("=" * 70 + "\nOCR METNİ:\n" + "=" * 70)
                print(full_text)
                print("=" * 70)

            lines = [l.strip() for l in full_text.split("\n") if l.strip()]

            firma_ismi, fatura_turu = detect_firma(full_text, self.firmalar)

            # Ana mimari:
            # 1) Her alan kendi aday listesini üretir.
            # 2) Resolver aynı / iç içe geçen değerleri tek alana verir.
            # 3) Değeri kaybeden alan otomatik olarak kendi 2. / 3. adayına düşer.
            sozlesme_candidates = extract_sozlesme_candidates(lines, items)
            tesisat_candidates = extract_tesisat_candidates(lines, items)
            musteri_candidates = extract_musteri_candidates(lines, items)
            fatura_candidates = extract_fatura_candidates(lines, items, sozlesme_no=None)
            belge_candidates = extract_belge_candidates(lines, items)

            candidate_map = {
                "sozlesme_no": sozlesme_candidates,
                "tesisat_no": tesisat_candidates,
                "musteri_no": musteri_candidates,
                "fatura_no": fatura_candidates,
                "belge_no": belge_candidates,
            }

            (
                sozlesme_no, sozlesme_sc,
                tesisat_no, tesisat_sc,
                musteri_no, musteri_sc,
                fatura_no, fatura_sc,
                belge_no, belge_sc,
            ) = resolve_field_candidates(candidate_map)

            tutar, tutar_sc = extract_tutar(lines, items)
            son_odeme, odeme_sc = extract_son_odeme_tarihi(lines, items)

            result: Dict[str, Any] = {
                "firma_ismi": firma_ismi,
                "fatura_turu": fatura_turu,
                "abone_bilgileri": {
                    "sozlesme_no": sozlesme_no,
                    "tesisat_no": tesisat_no,
                    "musteri_no": musteri_no,
                    "fatura_no": fatura_no,
                    "belge_no": belge_no,
                },
                "odeme": {
                    "tutar": tutar,
                    "son_odeme_tarihi": son_odeme,
                },
                "ocr_raw_text": full_text,
            }

            if enhanced_b64:
                result["enhanced_image_b64"] = enhanced_b64

            if include_scores:
                result["_scores"] = {
                    "sozlesme_no": round(sozlesme_sc, 3),
                    "tesisat_no": round(tesisat_sc, 3),
                    "musteri_no": round(musteri_sc, 3),
                    "fatura_no": round(fatura_sc, 3),
                    "belge_no": round(belge_sc, 3),
                    "tutar": round(tutar_sc, 3),
                    "son_odeme": round(odeme_sc, 3),
                }
                result["_candidate_counts"] = {
                    "sozlesme_no": len(sozlesme_candidates),
                    "tesisat_no": len(tesisat_candidates),
                    "musteri_no": len(musteri_candidates),
                    "fatura_no": len(fatura_candidates),
                    "belge_no": len(belge_candidates),
                }

            return result

        except FileNotFoundError:
            raise
        except Exception as e:
            logging.error(f"Analiz hatası: {e}")
            raise
        finally:
            try:
                del image
            except Exception:
                pass
            try:
                del items
            except Exception:
                pass
            try:
                del lines
            except Exception:
                pass
            gc.collect()
