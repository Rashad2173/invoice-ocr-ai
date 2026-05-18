#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

from field_extractors import *
from invoice_analyzer import InvoiceAnalyzer
from normalization import *
from ocr_engine import OCRItem, OCREngine


def main():
    parser = argparse.ArgumentParser(description="Türk Fatura Analiz Sistemi — PaddleOCR")
    parser.add_argument("--image", required=True, help="Fatura görsel dosyası")
    parser.add_argument("--firmalar", default="firmalar.json", help="Firma listesi JSON")
    parser.add_argument("--model-dir", default=r"C:\paddle_models", help="PaddleOCR model dizini")
    parser.add_argument("--print-ocr", action="store_true", help="OCR metnini yazdır")
    parser.add_argument("--scores", action="store_true", help="Güven skorlarını dahil et")
    parser.add_argument("--return-enhanced", action="store_true", help="Debug için enhanced_image_b64 ekle")
    parser.add_argument("--output", help="Sonuç JSON dosyası")
    args = parser.parse_args()

    try:
        analyzer = InvoiceAnalyzer(firmalar_path=args.firmalar, model_dir=args.model_dir)
        result = analyzer.analyze(
            image_path=args.image,
            print_ocr_text=args.print_ocr,
            include_scores=args.scores,
            return_enhanced=args.return_enhanced,
        )

        out = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Kaydedildi: {args.output}")
        else:
            print(out)

    except FileNotFoundError as e:
        print(f"Hata: {e}", file=__import__("sys").stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"Beklenmeyen hata: {e}", file=__import__("sys").stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
