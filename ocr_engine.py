import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from normalization import join_split_digits, normalize_text


import os
import io
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr, contextmanager

warnings.filterwarnings("ignore")
ENV_CONFIG = {
    "FLAGS_use_mkldnn": "0",
    "FLAGS_use_mkldnn_int8": "0",
    "FLAGS_enable_mkldnn_bfloat16": "0",
    "FLAGS_enable_onednn": "0",
    "OMP_NUM_THREADS": "1",
    "KMP_WARNINGS": "0",
    "FLAGS_log_level": "3",
    "GLOG_v": "0",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}
for _k, _v in ENV_CONFIG.items():
    os.environ[_k] = _v

logging.basicConfig(level=logging.ERROR)
for _n in ["ppocr", "paddle", "ppocr.utils", "ppocr.postprocess"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import paddle
    from paddleocr import PaddleOCR

try:
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except:
    pass


@contextmanager
def silent():
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


@dataclass
class OCRItem:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float

    @property
    def norm(self) -> str:
        return normalize_text(self.text)

    @property
    def joined(self) -> str:
        return join_split_digits(self.text)


class OCREngine:
    def __init__(self, model_dir: str = r"C:\paddle_models"):
        self._engine = None
        self.model_dir = Path(model_dir)

    @property
    def engine(self) -> PaddleOCR:
        if self._engine is None:
            with silent():
                self._engine = PaddleOCR(
                    lang="en",
                    use_angle_cls=True,
                    det_model_dir=str(self.model_dir / "det" / "en" / "en_PP-OCRv3_det_infer"),
                    rec_model_dir=str(self.model_dir / "rec" / "en" / "en_PP-OCRv4_rec_infer"),
                    cls_model_dir=str(self.model_dir / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"),
                    enable_mkldnn=False
                )
        return self._engine

    def run(self, image: np.ndarray, thr: float = 0.25) -> Tuple[str, List[OCRItem]]:
        """PaddleOCR çalıştırır; full_text ve koordinatlı OCRItem listesi döndürür."""
        rgb = None
        raw = None
        items: List[OCRItem] = []
        texts: List[str] = []

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with silent():
                raw = self.engine.ocr(rgb, cls=False)

            if raw and raw[0]:
                for line in raw[0]:
                    box = line[0]
                    text = line[1][0]
                    conf = float(line[1][1] or 0)

                    if conf < thr:
                        continue

                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    items.append(OCRItem(
                        text=text,
                        confidence=conf,
                        x1=min(xs),
                        y1=min(ys),
                        x2=max(xs),
                        y2=max(ys),
                        cx=(min(xs) + max(xs)) / 2,
                        cy=(min(ys) + max(ys)) / 2,
                    ))
                    texts.append(text)

            full = re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", "\n".join(texts))).strip()
            return full, items

        finally:
            try:
                del rgb
            except Exception:
                pass
            try:
                del raw
            except Exception:
                pass
            gc.collect()
