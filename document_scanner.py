import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = ["ScanResult", "DocumentScanner", "draw_debug_visualization"]


@dataclass
class ScanResult:
    success: bool
    image: Optional[np.ndarray] = None           # Warped renkli (BGR)
    enhanced_image: Optional[np.ndarray] = None  # Enhanced gri — arayüz + OCR varsayılanı
    variants: Dict[str, np.ndarray] = field(default_factory=dict)
    corners: Optional[List[Tuple[int, int]]] = None
    confidence: float = 0.0
    message: str = ""


class DocumentScanner:
    def __init__(
        self,
        min_area_ratio: float = 0.08,
        max_area_ratio: float = 0.995,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 8.0,
        detection_width: int = 1280,
        ocr_min_width: int = 1600,
        long_doc_min_height: int = 2600,
    ):
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.detection_width = int(detection_width)
        self.ocr_min_width = int(ocr_min_width)
        self.long_doc_min_height = int(long_doc_min_height)

    # ------------------------------------------------------------------
    # EXIF ROTATION
    # ------------------------------------------------------------------
    @staticmethod
    def fix_exif_rotation(image_bytes: bytes) -> np.ndarray:
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            pil = ImageOps.exif_transpose(pil)
            pil = pil.convert("RGB")
            arr = np.array(pil)
            pil.close()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ==================================================================
    # SENIN KODUNDAN: preprocess_image
    # ==================================================================
    @staticmethod
    def _preprocess_image(img: np.ndarray):
        """
        process_receipt_advanced -> preprocess_image
        Donerur: gray, bilateral, enhanced, blur
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        return gray, bilateral, enhanced, blur

    # ==================================================================
    # SENIN KODUNDAN: is_valid_quad
    # ==================================================================
    @staticmethod
    def _is_valid_quad(quad: np.ndarray, img_shape) -> bool:
        h, w = img_shape[:2]
        for x, y in quad:
            if x < -10 or x > w + 10 or y < -10 or y > h + 10:
                return False
        quad_area = cv2.contourArea(quad.astype(int))
        if quad_area < 0.05 * h * w:
            return False
        return True

    # ==================================================================
    # SENIN KODUNDAN: order_points (centroid tabanli)
    # ==================================================================
    @staticmethod
    def _order_points_receipt(pts: np.ndarray) -> np.ndarray:
        """
        process_receipt_advanced -> order_points
        Centroid tabanli: TL, TR, BR, BL
        """
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]
        s = pts_sorted.sum(axis=1)
        tl_idx = np.argmin(s)
        pts_sorted = np.roll(pts_sorted, -tl_idx, axis=0)
        return pts_sorted

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        return self._order_points_receipt(np.array(pts, dtype=np.float32))

    # ==================================================================
    # SENIN KODUNDAN: four_point_transform (perspektif orani duzeltmeli)
    # ==================================================================
    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray,
                             target_aspect: float = 1.414) -> Optional[np.ndarray]:
        """
        process_receipt_advanced -> four_point_transform
        Derin perspektifte ust/alt genislik farki %40'i asarsa ortalama kullanir.
        """
        rect = DocumentScanner._order_points_receipt(pts.astype(np.float32))
        tl, tr, br, bl = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 0 or maxHeight <= 0:
            return None

        avg_width = (widthA + widthB) / 2
        avg_height = (heightA + heightB) / 2
        aspect = avg_height / avg_width if avg_width > 0 else target_aspect

        width_ratio = (
            min(widthA, widthB) / max(widthA, widthB)
            if max(widthA, widthB) > 0 else 1
        )
        if width_ratio < 0.6:
            maxWidth = int(avg_width)
            maxHeight = int(avg_width * aspect)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    # ==================================================================
    # SENIN KODUNDAN: fallback kosesi tespiti (5 adim)
    # ==================================================================
    def _detect_corners_fallback(
        self, image: np.ndarray,
        gray: np.ndarray, bilateral: np.ndarray,
        enhanced: np.ndarray, blur: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        process_receipt_advanced -> detect_receipt_corners_fallback
        5 adimli fallback; manuel secim (Fallback 5) web ortaminda atlanir.
        """
        h, w = image.shape[:2]

        # FALLBACK 1: Adaptive Threshold
        adapt_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        adapt_close = cv2.morphologyEx(adapt_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours_a1, _ = cv2.findContours(adapt_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours_a1, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(cnt)
            if area < 1500 or area > 0.99 * h * w:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        # FALLBACK 2: Canny (dusuk threshold)
        edges_low = cv2.Canny(blur, 15, 50)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_low = cv2.dilate(edges_low, kernel2, iterations=3)

        contours_a2, _ = cv2.findContours(edges_low, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours_a2, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        # FALLBACK 3: Bilateral + Threshold
        bilateral_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 19, 8,
        )
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bilateral_close = cv2.morphologyEx(
            bilateral_thresh, cv2.MORPH_CLOSE, kernel3, iterations=3
        )

        contours_a3, _ = cv2.findContours(bilateral_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours_a3, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(cnt)
            if area < 2000 or area > 0.98 * h * w:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity > 0.60:
                x, y, cw, ch = cv2.boundingRect(cnt)
                ar = ch / cw if cw > 0 else 0
                if 0.15 < ar < 5.5:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
                    if len(approx) == 4:
                        return approx.reshape(4, 2).astype(np.float32)

        # FALLBACK 4: Hough Lines + K-Means
        edges_hough = cv2.Canny(blur, 20, 80)
        lines = cv2.HoughLinesP(
            edges_hough, 1, np.pi / 180, 40,
            minLineLength=min(w, h) // 5, maxLineGap=30,
        )
        if lines is not None and len(lines) >= 4:
            intersections = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    p1, p2 = lines[i][0][:2], lines[i][0][2:]
                    p3, p4 = lines[j][0][:2], lines[j][0][2:]
                    denom = (
                        (p1[0] - p2[0]) * (p3[1] - p4[1])
                        - (p1[1] - p2[1]) * (p3[0] - p4[0])
                    )
                    if abs(denom) > 1e-10:
                        t = (
                            (p1[0] - p3[0]) * (p3[1] - p4[1])
                            - (p1[1] - p3[1]) * (p3[0] - p4[0])
                        ) / denom
                        xi = p1[0] + t * (p2[0] - p1[0])
                        yi = p1[1] + t * (p2[1] - p1[1])
                        if 0 <= xi < w and 0 <= yi < h:
                            intersections.append([xi, yi])

            if len(intersections) >= 4:
                pts_arr = np.array(intersections, dtype=np.float32)
                _, _, centers = cv2.kmeans(
                    pts_arr, 4, None,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                    10, cv2.KMEANS_RANDOM_CENTERS,
                )
                return centers.astype(np.float32)

        return None

    # ==================================================================
    # SENIN KODUNDAN: primary kose tespiti (Otsu)
    # ==================================================================
    def _detect_corners_primary(
        self, img: np.ndarray, blur: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        process_receipt_advanced -> primary Otsu thresholding blogu
        """
        h, w = img.shape[:2]

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000 or area > 0.98 * h * w:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity > 0.65:
                x, y, cw, ch = cv2.boundingRect(cnt)
                ar = ch / cw if cw > 0 else 0
                if 0.15 < ar < 5.5:
                    peri = cv2.arcLength(cnt, True)
                    epsilon = 0.025 * peri if ar > 2 else 0.035 * peri
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) == 4:
                        quad = approx.reshape(4, 2).astype(np.float32)
                        if self._is_valid_quad(quad, img.shape):
                            return quad

        return None

    # ==================================================================
    # SENIN KODUNDAN: rotate_to_portrait
    # ==================================================================
    @staticmethod
    def _rotate_to_portrait(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    # ==================================================================
    # SENIN KODUNDAN: build_variants — ADIM 3-4-5-6 birebir
    # ==================================================================
    def build_variants(
        self, warped: np.ndarray, original: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        process_receipt_advanced adimlari:
          "warped"               -> ADIM 3: perspektif duzeltilmis renkli
          "grayscale"            -> ADIM 4: gri tonlama
          "threshold"            -> ADIM 5: adaptif eslikleme (blockSize=11, C=3)
          "enhanced"             -> ADIM 6: CLAHE (clipLimit=2.5)  <- OCR varsayilani
          "original_normalized"  -> orijinal, EXIF+upscale
        """
        variants: Dict[str, np.ndarray] = {}

        warped_up = self._upscale_for_ocr(warped.copy())

        # ADIM 3: Warped renkli
        variants["warped"] = warped_up

        # ADIM 4: Gri versiyonu
        warped_gray = cv2.cvtColor(warped_up, cv2.COLOR_BGR2GRAY)
        variants["grayscale"] = warped_gray

        # ADIM 5: Adaptif threshold
        warped_thresh = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 3,
        )
        variants["threshold"] = warped_thresh

        # ADIM 6: Kontrast artirma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(warped_gray)
        variants["enhanced"] = enhanced_gray

        # Orijinal normalize
        orig_up = self._upscale_for_ocr(original.copy())
        variants["original_normalized"] = orig_up

        return variants

    # ------------------------------------------------------------------
    # UPSCALE (OCR icin)
    # ------------------------------------------------------------------
    def _upscale_for_ocr(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h > 2.2 * w:
            scale = max(
                self.long_doc_min_height / float(h),
                self.ocr_min_width / float(w),
                1.0,
            )
        else:
            scale = max(self.ocr_min_width / float(w), 1.0)

        if scale > 1.01:
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        return image

    # ------------------------------------------------------------------
    # enhance_for_ocr — orijinal, degistirilmedi
    # ------------------------------------------------------------------
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(th > 0))
        if len(coords) < 100:
            return image
        angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) < 0.3 or abs(angle) > 12:
            return image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        image = self._upscale_for_ocr(image)
        image = self._deskew(image)
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
        sharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    # ==================================================================
    # ANA TARAMA: process_receipt_advanced akisi
    # preprocess -> primary -> fallback -> four_point_transform
    # -> rotate_to_portrait -> build_variants
    # ==================================================================
    def scan_document(self, image: np.ndarray, enhance: bool = True) -> ScanResult:
        if image is None or image.size == 0:
            return ScanResult(False, message="Invalid image")

        original = image.copy()
        h, w = image.shape[:2]

        # 1. On isleme
        gray, bilateral, enhanced_prep, blur = self._preprocess_image(image)

        # 2. Primary kose tespiti (Otsu)
        quad = self._detect_corners_primary(image, blur)

        # 3. Fallback
        if quad is None:
            quad = self._detect_corners_fallback(
                image, gray, bilateral, enhanced_prep, blur
            )

        # 4. Kose bulunamadi
        if quad is None:
            fallback = self._rotate_to_portrait(image.copy())
            variants = self.build_variants(fallback, original) if enhance else {}
            return ScanResult(
                success=False,
                image=fallback,
                enhanced_image=variants.get("enhanced"),
                variants=variants,
                corners=None,
                confidence=0.0,
                message="Fatura koseleri tespit edilemedi; normalize goruntu donduruldu.",
            )

        # 5. Perspektif duzeltmesi
        warped = self.four_point_transform(image, quad, target_aspect=1.414)
        if warped is None or warped.size == 0:
            return ScanResult(False, message="Perspektif donusumu basarisiz.")

        # 6. Portrait'e dondur
        warped = self._rotate_to_portrait(warped)

        # 7. Varyantlar uret (ADIM 3->4->5->6)
        variants = self.build_variants(warped, original) if enhance else {}

        # 8. Corners listesi
        quad_ordered = self._order_points_receipt(quad)
        corners_list = [(int(x), int(y)) for x, y in quad_ordered]

        # Confidence: 3 faktörün ağırlıklı ortalaması
        # 1. Alan oranı — fatura görüntünün %10'unu kaplıyorsa zaten iyi
        quad_area = float(cv2.contourArea(quad_ordered.astype(np.int32)))
        area_score = float(np.clip(quad_area / (h * w) / 0.10, 0.0, 1.0))

        # 2. Dikdörtgensellik — köşe açıları 90°'ye yakınlığı
        def _ang(a, b, c):
            ba, bc = a - b, c - b
            d = np.linalg.norm(ba) * np.linalg.norm(bc)
            return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / d, -1, 1)))) if d > 1e-6 else 90.0

        q = quad_ordered.astype(np.float32)
        devs = [abs(_ang(q[3], q[0], q[1]) - 90),
                abs(_ang(q[0], q[1], q[2]) - 90),
                abs(_ang(q[1], q[2], q[3]) - 90),
                abs(_ang(q[2], q[3], q[0]) - 90)]
        rect_score = float(np.clip(1.0 - np.mean(devs) / 45.0, 0.0, 1.0))

        # 3. En-boy oranı — fatura genellikle uzun (ar > 1.5)
        warp_h, warp_w = warped.shape[:2]
        ar = max(warp_h, warp_w) / max(min(warp_h, warp_w), 1)
        ar_score = float(np.clip((ar - 1.0) / 3.0, 0.0, 1.0))

        confidence = float(np.clip(
            (area_score * 0.50 + rect_score * 0.35 + ar_score * 0.15) * 100.0,
            0.0, 100.0,
        ))

        return ScanResult(
            success=True,
            image=warped,
            enhanced_image=variants.get("enhanced"),
            variants=variants,
            corners=corners_list,
            confidence=round(confidence, 2),
            message=f"Belge tespit edildi ({confidence:.1f}%)",
        )

    def scan_from_bytes(self, image_bytes: bytes, enhance: bool = True) -> ScanResult:
        try:
            image = self.fix_exif_rotation(image_bytes)
            if image is None or image.size == 0:
                return ScanResult(False, message="Failed to decode image")
            return self.scan_document(image, enhance=enhance)
        except Exception as e:
            return ScanResult(False, message=f"Error: {e}")

    def variant_to_bytes(
        self,
        result: ScanResult,
        variant_name: str = "enhanced",
        fmt: str = "JPEG",
        quality: int = 95,
    ) -> Optional[bytes]:
        img = result.variants.get(variant_name)
        if img is None or img.size == 0:
            return None
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format=fmt, quality=quality)
        return buf.getvalue()

    def result_to_bytes(
        self,
        result: ScanResult,
        format: str = "JPEG",
        quality: int = 95,
        use_enhanced: bool = True,
    ) -> Optional[bytes]:
        variant = "enhanced" if use_enhanced else "warped"
        return self.variant_to_bytes(result, variant_name=variant, fmt=format, quality=quality)

    def build_ocr_variants(self, image: np.ndarray) -> List[np.ndarray]:
        base = self.enhance_for_ocr(image)
        variants = [base]
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10,
        )
        variants.append(binary)
        enlarged = cv2.resize(base, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        variants.append(enlarged)
        return variants

    def crop_roi_retry(
        self,
        image: np.ndarray,
        y1: int,
        y2: int,
        pad_ratio: float = 0.25,
        scale: float = 3.0,
    ) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        if y2 <= y1:
            return None
        pad = int((y2 - y1) * pad_ratio) + 8
        yy1, yy2 = max(0, y1 - pad), min(h, y2 + pad)
        crop = image[yy1:yy2, 0:w]
        if crop.size == 0:
            return None
        up = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = (
            cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up.copy()
        )
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.8)
        gray = cv2.addWeighted(gray, 1.9, blurred, -0.9, 0)
        return np.clip(gray, 0, 255).astype(np.uint8)


def draw_debug_visualization(
    image: np.ndarray,
    corners: List[Tuple[int, int]],
    confidence: float,
) -> np.ndarray:
    vis = image.copy()
    if corners:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (x, y), 8, (255, 0, 0), -1)
            cv2.putText(
                vis, str(i + 1), (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
    cv2.putText(
        vis, f"Confidence: {confidence:.1f}%",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
    )
    return vis