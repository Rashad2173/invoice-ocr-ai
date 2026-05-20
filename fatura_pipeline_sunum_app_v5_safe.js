// Fatura OCR Sunum Akışı v4
// Bu dosya sadece sunum/demo içindir. Backend'e istek atmaz.
// Önemli: Bu sürümde belge bulma tarafına senin app.js içindeki jscanify tabanlı akış direkt eklendi.

const DEMO_DELAY_MS = 500;
const MIN_OCR_WIDTH = 1800;
const MAX_INPUT_SIDE = 2400;

const STEP_ORDER = [
  'original',
  'detect',
  'orientation',
  'upscale',
  'gray',
  'bilateral',
  'clahe',
  'gaussian',
  'sharpen'
];

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const runBtn = document.getElementById('runBtn');
const resetBtn = document.getElementById('resetBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const progressBar = document.getElementById('progressBar');
const speakerText = document.getElementById('speakerText');
const variantJson = document.getElementById('variantJson');
const scannerStatus = document.getElementById('scannerStatus');
const imageModal = document.getElementById('imageModal');
const imageModalImg = document.getElementById('imageModalImg');
const imageModalTitle = document.getElementById('imageModalTitle');
const imageModalClose = document.getElementById('imageModalClose');

let currentFile = null;
let originalCanvas = null;
let isRunning = false;
let scanner = null;
let jscanifyReady = false;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(text, mode = '') {
  statusText.textContent = text;
  statusDot.className = 'dot' + (mode ? ' ' + mode : '');
}

function setScannerStatus(text, mode = '') {
  if (!scannerStatus) return;
  scannerStatus.textContent = text;
  scannerStatus.className = 'scanner-status' + (mode ? ' ' + mode : '');
}

function setProgress(index) {
  const pct = Math.round((index / Math.max(1, STEP_ORDER.length - 1)) * 100);
  progressBar.style.width = pct + '%';
}

function setActiveStep(stepKey) {
  document.querySelectorAll('.step-card').forEach((card) => {
    const isCurrent = card.dataset.step === stepKey;
    card.classList.toggle('active', isCurrent);
  });
}

function setDoneStep(stepKey) {
  const card = document.querySelector(`.step-card[data-step="${stepKey}"]`);
  if (card) {
    card.classList.remove('active');
    card.classList.add('done');
  }
}

function clearStepState() {
  document.querySelectorAll('.step-card').forEach((card) => {
    card.classList.remove('active', 'done');
  });
}

function clearPreview(stepKey, text) {
  const preview = document.getElementById('preview-' + stepKey);
  const meta = document.getElementById('meta-' + stepKey);
  if (preview) {
    preview.classList.remove('is-portrait', 'is-long-document');
    preview.innerHTML = `<div class="placeholder">${text || 'Bekleniyor'}</div>`;
  }
  if (meta) meta.innerHTML = '';
}

function clearAllPreviews() {
  clearPreview('original', 'Henüz görsel yüklenmedi');
  clearPreview('detect', 'Belge tespiti bekleniyor');
  clearPreview('orientation', 'Yön düzeltme bekleniyor');
  clearPreview('upscale', 'Upscale bekleniyor');
  clearPreview('gray', 'Grayscale bekleniyor');
  clearPreview('bilateral', 'Bilateral bekleniyor');
  clearPreview('clahe', 'CLAHE bekleniyor');
  clearPreview('gaussian', 'Gaussian bekleniyor');
  clearPreview('sharpen', 'Final çıktı bekleniyor');
}

function canvasToDataUrl(canvas) {
  // Sunumda ara işlem görsellerini bozmamak için PNG kullanıyoruz.
  return canvas.toDataURL('image/png');
}

function canvasSizeTag(canvas) {
  return `${canvas.width}×${canvas.height}px`;
}

function setPreview(stepKey, canvas, tags = []) {
  const preview = document.getElementById('preview-' + stepKey);
  const meta = document.getElementById('meta-' + stepKey);
  if (!preview || !canvas) return;

  // Görsel oranına göre kutuyu büyütüyoruz. Böylece dikey fatura kart içinde kesilmez.
  const ratio = canvas.height / Math.max(1, canvas.width);
  preview.classList.remove('is-portrait', 'is-long-document');
  if (ratio > 2.35) preview.classList.add('is-long-document');
  else if (ratio > 1.15) preview.classList.add('is-portrait');

  const src = canvasToDataUrl(canvas);
  const title = document.querySelector(`.step-card[data-step="${stepKey}"] .step-title`)?.textContent || stepKey;
  preview.innerHTML = `<img src="${src}" alt="${stepKey}" data-title="${title}">`;

  const allTags = [canvasSizeTag(canvas), ...tags];
  if (meta) {
    meta.innerHTML = allTags
      .map((tag) => {
        const cls = typeof tag === 'object' ? tag.cls || '' : '';
        const txt = typeof tag === 'object' ? tag.text : tag;
        return `<span class="tag ${cls}">${txt}</span>`;
      })
      .join('') + '<span class="zoom-hint">tıkla büyüt</span>';
  }
}

function setSpeaker(text) {
  // Bu alan HTML'de silinmişse akışın takılmaması için kontrol ediyoruz.
  if (speakerText) speakerText.textContent = text;
}

function setVariantInfo(payload) {
  // Bu alan HTML'de silinmişse akışın takılmaması için kontrol ediyoruz.
  if (variantJson) variantJson.textContent = JSON.stringify(payload, null, 2);
}

function cloneCanvas(sourceCanvas) {
  const canvas = document.createElement('canvas');
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0);
  return canvas;
}

function resizeForDemo(sourceCanvas, maxSide = MAX_INPUT_SIDE) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  const longSide = Math.max(w, h);

  if (longSide <= maxSide) return sourceCanvas;

  const scale = maxSide / longSide;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function loadImageFileAsCanvas(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve(resizeForDemo(canvas));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Görsel okunamadı'));
    };
    img.src = url;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenCV + jscanify hazırlık
// ─────────────────────────────────────────────────────────────────────────────
function waitForOpenCv(timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();

    function poll() {
      try {
        if (window.cv && typeof cv.imread === 'function' && typeof cv.Mat === 'function') {
          resolve();
          return;
        }
      } catch (_) {}

      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error('OpenCV yüklenemedi veya hazır değil'));
        return;
      }
      setTimeout(poll, 120);
    }

    poll();
  });
}

async function waitForOpenCvAndScanner(timeoutMs = 15000) {
  await waitForOpenCv(timeoutMs);

  if (scanner && jscanifyReady) return scanner;

  try {
    if (window.jscanify) {
      scanner = new window.jscanify();
      jscanifyReady = true;
      setScannerStatus('jscanify aktif ✓', 'ok');
      return scanner;
    }
  } catch (e) {
    console.warn('jscanify başlatılamadı:', e);
  }

  jscanifyReady = false;
  setScannerStatus('jscanify yok — OpenCV fallback kullanılacak', 'warn');
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Senin app.js içindeki belge tespiti mantığına uygun yardımcılar
// ─────────────────────────────────────────────────────────────────────────────
function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function polygonArea(pts) {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(area / 2);
}

// Orijinal koddaki açı/centroid yaklaşımına göre sıralama
function orderPoints(pts) {
  const points = pts.map(p => ({ x: p.x, y: p.y }));
  const center = {
    x: points.reduce((a, p) => a + p.x, 0) / points.length,
    y: points.reduce((a, p) => a + p.y, 0) / points.length
  };

  points.sort((a, b) => {
    const aa = Math.atan2(a.y - center.y, a.x - center.x);
    const bb = Math.atan2(b.y - center.y, b.x - center.x);
    return aa - bb;
  });

  let tlIndex = 0;
  let minSum = Infinity;
  for (let i = 0; i < points.length; i++) {
    const s = points[i].x + points[i].y;
    if (s < minSum) {
      minSum = s;
      tlIndex = i;
    }
  }

  return [
    points[tlIndex],
    points[(tlIndex + 1) % 4],
    points[(tlIndex + 2) % 4],
    points[(tlIndex + 3) % 4],
  ];
}

function expandQuadHorizontally(pts, canvasWidth, expandRatio = 0.12) {
  const [tl, tr, br, bl] = pts.map(p => ({ ...p }));

  const topDx = (tr.x - tl.x) * expandRatio;
  const bottomDx = (br.x - bl.x) * expandRatio;

  tl.x = Math.max(0, tl.x - topDx);
  tr.x = Math.min(canvasWidth, tr.x + topDx);
  bl.x = Math.max(0, bl.x - bottomDx);
  br.x = Math.min(canvasWidth, br.x + bottomDx);

  return [tl, tr, br, bl];
}

// Bu fonksiyon bilerek senin app.js'teki eşiklere yakın tutuldu.
function isValidDocumentQuad(corners, canvasWidth, canvasHeight) {
  if (!corners || corners.length !== 4) return false;

  const pts = orderPoints(corners);

  for (const p of pts) {
    if (p.x < 0 || p.x > canvasWidth || p.y < 0 || p.y > canvasHeight) return false;
  }

  const area = polygonArea(pts);
  const imgArea = canvasWidth * canvasHeight;
  const areaRatio = area / imgArea;

  if (areaRatio < 0.20 || areaRatio > 0.95) return false;

  const topW = distance(pts[0], pts[1]);
  const bottomW = distance(pts[3], pts[2]);
  const leftH = distance(pts[0], pts[3]);
  const rightH = distance(pts[1], pts[2]);

  const avgW = (topW + bottomW) / 2;
  const avgH = (leftH + rightH) / 2;

  if (avgW < 120 || avgH < 120) return false;

  const longSide = Math.max(avgW, avgH);
  const shortSide = Math.max(1, Math.min(avgW, avgH));
  const aspect = longSide / shortSide;
  if (aspect < 1.2 || aspect > 4.2) return false;

  const widthBalance = Math.min(topW, bottomW) / Math.max(topW, bottomW);
  const heightBalance = Math.min(leftH, rightH) / Math.max(leftH, rightH);

  const widthRatio = Math.max(topW, bottomW) / Math.max(1, Math.min(topW, bottomW));
  const heightRatio = Math.max(leftH, rightH) / Math.max(1, Math.min(leftH, rightH));

  if (widthBalance < 0.50 || heightBalance < 0.50) return false;
  if (widthRatio > 1.75) return false;
  if (heightRatio > 1.55) return false;

  const angles = [];
  for (let i = 0; i < 4; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % 4];
    const p3 = pts[(i + 2) % 4];

    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };

    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.hypot(v1.x, v1.y);
    const mag2 = Math.hypot(v2.x, v2.y);

    if (mag1 < 1 || mag2 < 1) return false;

    const cos = dot / (mag1 * mag2);
    const angle = Math.acos(Math.max(-1, Math.min(1, cos))) * (180 / Math.PI);
    angles.push(angle);
  }

  const validAngles = angles.every(a => a > 60 && a < 120);
  if (!validAngles) return false;

  const minEdge = Math.min(
    distance(pts[0], pts[1]),
    distance(pts[1], pts[2]),
    distance(pts[2], pts[3]),
    distance(pts[3], pts[0])
  );
  if (minEdge < 40) return false;

  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;

  const dx = Math.abs(cx - canvasWidth / 2) / canvasWidth;
  const dy = Math.abs(cy - canvasHeight / 2) / canvasHeight;

  if (dx > 0.28 || dy > 0.28) return false;

  return true;
}

// Sunumda daha az boş kalması için fallback detector daha esnek çalışır.
function isValidDocumentQuadRelaxed(corners, canvasWidth, canvasHeight) {
  if (!corners || corners.length !== 4) return false;
  const pts = orderPoints(corners);
  const areaRatio = polygonArea(pts) / (canvasWidth * canvasHeight);
  if (areaRatio < 0.08 || areaRatio > 0.98) return false;

  const topW = distance(pts[0], pts[1]);
  const bottomW = distance(pts[3], pts[2]);
  const leftH = distance(pts[0], pts[3]);
  const rightH = distance(pts[1], pts[2]);
  const avgW = (topW + bottomW) / 2;
  const avgH = (leftH + rightH) / 2;
  const aspect = Math.max(avgW, avgH) / Math.max(1, Math.min(avgW, avgH));
  const minEdge = Math.min(topW, bottomW, leftH, rightH);

  if (avgW < 80 || avgH < 80) return false;
  if (minEdge < 30) return false;
  if (aspect < 1.05 || aspect > 6.0) return false;
  return true;
}

function drawDetectedQuad(sourceCanvas, pts, color = '#22d3ee') {
  const canvas = cloneCanvas(sourceCanvas);
  const ctx = canvas.getContext('2d');
  const ordered = orderPoints(pts);

  ctx.save();
  ctx.lineWidth = Math.max(4, Math.round(Math.max(canvas.width, canvas.height) * 0.006));
  ctx.strokeStyle = color;
  ctx.fillStyle = color === '#34d399' ? 'rgba(52, 211, 153, .14)' : 'rgba(34, 211, 238, .12)';
  ctx.beginPath();
  ctx.moveTo(ordered[0].x, ordered[0].y);
  for (let i = 1; i < ordered.length; i++) ctx.lineTo(ordered[i].x, ordered[i].y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ordered.forEach((p, idx) => {
    ctx.beginPath();
    ctx.fillStyle = '#34d399';
    ctx.arc(p.x, p.y, Math.max(8, canvas.width * 0.006), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#020617';
    ctx.font = `${Math.max(14, canvas.width * 0.014)}px monospace`;
    ctx.fillText(String(idx + 1), p.x + 9, p.y - 9);
  });
  ctx.restore();

  return canvas;
}

function perspectiveCrop(sourceCanvas, pts) {
  const [tl, tr, br, bl] = orderPoints(pts);
  const widthA = distance(br, bl);
  const widthB = distance(tr, tl);
  const heightA = distance(tr, br);
  const heightB = distance(tl, bl);

  const maxWidth = Math.max(1, Math.round(Math.max(widthA, widthB)));
  const maxHeight = Math.max(1, Math.round(Math.max(heightA, heightB)));

  const src = cv.imread(sourceCanvas);
  const dst = new cv.Mat();
  const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
    tl.x, tl.y,
    tr.x, tr.y,
    br.x, br.y,
    bl.x, bl.y
  ]);
  const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    maxWidth - 1, 0,
    maxWidth - 1, maxHeight - 1,
    0, maxHeight - 1
  ]);
  const M = cv.getPerspectiveTransform(srcTri, dstTri);
  cv.warpPerspective(src, dst, M, new cv.Size(maxWidth, maxHeight), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

  const out = document.createElement('canvas');
  out.width = maxWidth;
  out.height = maxHeight;
  cv.imshow(out, dst);

  src.delete();
  dst.delete();
  srcTri.delete();
  dstTri.delete();
  M.delete();

  return out;
}

function tryHighlightPaper(sourceCanvas) {
  if (!scanner) return null;
  try {
    const highlighted = scanner.highlightPaper(sourceCanvas, {
      color: 'rgb(0,255,120)',
      thickness: 4
    });
    return highlighted || null;
  } catch (e) {
    console.warn('highlightPaper hata:', e);
    return null;
  }
}

// Senin app.js içindeki tryExtractPaper akışının sunuma uyarlanmış hali.
function tryExtractPaperJscanify(sourceCanvas) {
  if (!scanner) return null;

  try {
    const mat = cv.imread(sourceCanvas);
    const contour = scanner.findPaperContour(mat);
    mat.delete();

    if (!contour) return null;

    const corners = scanner.getCornerPoints(contour);
    if (!corners) return null;

    const rawCorners = [
      corners.topLeftCorner,
      corners.topRightCorner,
      corners.bottomRightCorner,
      corners.bottomLeftCorner
    ].filter(Boolean);

    if (rawCorners.length !== 4) return null;
    if (!isValidDocumentQuad(rawCorners, sourceCanvas.width, sourceCanvas.height)) return null;

    let ordered = orderPoints(rawCorners);
    ordered = expandQuadHorizontally(ordered, sourceCanvas.width, 0.14);

    const [tl, tr, br, bl] = ordered;
    const realW = Math.round(Math.max(distance(tl, tr), distance(bl, br)));
    const realH = Math.round(Math.max(distance(tl, bl), distance(tr, br)));

    const widthCoverage = realW / sourceCanvas.width;
    const heightCoverage = realH / sourceCanvas.height;
    const longCoverage = Math.max(widthCoverage, heightCoverage);
    const shortCoverage = Math.min(widthCoverage, heightCoverage);

    if (longCoverage < 0.55) return null;
    if (shortCoverage < 0.20) return null;

    const maxSide = 1600;
    const scale = Math.min(maxSide / Math.max(realW, realH), 1.0);
    const outW = Math.max(1, Math.round(realW * scale));
    const outH = Math.max(1, Math.round(realH * scale));

    const cropCanvas = scanner.extractPaper(sourceCanvas, outW, outH, {
      topLeftCorner: tl,
      topRightCorner: tr,
      bottomRightCorner: br,
      bottomLeftCorner: bl
    });

    if (!cropCanvas) return null;

    const overlayCanvas = tryHighlightPaper(sourceCanvas) || drawDetectedQuad(sourceCanvas, ordered, '#34d399');
    return {
      found: true,
      method: 'jscanify / senin tryExtractPaper',
      points: ordered,
      overlayCanvas,
      cropCanvas,
      areaRatio: polygonArea(ordered) / (sourceCanvas.width * sourceCanvas.height)
    };
  } catch (e) {
    console.warn('tryExtractPaperJscanify hata:', e);
    return null;
  }
}

function extractPointsFromApprox(approx) {
  const pts = [];
  for (let j = 0; j < 4; j++) {
    pts.push({
      x: approx.data32S[j * 2],
      y: approx.data32S[j * 2 + 1]
    });
  }
  return pts;
}

function findBestQuadInBinary(binaryMat, sourceCanvas) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  let best = null;
  let bestArea = 0;

  try {
    cv.findContours(binaryMat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const canvasArea = sourceCanvas.width * sourceCanvas.height;

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = Math.abs(cv.contourArea(contour));
      const peri = cv.arcLength(contour, true);

      for (const epsilon of [0.018, 0.025, 0.035, 0.05]) {
        const approx = new cv.Mat();
        cv.approxPolyDP(contour, approx, epsilon * peri, true);

        if (approx.rows === 4 && area > bestArea) {
          const pts = extractPointsFromApprox(approx);
          if (isValidDocumentQuadRelaxed(pts, sourceCanvas.width, sourceCanvas.height) && area / canvasArea > 0.08) {
            bestArea = area;
            best = pts;
          }
        }
        approx.delete();
      }

      contour.delete();
    }

    return best;
  } finally {
    hierarchy.delete();
    contours.delete();
  }
}

// jscanify bulamazsa devreye giren OpenCV fallback.
function findDocumentAndCropOpenCvFallback(sourceCanvas) {
  let src = null;
  let gray = null;
  let blur = null;
  let edges = null;
  let dilated = null;
  let adaptive = null;
  let kernel = null;
  let best = null;

  try {
    src = cv.imread(sourceCanvas);
    gray = new cv.Mat();
    blur = new cv.Mat();
    edges = new cv.Mat();
    dilated = new cv.Mat();
    adaptive = new cv.Mat();
    kernel = cv.Mat.ones(5, 5, cv.CV_8U);

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);

    // Pass 1: Canny + dilate
    cv.Canny(blur, edges, 40, 140);
    cv.dilate(edges, dilated, kernel);
    best = findBestQuadInBinary(dilated, sourceCanvas);

    // Pass 2: Adaptive threshold
    if (!best) {
      cv.adaptiveThreshold(gray, adaptive, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 9);
      cv.morphologyEx(adaptive, adaptive, cv.MORPH_CLOSE, kernel);
      best = findBestQuadInBinary(adaptive, sourceCanvas);
    }

    if (!best) {
      return {
        found: false,
        method: 'fallback başarısız',
        overlayCanvas: sourceCanvas,
        cropCanvas: sourceCanvas,
        areaRatio: null
      };
    }

    return {
      found: true,
      method: 'OpenCV fallback contour',
      points: orderPoints(best),
      overlayCanvas: drawDetectedQuad(sourceCanvas, best, '#22d3ee'),
      cropCanvas: perspectiveCrop(sourceCanvas, best),
      areaRatio: polygonArea(best) / (sourceCanvas.width * sourceCanvas.height)
    };
  } finally {
    [src, gray, blur, edges, dilated, adaptive, kernel].forEach((m) => {
      try { if (m) m.delete(); } catch (_) {}
    });
  }
}

function findDocumentAndCrop(sourceCanvas) {
  const fromJscanify = tryExtractPaperJscanify(sourceCanvas);
  if (fromJscanify && fromJscanify.found) return fromJscanify;

  const fromFallback = findDocumentAndCropOpenCvFallback(sourceCanvas);
  if (fromFallback && fromFallback.found) return fromFallback;

  return {
    found: false,
    method: scanner ? 'jscanify + fallback denendi' : 'OpenCV fallback denendi',
    overlayCanvas: sourceCanvas,
    cropCanvas: sourceCanvas,
    areaRatio: null
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Orientation helpers
// ─────────────────────────────────────────────────────────────────────────────
function rotateCanvas90Clockwise(sourceCanvas) {
  const out = document.createElement('canvas');
  out.width = sourceCanvas.height;
  out.height = sourceCanvas.width;
  const ctx = out.getContext('2d');
  ctx.translate(out.width / 2, out.height / 2);
  ctx.rotate(Math.PI / 2);
  ctx.drawImage(sourceCanvas, -sourceCanvas.width / 2, -sourceCanvas.height / 2);
  return out;
}

function rotateCanvas90CounterClockwise(sourceCanvas) {
  const out = document.createElement('canvas');
  out.width = sourceCanvas.height;
  out.height = sourceCanvas.width;
  const ctx = out.getContext('2d');
  ctx.translate(out.width / 2, out.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.drawImage(sourceCanvas, -sourceCanvas.width / 2, -sourceCanvas.height / 2);
  return out;
}

function getRegionDarkRatio(ctx, x, y, w, h, threshold = 175) {
  if (w <= 0 || h <= 0) return 0;
  const data = ctx.getImageData(x, y, w, h).data;
  let dark = 0;
  let total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    if (gray < threshold) dark++;
    total++;
  }

  return total ? dark / total : 0;
}

function getTopTextDensityScore(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  if (!w || !h) return 0;

  const topH = Math.max(1, Math.floor(h * 0.24));
  const upperMidY = Math.floor(h * 0.24);
  const upperMidH = Math.max(1, Math.floor(h * 0.18));
  const lowerMidY = Math.floor(h * 0.58);
  const lowerMidH = Math.max(1, Math.floor(h * 0.18));

  const topDark = getRegionDarkRatio(ctx, 0, 0, w, topH, 175);
  const upperMidDark = getRegionDarkRatio(ctx, 0, upperMidY, w, upperMidH, 175);
  const lowerMidDark = getRegionDarkRatio(ctx, 0, lowerMidY, w, lowerMidH, 175);

  return (topDark * 2.3) + (upperMidDark * 0.9) - (lowerMidDark * 1.0);
}

function normalizeDocumentOrientationSmart(sourceCanvas) {
  if (sourceCanvas.height >= sourceCanvas.width) {
    return { canvas: sourceCanvas, rotated: false, direction: 'portrait-ok' };
  }

  const cw = rotateCanvas90Clockwise(sourceCanvas);
  const ccw = rotateCanvas90CounterClockwise(sourceCanvas);
  const scoreCW = getTopTextDensityScore(cw);
  const scoreCCW = getTopTextDensityScore(ccw);

  if (scoreCW >= scoreCCW) {
    return { canvas: cw, rotated: true, direction: 'clockwise', scoreCW, scoreCCW };
  }
  return { canvas: ccw, rotated: true, direction: 'counter-clockwise', scoreCW, scoreCCW };
}

function upscaleCanvas(sourceCanvas, minWidth = MIN_OCR_WIDTH) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;

  if (w >= minWidth) {
    return { canvas: sourceCanvas, scaled: false, scale: 1 };
  }

  const scale = minWidth / Math.max(1, w);
  const out = document.createElement('canvas');
  out.width = Math.round(w * scale);
  out.height = Math.round(h * scale);
  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(sourceCanvas, 0, 0, out.width, out.height);

  return { canvas: out, scaled: true, scale };
}

function cvToCanvas(mat) {
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);
  return canvas;
}

function toGrayCanvas(sourceCanvas) {
  let src = null;
  let gray = null;
  try {
    src = cv.imread(sourceCanvas);
    gray = new cv.Mat();
    if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    else src.copyTo(gray);
    return cvToCanvas(gray);
  } finally {
    try { if (src) src.delete(); } catch (_) {}
    try { if (gray) gray.delete(); } catch (_) {}
  }
}

function bilateralCanvas(grayCanvas) {
  let src = null;
  let gray = null;
  let dst = null;
  try {
    src = cv.imread(grayCanvas);
    gray = new cv.Mat();
    dst = new cv.Mat();
    if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    else src.copyTo(gray);
    cv.bilateralFilter(gray, dst, 6, 40, 40, cv.BORDER_DEFAULT);
    return cvToCanvas(dst);
  } finally {
    [src, gray, dst].forEach((m) => { try { if (m) m.delete(); } catch (_) {} });
  }
}

function claheCanvas(inputCanvas) {
  let src = null;
  let gray = null;
  let dst = null;
  let clahe = null;
  try {
    src = cv.imread(inputCanvas);
    gray = new cv.Mat();
    dst = new cv.Mat();
    if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    else src.copyTo(gray);
    clahe = new cv.CLAHE(1.8, new cv.Size(10, 10));
    clahe.apply(gray, dst);
    return cvToCanvas(dst);
  } finally {
    [src, gray, dst].forEach((m) => { try { if (m) m.delete(); } catch (_) {} });
    try { if (clahe) clahe.delete(); } catch (_) {}
  }
}

function gaussianCanvas(inputCanvas) {
  let src = null;
  let gray = null;
  let dst = null;
  try {
    src = cv.imread(inputCanvas);
    gray = new cv.Mat();
    dst = new cv.Mat();
    if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    else src.copyTo(gray);
    cv.GaussianBlur(gray, dst, new cv.Size(0, 0), 0.7, 0.7);
    return cvToCanvas(dst);
  } finally {
    [src, gray, dst].forEach((m) => { try { if (m) m.delete(); } catch (_) {} });
  }
}

function sharpenCanvas(claheInputCanvas, blurredInputCanvas) {
  let claheSrc = null;
  let blurSrc = null;
  let claheGray = null;
  let blurGray = null;
  let sharp = null;
  try {
    claheSrc = cv.imread(claheInputCanvas);
    blurSrc = cv.imread(blurredInputCanvas);
    claheGray = new cv.Mat();
    blurGray = new cv.Mat();
    sharp = new cv.Mat();

    if (claheSrc.channels() === 4) cv.cvtColor(claheSrc, claheGray, cv.COLOR_RGBA2GRAY);
    else if (claheSrc.channels() === 3) cv.cvtColor(claheSrc, claheGray, cv.COLOR_RGB2GRAY);
    else claheSrc.copyTo(claheGray);

    if (blurSrc.channels() === 4) cv.cvtColor(blurSrc, blurGray, cv.COLOR_RGBA2GRAY);
    else if (blurSrc.channels() === 3) cv.cvtColor(blurSrc, blurGray, cv.COLOR_RGB2GRAY);
    else blurSrc.copyTo(blurGray);

    cv.addWeighted(claheGray, 1.25, blurGray, -0.25, 0, sharp);
    return cvToCanvas(sharp);
  } finally {
    [claheSrc, blurSrc, claheGray, blurGray, sharp].forEach((m) => { try { if (m) m.delete(); } catch (_) {} });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────
async function runPipeline() {
  if (!originalCanvas || isRunning) return;

  isRunning = true;
  runBtn.disabled = true;
  clearStepState();
  setProgress(0);
  setStatus('Sunum akışı çalışıyor...', 'work');

  try {
    await waitForOpenCvAndScanner();

    let idx = 0;

    setActiveStep('original');
    setSpeaker('İlk aşamada sistem kameradan veya dosyadan gelen ham fatura görüntüsünü alır. Bu görüntü henüz OCR için optimize edilmemiştir.');
    setPreview('original', originalCanvas, [{ text: 'input', cls: 'ok' }]);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('original');

    setActiveStep('detect');
    setSpeaker('Belge tespitinde dört köşe bulunur ve perspektif düzeltilmiş kırpılmış fatura üretilir. Bu kutuda artık overlay değil, tespitten sonra oluşan gerçek crop sonucu gösterilir.');
    const detected = findDocumentAndCrop(originalCanvas);
    const cropCanvas = detected.cropCanvas || originalCanvas;
    const detectTags = detected.found
      ? [
          { text: 'belge bulundu', cls: 'ok' },
          detected.method,
          detected.areaRatio != null ? `alan ${(detected.areaRatio * 100).toFixed(1)}%` : 'alan hesaplanmadı'
        ]
      : [
          { text: 'kontur bulunamadı', cls: 'warn' },
          detected.method,
          'fallback: orijinal görüntüyle devam'
        ];
    // Bu adımın gerçek çıktısı cropCanvas'tır. Overlay'i değil, kırpılmış belgeyi gösteriyoruz.
    setPreview('detect', cropCanvas, detectTags);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('detect');

    setActiveStep('orientation');
    setSpeaker('Yön düzeltme aşamasında belge yatay geldiyse 90 derece döndürme adayları denenir ve üst bölgedeki metin yoğunluğuna göre doğru yön seçilir.');
    const oriented = normalizeDocumentOrientationSmart(cropCanvas);
    setPreview('orientation', oriented.canvas, [
      oriented.rotated ? { text: 'döndürüldü', cls: 'ok' } : { text: 'dikey zaten uygun', cls: 'ok' },
      oriented.direction
    ]);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('orientation');

    setActiveStep('upscale');
    setSpeaker('Upscale aşamasında belge genişliği OCR için yeterli değilse büyütülür. Böylece küçük yazıların karakter ayrımı daha kolay olur.');
    const upscaled = upscaleCanvas(oriented.canvas, MIN_OCR_WIDTH);
    setPreview('upscale', upscaled.canvas, [
      upscaled.scaled ? { text: `x${upscaled.scale.toFixed(2)}`, cls: 'ok' } : { text: 'büyütme gerekmedi', cls: 'ok' },
      `minWidth ${MIN_OCR_WIDTH}`
    ]);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('upscale');

    setActiveStep('gray');
    setSpeaker('Grayscale aşamasında renk kanalları kaldırılır. OCR için daha sade, tek kanallı görüntü elde edilir.');
    const gray = toGrayCanvas(upscaled.canvas);
    setPreview('gray', gray, ['RGBA/RGB → GRAY', { text: 'tek kanal', cls: 'ok' }]);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('gray');

    setActiveStep('bilateral');
    setSpeaker('Bilateral filtre normal bulanıklaştırmadan farklı olarak yazı kenarlarını koruyarak gürültüyü azaltır.');
    const bilateral = bilateralCanvas(gray);
    setPreview('bilateral', bilateral, ['d=6', 'σColor=40', 'σSpace=40']);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('bilateral');

    setActiveStep('clahe');
    setSpeaker('CLAHE lokal kontrast artırımı yapar. Özellikle silik, gölgeli veya ışığı dengesiz faturalarda yazıları belirginleştirir.');
    const clahe = claheCanvas(bilateral);
    setPreview('clahe', clahe, ['clipLimit=1.8', 'tile=10×10', { text: 'kontrast arttı', cls: 'ok' }]);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('clahe');

    setActiveStep('gaussian');
    setSpeaker('Gaussian Blur aşaması çok küçük piksel gürültülerini dengeler. Bu çıktı sharpen adımında referans olarak kullanılır.');
    const gaussian = gaussianCanvas(clahe);
    setPreview('gaussian', gaussian, ['σ=0.7', 'mikro gürültü azaltma']);
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('gaussian');

    setActiveStep('sharpen');
    setSpeaker('Son aşamada unsharp mask mantığıyla karakter kenarları keskinleştirilir. Bu final görüntü PaddleOCR tarafına gönderilecek OCR varyantıdır.');
    const sharpened = sharpenCanvas(clahe, gaussian);
    setPreview('sharpen', sharpened, [
      'addWeighted(1.25, -0.25)',
      { text: 'OCR final', cls: 'ok' }
    ]);
    setVariantInfo({
      auto_scan: false,
      ocr_variant: 'enhanced',
      ui_variant: 'enhanced',
      detection_method: detected.method,
      detected: detected.found,
      output: 'sharpened_canvas',
      final_size: `${sharpened.width}x${sharpened.height}`
    });
    setProgress(idx++);
    await sleep(DEMO_DELAY_MS);
    setDoneStep('sharpen');

    setActiveStep('');
    setProgress(STEP_ORDER.length - 1);
    setStatus('Akış tamamlandı ✓', 'ready');
  } catch (err) {
    console.error(err);
    setStatus('Hata: ' + (err?.message || err), 'err');
  } finally {
    isRunning = false;
    runBtn.disabled = !originalCanvas;
  }
}

async function handleFile(file, autoRun = true) {
  if (!file) return;

  if (!/^image\//i.test(file.type)) {
    setStatus('Lütfen görsel dosyası seç: JPG / PNG / WEBP', 'err');
    return;
  }

  currentFile = file;
  clearAllPreviews();
  clearStepState();
  setProgress(0);
  setStatus('Görsel yükleniyor...', 'work');

  try {
    originalCanvas = await loadImageFileAsCanvas(file);
    setPreview('original', originalCanvas, [
      { text: file.name || 'image', cls: 'ok' },
      `${Math.round((file.size || 0) / 1024)} KB`
    ]);
    setStatus('Görsel hazır — akış başlatılıyor', 'ready');
    runBtn.disabled = false;
    resetBtn.disabled = false;

    if (autoRun) {
      await sleep(250);
      runPipeline();
    }
  } catch (err) {
    console.error(err);
    setStatus('Görsel okunamadı: ' + (err?.message || err), 'err');
  }
}

function resetAll() {
  currentFile = null;
  originalCanvas = null;
  fileInput.value = '';
  clearAllPreviews();
  clearStepState();
  setProgress(0);
  setSpeaker('Fatura yüklendiğinde sistem önce belgeyi tespit eder. Daha sonra yön düzeltme, büyütme, griye çevirme, gürültü azaltma, kontrast artırma ve keskinleştirme adımlarından geçirerek OCR için en okunabilir görüntüyü hazırlar.');
  setVariantInfo({
    ocr_variant: 'enhanced',
    auto_scan: false,
    output: 'sharpened_canvas'
  });
  setStatus('Görsel bekleniyor', '');
  runBtn.disabled = true;
  resetBtn.disabled = true;
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  handleFile(file, true);
});

runBtn.addEventListener('click', runPipeline);
resetBtn.addEventListener('click', resetAll);

dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('drag');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('drag');
  });
});

dropZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files?.[0];
  handleFile(file, true);
});


function openImageModal(img) {
  if (!imageModal || !imageModalImg || !img) return;
  imageModalImg.src = img.src;
  imageModalTitle.textContent = img.dataset.title || img.alt || 'Görsel';
  imageModal.classList.add('open');
  imageModal.setAttribute('aria-hidden', 'false');
}

function closeImageModal() {
  if (!imageModal || !imageModalImg) return;
  imageModal.classList.remove('open');
  imageModal.setAttribute('aria-hidden', 'true');
  imageModalImg.src = '';
}

document.addEventListener('click', (event) => {
  const img = event.target.closest('.preview img');
  if (img) openImageModal(img);
});

imageModalClose?.addEventListener('click', closeImageModal);
imageModal?.addEventListener('click', (event) => {
  if (event.target === imageModal) closeImageModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeImageModal();
});

window.addEventListener('load', async () => {
  resetAll();
  try {
    await waitForOpenCv(15000);
    setStatus('OpenCV hazır — görsel bekleniyor', 'ready');
    await waitForOpenCvAndScanner(2000);
  } catch (e) {
    setStatus('OpenCV yüklenemedi. İnternet / CDN erişimini kontrol et.', 'err');
    setScannerStatus('OpenCV yok', 'warn');
  }
});
