// ── DOM ──────────────────────────────────────────────────────────────────────
const fileEl      = document.getElementById('file');
const addPhotoBtn = document.getElementById('addPhoto');
const analyzeBtn  = document.getElementById('analyze');
const clearBtn    = document.getElementById('clear');
const copyBtn     = document.getElementById('copy');
const drop        = document.getElementById('drop');
const bar         = document.getElementById('bar');
const meta        = document.getElementById('meta');
const fname       = document.getElementById('fname');
const fsize       = document.getElementById('fsize');
const dot         = document.getElementById('dot');
const sel         = document.getElementById('sel');
const out         = document.getElementById('out');
const err         = document.getElementById('err');
const imgPanel    = document.getElementById('imgPanel');
const imgView     = document.getElementById('imgView');
const tabOrig     = document.getElementById('tabOrig');
const tabEnhanced = document.getElementById('tabEnhanced');
const debugWrap   = document.getElementById('debugWrap');

const elFirma    = document.getElementById('firma');
const elTur      = document.getElementById('tur');
const elSozlesme = document.getElementById('sozlesme');
const elTesisat  = document.getElementById('tesisat');
const elMusteri  = document.getElementById('musteri');
const elFaturaNo = document.getElementById('fatura_no');
const elBelgeNo  = document.getElementById('belge_no');
const elSonOdeme = document.getElementById('son_odeme');
const elTutar    = document.getElementById('tutar');
const elOcrTime  = document.getElementById('ocr_time');
const elDocId    = document.getElementById('document_id');
const ocrDot     = document.getElementById('ocrDot');
const ocrStatus  = document.getElementById('ocrStatus');

const choiceSheet  = document.getElementById('choiceSheet');
const choiceClose  = document.getElementById('choiceClose');
const choiceCamera = document.getElementById('choiceCamera');
const choiceFile   = document.getElementById('choiceFile');
const cameraSheet  = document.getElementById('cameraSheet');
const sClose       = document.getElementById('sClose');
const sSnap        = document.getElementById('sSnap');
const sInfo        = document.getElementById('sInfo');
const video        = document.getElementById('video');
const cam          = document.getElementById('cam');

// ── State ────────────────────────────────────────────────────────────────────
drop._droppedFile = null;

let stream = null;
let scanning = false;
let scanAttempts = 0;
let stableDetections = 0;

const MAX_SCAN_ATTEMPTS = 40;
const STABLE_FRAMES_REQUIRED = 5;
const OCR_TIMEOUT_MS = 90000;

let origSrc       = null;
let enhancedB64   = null;
let activeTab     = 'orig';
let debugOpen     = false;

let processedFile = null;
let jscanifyReady = false;
let scanner = null;

let scanInProgress = false;
let lastScanTime = 0;
const LIVE_SCAN_INTERVAL = 220;

const LIVE_W = 960;
const LIVE_H = 720;

// canlı preview yardımcı canvasları
const liveCanvas = document.createElement('canvas');
const liveCtx = liveCanvas.getContext('2d');

// ── OpenCV + jscanify readiness ──────────────────────────────────────────────
function waitForOpenCvAndScanner(timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    if (jscanifyReady && scanner) {
      resolve();
      return;
    }

    const startedAt = Date.now();

    function tryInit() {
      try {
        if (window.cv && window.jscanify) {
          if (typeof cv.imread === 'function') {
            scanner = scanner || new window.jscanify();
            jscanifyReady = true;
            resolve();
            return;
          }
        }
      } catch (_) {
        // polling devam
      }

      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error('OpenCV / jscanify yüklenemedi'));
        return;
      }

      setTimeout(tryInit, 120);
    }

    tryInit();
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function bytesToSize(n) {
  if (!n) return '0 B';
  const s = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(n) / Math.log(1024));
  return (n / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + s[i];
}

function formatDur(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  if (isNaN(n)) return String(v);
  return n.toFixed(3) + ' sn';
}

function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(id));
}

async function tryAutoFocus() {
  if (!stream) return;

  const track = stream.getVideoTracks?.()[0];
  if (!track) return;

  const caps = typeof track.getCapabilities === 'function'
    ? track.getCapabilities()
    : {};

  const constraints = {};

  if (Array.isArray(caps.focusMode) && caps.focusMode.includes('continuous')) {
    constraints.focusMode = 'continuous';
  } else if (Array.isArray(caps.focusMode) && caps.focusMode.includes('single-shot')) {
    constraints.focusMode = 'single-shot';
  }

  if (!Object.keys(constraints).length) return;

  try {
    await track.applyConstraints({ advanced: [constraints] });
  } catch (e) {
    console.warn('Autofocus uygulanamadı:', e);
  }
}

function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.95) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), type, quality);
  });
}

function canvasToBase64(canvas, type = 'image/jpeg', quality = 0.95) {
  return canvas.toDataURL(type, quality).split(',')[1];
}

function cloneCanvas(srcCanvas) {
  const c = document.createElement('canvas');
  c.width = srcCanvas.width;
  c.height = srcCanvas.height;
  c.getContext('2d').drawImage(srcCanvas, 0, 0);
  return c;
}

// ── Orientation helpers ──────────────────────────────────────────────────────
function rotateCanvas90Clockwise(srcCanvas) {
  const out = document.createElement('canvas');
  out.width = srcCanvas.height;
  out.height = srcCanvas.width;

  const ctx = out.getContext('2d');
  ctx.translate(out.width / 2, out.height / 2);
  ctx.rotate(Math.PI / 2);
  ctx.drawImage(srcCanvas, -srcCanvas.width / 2, -srcCanvas.height / 2);

  return out;
}

function rotateCanvas90CounterClockwise(srcCanvas) {
  const out = document.createElement('canvas');
  out.width = srcCanvas.height;
  out.height = srcCanvas.width;

  const ctx = out.getContext('2d');
  ctx.translate(out.width / 2, out.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.drawImage(srcCanvas, -srcCanvas.width / 2, -srcCanvas.height / 2);

  return out;
}

function getRegionDarkRatio(ctx, x, y, w, h, threshold = 175) {
  if (w <= 0 || h <= 0) return 0;

  const data = ctx.getImageData(x, y, w, h).data;
  let dark = 0;
  let total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

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

function normalizeDocumentOrientationSmart(srcCanvas) {
  if (!srcCanvas) return srcCanvas;

  if (srcCanvas.height >= srcCanvas.width) {
    return srcCanvas;
  }

  const cw = rotateCanvas90Clockwise(srcCanvas);
  const ccw = rotateCanvas90CounterClockwise(srcCanvas);

  const scoreCW = getTopTextDensityScore(cw);
  const scoreCCW = getTopTextDensityScore(ccw);

  return scoreCW >= scoreCCW ? cw : ccw;
}

function upscaleCanvas(canvas, minWidth = 1800) {
  const w = canvas.width;
  const h = canvas.height;

  if (w >= minWidth) return canvas;

  const scale = minWidth / w;
  const out = document.createElement('canvas');
  out.width = Math.round(w * scale);
  out.height = Math.round(h * scale);

  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, out.width, out.height);

  return out;
}

// ── Geometry helpers ─────────────────────────────────────────────────────────
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

function isValidDocumentQuad(corners, canvasWidth, canvasHeight) {
  console.log('  🔐 isValidDocumentQuad kontrol başladı');

  if (!corners || corners.length !== 4) {
    console.log('    ❌ corners length !== 4');
    return false;
  }

  const pts = orderPoints(corners);

  for (const p of pts) {
    if (p.x < 0 || p.x > canvasWidth || p.y < 0 || p.y > canvasHeight) {
      console.log('    ❌ köşe sınır dışı:', p);
      return false;
    }
  }

  const area = polygonArea(pts);
  const imgArea = canvasWidth * canvasHeight;
  const areaRatio = area / imgArea;
  console.log('    ✓ areaRatio:', areaRatio.toFixed(3));

  if (areaRatio < 0.20 || areaRatio > 0.95) {
    console.log('    ❌ areaRatio dışında: ' + areaRatio.toFixed(3));
    return false;
  }

  const topW = distance(pts[0], pts[1]);
  const bottomW = distance(pts[3], pts[2]);
  const leftH = distance(pts[0], pts[3]);
  const rightH = distance(pts[1], pts[2]);

  const avgW = (topW + bottomW) / 2;
  const avgH = (leftH + rightH) / 2;
  console.log('    ✓ avgW:', avgW.toFixed(1), 'avgH:', avgH.toFixed(1));

  if (avgW < 120 || avgH < 120) {
    console.log('    ❌ avgW/H çok küçük');
    return false;
  }

  const longSide = Math.max(avgW, avgH);
  const shortSide = Math.max(1, Math.min(avgW, avgH));
  const aspect = longSide / shortSide;
  console.log('    ✓ aspect:', aspect.toFixed(2));
  if (aspect < 1.2 || aspect > 4.2) {
    console.log('    ❌ aspect ratio dışında: ' + aspect.toFixed(2));
    return false;
  }

  const widthBalance = Math.min(topW, bottomW) / Math.max(topW, bottomW);
  const heightBalance = Math.min(leftH, rightH) / Math.max(leftH, rightH);
  console.log('    ✓ widthBalance:', widthBalance.toFixed(2), 'heightBalance:', heightBalance.toFixed(2));

  const widthRatio = Math.max(topW, bottomW) / Math.max(1, Math.min(topW, bottomW));
  const heightRatio = Math.max(leftH, rightH) / Math.max(1, Math.min(leftH, rightH));
  console.log('    ✓ widthRatio:', widthRatio.toFixed(2), 'heightRatio:', heightRatio.toFixed(2));

  if (widthBalance < 0.50 || heightBalance < 0.50) {
    console.log('    ❌ balance çok düşük');
    return false;
  }

  if (widthRatio > 1.75) {
    console.log('    ❌ üst-alt genişlik farkı fazla');
    return false;
  }

  if (heightRatio > 1.55) {
    console.log('    ❌ sol-sağ yükseklik farkı fazla');
    return false;
  }

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

    if (mag1 < 1 || mag2 < 1) {
      console.log('    ❌ magnitude çok küçük');
      return false;
    }

    const cos = dot / (mag1 * mag2);
    const angle = Math.acos(Math.max(-1, Math.min(1, cos))) * (180 / Math.PI);
    angles.push(angle);
  }

  console.log('    ✓ angles:', angles.map(a => a.toFixed(1)).join(', '));

  const validAngles = angles.every(a => a > 60 && a < 120);
  if (!validAngles) {
    console.log('    ❌ açılar dışında');
    return false;
  }

  const minEdge = Math.min(
    distance(pts[0], pts[1]),
    distance(pts[1], pts[2]),
    distance(pts[2], pts[3]),
    distance(pts[3], pts[0])
  );
  console.log('    ✓ minEdge:', minEdge.toFixed(1));

  if (minEdge < 40) {
    console.log('    ❌ minEdge çok küçük');
    return false;
  }

  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;

  const dx = Math.abs(cx - canvasWidth / 2) / canvasWidth;
  const dy = Math.abs(cy - canvasHeight / 2) / canvasHeight;
  console.log('    ✓ center offset - dx:', dx.toFixed(3), 'dy:', dy.toFixed(3));

  if (dx > 0.28 || dy > 0.28) {
    console.log('    ❌ merkez offset çok büyük');
    return false;
  }

  console.log('    ✅ tüm kontroller PASS');
  return true;
}
function tryExtractPaper(sourceCanvas) {
  console.log('📄 tryExtractPaper çağrıldı');
  try {
    const mat = cv.imread(sourceCanvas);
    console.log('  ✓ cv.imread başarılı');

    const contour = scanner.findPaperContour(mat);
    mat.delete();
    console.log('  ✓ findPaperContour:', contour ? 'bulundu' : 'bulunamadı');

    if (!contour) {
      console.log('  ❌ contour null, return null');
      return null;
    }

    const corners = scanner.getCornerPoints(contour);
    console.log('  ✓ getCornerPoints:', corners ? 'başarılı' : 'null');
    if (!corners) {
      console.log('  ❌ corners null, return null');
      return null;
    }

    const rawCorners = [
      corners.topLeftCorner,
      corners.topRightCorner,
      corners.bottomRightCorner,
      corners.bottomLeftCorner
    ].filter(Boolean);

    console.log('  ✓ rawCorners sayısı:', rawCorners.length);
    if (rawCorners.length !== 4) {
      console.log('  ❌ rawCorners !== 4, return null');
      return null;
    }

    console.log('  🔎 isValidDocumentQuad çağrılıyor...');
    if (!isValidDocumentQuad(rawCorners, sourceCanvas.width, sourceCanvas.height)) {
      console.log('  ❌ isValidDocumentQuad RED, return null');
      return null;
    }
    console.log('  ✓ isValidDocumentQuad PASS');

    let ordered = orderPoints(rawCorners);
    ordered = expandQuadHorizontally(ordered, sourceCanvas.width, 0.14);

    const [tl, tr, br, bl] = ordered;

    

    const realW = Math.round(Math.max(
      distance(tl, tr),
      distance(bl, br)
    ));

    const realH = Math.round(Math.max(
      distance(tl, bl),
      distance(tr, br)
    ));

    console.log('  ✓ realW:', realW, 'realH:', realH);

    const widthCoverage = realW / sourceCanvas.width;
    const heightCoverage = realH / sourceCanvas.height;

    const longCoverage = Math.max(widthCoverage, heightCoverage);
    const shortCoverage = Math.min(widthCoverage, heightCoverage);

    console.log(
      '  ✓ widthCoverage:', widthCoverage.toFixed(3),
      'heightCoverage:', heightCoverage.toFixed(3),
      'longCoverage:', longCoverage.toFixed(3),
      'shortCoverage:', shortCoverage.toFixed(3)
    );

    if (longCoverage < 0.55) {
      console.log('  ❌ belge yeterince alan kaplamıyor, muhtemelen yanlış crop');
      return null;
    }

    if (shortCoverage < 0.20) {
      console.log('  ❌ belge kısa kenarda fazla dar, muhtemelen yanlış crop');
      return null;
    }

    const maxSide = 1600;
    const scale = Math.min(maxSide / Math.max(realW, realH), 1.0);
    const outW = Math.round(realW * scale);
    const outH = Math.round(realH * scale);

    console.log('  ✓ extractPaper çağrılıyor... outW:', outW, 'outH:', outH);
    const result = scanner.extractPaper(sourceCanvas, outW, outH, {
      topLeftCorner: tl,
      topRightCorner: tr,
      bottomRightCorner: br,
      bottomLeftCorner: bl
    });

    console.log('✅ tryExtractPaper başarılı:', result ? 'canvas döndü' : 'null');
    return result || null;

  } catch (e) {
    console.error('❌ tryExtractPaper hata:', e);
    return null;
  }
}

// ── Kalite ölçümü ─────────────────────────────────────────────────────────────
function estimateImageQuality(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const img = ctx.getImageData(0, 0, w, h).data;

  let mean = 0;
  let variance = 0;
  const pixels = w * h;

  for (let i = 0; i < img.length; i += 4) {
    const gray = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    mean += gray;
  }

  mean /= Math.max(pixels, 1);

  for (let i = 0; i < img.length; i += 4) {
    const gray = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    variance += Math.pow(gray - mean, 2);
  }

  variance /= Math.max(pixels, 1);
  return variance;
}

// ── Enhance ──────────────────────────────────────────────────────────────────
// ── Kalite ölçümü (Multi-metric) ─────────────────────────────────────────────
function estimateImageQuality(canvas) {
  console.log('📊 estimateImageQuality başladı');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const img = ctx.getImageData(0, 0, w, h).data;

  let mean = 0;
  let darkPixels = 0;
  let blackPixels = 0;
  let whitePixels = 0;
  
  // Pass 1: Mean ve pixel oranları
  for (let i = 0; i < img.length; i += 4) {
    const gray = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    mean += gray;
    if (gray < 175) darkPixels++;
    if (gray < 50) blackPixels++;
    if (gray > 240) whitePixels++;
  }
  
  const pixels = w * h;
  mean /= pixels;
  const darkRatio = darkPixels / pixels;
  const textRatio = blackPixels / pixels;
  const whiteRatio = whitePixels / pixels;
  
  console.log(`  ✓ mean: ${mean.toFixed(1)}, darkRatio: ${(darkRatio*100).toFixed(1)}%, textRatio: ${(textRatio*100).toFixed(1)}%, whiteRatio: ${(whiteRatio*100).toFixed(1)}%`);
  
  // Pass 2: Variance (sharpness)
  let variance = 0;
  for (let i = 0; i < img.length; i += 4) {
    const gray = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    variance += Math.pow(gray - mean, 2);
  }
  variance /= pixels;
  
  // Pass 3: Contrast (MAD - Mean Absolute Deviation)
  let contrast = 0;
  for (let i = 0; i < img.length; i += 4) {
    const gray = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    contrast += Math.abs(gray - mean);
  }
  contrast /= pixels;
  
  console.log(`  ✓ variance: ${variance.toFixed(1)}, contrast: ${contrast.toFixed(1)}`);
  
  // Score hesapla (0-100)
  let score = 50;
  
  // 1. Brightness kontrolü (ideal: 150-220)
  if (mean < 60) {
    score -= 30;
    console.log(`    ❌ Çok karanlık (${mean.toFixed(0)})`);
  } else if (mean < 100) {
    score -= 20;
    console.log(`    ⚠️ Karanlık (${mean.toFixed(0)})`);
  } else if (mean > 220) {
    score -= 25;
    console.log(`    ❌ Çok aydınlık (${mean.toFixed(0)})`);
  } else if (mean > 180) {
    score -= 10;
    console.log(`    ⚠️ Biraz aydınlık (${mean.toFixed(0)})`);
  } else if (mean >= 150 && mean <= 200) {
    score += 25;
    console.log(`    ✅ İdeal brightness (${mean.toFixed(0)})`);
  } else {
    score += 15;
    console.log(`    ✓ Kabul edilebilir brightness (${mean.toFixed(0)})`);
  }
  
  // 2. Variance (Sharpness/Blur) kontrolü
  if (variance < 100) {
    score -= 35;
    console.log(`    ❌ Çok blur (variance: ${variance.toFixed(0)})`);
  } else if (variance < 300) {
    score -= 20;
    console.log(`    ⚠️ Blur (variance: ${variance.toFixed(0)})`);
  } else if (variance > 2000) {
    score += 20;
    console.log(`    ✅ Sharp (variance: ${variance.toFixed(0)})`);
  } else if (variance > 800) {
    score += 15;
    console.log(`    ✅ İyi sharp (variance: ${variance.toFixed(0)})`);
  } else {
    score += 10;
    console.log(`    ✓ Kabul edilebilir sharp (variance: ${variance.toFixed(0)})`);
  }
  
  // 3. Contrast kontrolü
  if (contrast < 20) {
    score -= 30;
    console.log(`    ❌ Düşük contrast (${contrast.toFixed(1)})`);
  } else if (contrast < 40) {
    score -= 15;
    console.log(`    ⚠️ Az contrast (${contrast.toFixed(1)})`);
  } else if (contrast > 100) {
    score += 20;
    console.log(`    ✅ İyi contrast (${contrast.toFixed(1)})`);
  } else if (contrast > 60) {
    score += 15;
    console.log(`    ✓ Kabul edilebilir contrast (${contrast.toFixed(1)})`);
  } else {
    score += 10;
    console.log(`    ✓ Orta contrast (${contrast.toFixed(1)})`);
  }
  
  // 4. Text miktarı (belge için kritik)
  if (textRatio < 0.5) {
    score -= 30;
    console.log(`    ❌ Çok az siyah piksel (${(textRatio*100).toFixed(1)}%)`);
  } else if (textRatio < 2) {
    score -= 20;
    console.log(`    ⚠️ Az siyah piksel (${(textRatio*100).toFixed(1)}%)`);
  } else if (textRatio > 2 && textRatio < 25) {
    score += 25;
    console.log(`    ✅ İdeal text miktarı (${(textRatio*100).toFixed(1)}%)`);
  } else if (textRatio > 25) {
    score += 10;
    console.log(`    ✓ Çok text (${(textRatio*100).toFixed(1)}%)`);
  }
  
  // 5. Dark pixels kontrolü (belge benzeri)
  if (darkRatio < 5) {
    score -= 20;
    console.log(`    ⚠️ Az dark piksel (${(darkRatio*100).toFixed(1)}%)`);
  } else if (darkRatio >= 10 && darkRatio <= 50) {
    score += 20;
    console.log(`    ✅ İdeal dark piksel (${(darkRatio*100).toFixed(1)}%)`);
  } else if (darkRatio > 50) {
    score -= 15;
    console.log(`    ⚠️ Çok dark piksel (${(darkRatio*100).toFixed(1)}%)`);
  }
  
  // 6. White space kontrolü (çok beyaz = kötü)
  if (whiteRatio > 95) {
    score -= 25;
    console.log(`    ❌ Çok white space (${(whiteRatio*100).toFixed(1)}%)`);
  } else if (whiteRatio > 80) {
    score -= 15;
    console.log(`    ⚠️ Fazla white space (${(whiteRatio*100).toFixed(1)}%)`);
  }
  
  // Normalize (0-100)
  score = Math.max(0, Math.min(100, score));
  
  console.log(`📊 Final quality score: ${score}`);
  return score;
}

// ── Enhanced enhancement function ────────────────────────────────────────────
function enhanceDocumentCanvas(inputCanvas) {
  let src = null;
  let gray = null;
  let bilateral = null;
  let claheOut = null;
  let blurred = null;
  let sharp = null;

  try {
    const work = document.createElement('canvas');
    work.width = inputCanvas.width;
    work.height = inputCanvas.height;
    work.getContext('2d').drawImage(inputCanvas, 0, 0);

    src = cv.imread(work);
    gray = new cv.Mat();
    bilateral = new cv.Mat();
    claheOut = new cv.Mat();
    blurred = new cv.Mat();
    sharp = new cv.Mat();

    if (src.channels() === 4) {
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    } else if (src.channels() === 3) {
      cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    } else {
      src.copyTo(gray);
    }

    // HER ZAMAN GRİ (threshold yok)
    cv.bilateralFilter(gray, bilateral, 6, 40, 40, cv.BORDER_DEFAULT);

    const clahe = new cv.CLAHE(1.8, new cv.Size(10, 10));
    clahe.apply(bilateral, claheOut);
    clahe.delete();

    cv.GaussianBlur(claheOut, blurred, new cv.Size(0, 0), 0.7, 0.7);
    cv.addWeighted(claheOut, 1.25, blurred, -0.25, 0, sharp);

    const outCanvas = document.createElement('canvas');
    outCanvas.width = work.width;
    outCanvas.height = work.height;
    cv.imshow(outCanvas, sharp);

    return outCanvas;
  } finally {
    [src, gray, bilateral, claheOut, blurred, sharp].forEach(m => {
      try { if (m) m.delete(); } catch (_) {}
    });
  }
}

// ── Result boxes ─────────────────────────────────────────────────────────────
function setBox(elId, val, opts = {}) {
  const el  = document.getElementById(elId);
  const box = document.getElementById('box_' + elId);
  if (!el) return;

  const isEmpty = val === null || val === undefined || val === '' || val === 'null';
  if (isEmpty) {
    el.textContent = '—';
    el.classList.add('empty');
    if (box) box.classList.remove('has-value', 'animate');
  } else {
    const display = opts.suffix ? val + ' ' + opts.suffix : val;
    el.textContent = display;
    el.classList.remove('empty');
    if (box) {
      box.classList.remove('animate');
      void box.offsetWidth;
      box.classList.add('has-value', 'animate');
    }
  }
}

const RESULT_FIELDS = [
  'firma','tur','sozlesme','tesisat','musteri',
  'fatura_no','belge_no','son_odeme','tutar','ocr_time','document_id'
];

function clearResults() {
  RESULT_FIELDS.forEach(id => setBox(id, null));
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(tab) {
  activeTab = tab;
  tabOrig.classList.toggle('active', tab === 'orig');
  tabEnhanced.classList.toggle('active', tab === 'enhanced');

  if (tab === 'orig' && origSrc) {
    imgView.innerHTML = `<span class="img-label">Orijinal</span><img src="${origSrc}" alt="Orijinal">`;
  } else if (tab === 'enhanced' && enhancedB64) {
    imgView.innerHTML = `
      <span class="img-label">İyileştirilmiş</span>
      <span class="diff-badge">Enhanced</span>
      <img src="data:image/jpeg;base64,${enhancedB64}" alt="Enhanced">
    `;
  } else {
    imgView.innerHTML = `
      <div class="img-placeholder">
        <div class="ico">⏳</div>
        <span>Henüz işlenmedi</span>
      </div>
    `;
  }
}

// ── Reset UI ──────────────────────────────────────────────────────────────────
function resetUI() {
  fileEl.value = '';
  drop._droppedFile = null;
  processedFile = null;

  if (origSrc) URL.revokeObjectURL(origSrc);
  origSrc = null;
  enhancedB64 = null;
  activeTab = 'orig';

  imgPanel.style.display = 'none';
  imgView.innerHTML = `<div class="img-placeholder"><div class="ico">🖼</div><span>Görsel yükle</span></div>`;
  tabOrig.classList.add('active');
  tabEnhanced.classList.remove('active');
  tabEnhanced.disabled = true;

  meta.style.display = 'none';
  dot.className = 'dot off';
  sel.textContent = 'Seçilmedi';
  analyzeBtn.disabled = true;
  clearBtn.disabled = true;
  copyBtn.disabled = true;

  bar.style.display = 'none';
  err.style.display = 'none';
  err.textContent = '';
  debugWrap.style.display = 'none';
  out.textContent = '{}';

  clearResults();
  ocrDot.className = 'dot off';
  ocrStatus.textContent = 'Bekleniyor';
}

function setFile(f, skipAnalyze = false) {
  if (!f) {
    resetUI();
    return;
  }

  dot.className = 'dot';
  sel.textContent = 'Hazır';
  analyzeBtn.disabled = false;
  clearBtn.disabled = false;
  fname.textContent = f.name || 'capture.jpg';
  fsize.textContent = bytesToSize(f.size || 0);
  meta.style.display = 'flex';

  if (origSrc) URL.revokeObjectURL(origSrc);
  origSrc = URL.createObjectURL(f);

  imgPanel.style.display = 'block';
  tabOrig.classList.add('active');
  imgView.innerHTML = `<span class="img-label">Orijinal</span><img src="${origSrc}" alt="Orijinal">`;

  err.style.display = 'none';
  debugWrap.style.display = 'none';
  out.textContent = '{}';
  clearResults();

  ocrDot.className = 'dot off';
  ocrStatus.textContent = 'Hazır';

  if (!skipAnalyze) {
    setTimeout(() => runAnalyze(), 500);
  }
}

function toggleDebug() {
  debugOpen = !debugOpen;
  const inner = document.getElementById('debugInner');
  const arrow = document.getElementById('debugArrow');
  inner.style.display = debugOpen ? 'flex' : 'none';
  if (arrow) arrow.textContent = debugOpen ? '▲' : '▼';
}

// ── Camera overlay ────────────────────────────────────────────────────────────
const INTRO_MESSAGES = [
  { id: 'i1', msg: 'Faturayı çerçeve içine yerleştirin', type: '', delay: 400,  dur: 2200 },
  { id: 'i2', msg: 'Belgeyi dik ve sabit tutun', type: '', delay: 2900, dur: 2200 },
  { id: 'i3', msg: 'Tüm köşelerin görünür olduğundan emin olun', type: '', delay: 5400, dur: 2400 },
];

let _introTimers = [];
let _toastTimers = {};

function _buildOverlayDOM(camEl) {
  if (camEl.querySelector('.cam-overlay-root')) return;

  const dimmer = document.createElement('div');
  dimmer.className = 'cam-dimmer';
  dimmer.id = 'camDimmer';

  const frame = document.createElement('div');
  frame.className = 'cam-frame';
  frame.id = 'camFrame';
  frame.innerHTML = `
    <div class="cam-frame-border" id="camFrameBorder"></div>
    <div class="cam-corner tl"></div>
    <div class="cam-corner tr"></div>
    <div class="cam-corner bl"></div>
    <div class="cam-corner br"></div>
    <div class="cam-scan-line" id="camScanLine"></div>
    <canvas id="camDetectCanvas" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:.95;"></canvas>
  `;

  const toastWrap = document.createElement('div');
  toastWrap.className = 'cam-toast-wrap';
  toastWrap.id = 'camToastWrap';

  const root = document.createElement('div');
  root.className = 'cam-overlay-root';
  root.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:2;';
  root.appendChild(dimmer);
  root.appendChild(frame);
  root.appendChild(toastWrap);

  camEl.appendChild(root);
}

function _positionFrame() {
  const camEl   = document.getElementById('cam');
  const frameEl = document.getElementById('camFrame');
  const dimmer  = document.getElementById('camDimmer');
  if (!camEl || !frameEl || !dimmer) return;

  const cw = camEl.offsetWidth;
  const ch = camEl.offsetHeight;
  if (!cw || !ch) return;

  const fw = Math.round(cw * 0.70);
  const fh = Math.min(Math.round(fw * 2.4), Math.round(ch * 0.85));
  const fx = Math.round((cw - fw) / 2);
  const fy = Math.round((ch - fh) / 2);

  frameEl.style.cssText = `position:absolute;left:${fx}px;top:${fy}px;width:${fw}px;height:${fh}px;`;

  const x0 = fx, y0 = fy, x1 = fx + fw, y1 = fy + fh;
  dimmer.style.clipPath = `polygon(
    0px 0px,${cw}px 0px,${cw}px ${ch}px,0px ${ch}px,0px 0px,
    ${x0}px ${y0}px,${x0}px ${y1}px,${x1}px ${y1}px,${x1}px ${y0}px,${x0}px ${y0}px
  )`;
}

function _showCamToast(id, msg, type, durationMs) {
  const wrap = document.getElementById('camToastWrap');
  if (!wrap) return;

  let toast = document.getElementById('camToast_' + id);
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'camToast_' + id;
    toast.className = 'cam-toast';
    wrap.appendChild(toast);
  }

  toast.className = 'cam-toast' + (type ? ' ' + type : '');
  toast.textContent = msg;

  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('visible')));

  if (_toastTimers[id]) clearTimeout(_toastTimers[id]);
  if (durationMs > 0) {
    _toastTimers[id] = setTimeout(() => {
      toast.classList.remove('visible');
    }, durationMs);
  }
}

function _hideAllToasts() {
  const wrap = document.getElementById('camToastWrap');
  if (!wrap) return;
  wrap.querySelectorAll('.cam-toast').forEach(t => t.classList.remove('visible'));
}

function _setOverlayState(state) {
  const border   = document.getElementById('camFrameBorder');
  const scanLine = document.getElementById('camScanLine');
  if (!border) return;

  border.className = 'cam-frame-border';
  if (state === 'detecting') border.classList.add('detecting');
  if (state === 'detected') border.classList.add('detected');

  if (scanLine) {
    scanLine.style.animationPlayState = state === 'detected' ? 'paused' : 'running';
  }
}

function clearDetectCanvas() {
  const detectCanvas = document.getElementById('camDetectCanvas');
  if (!detectCanvas) return;
  const ctx = detectCanvas.getContext('2d');
  detectCanvas.width = detectCanvas.clientWidth || 1;
  detectCanvas.height = detectCanvas.clientHeight || 1;
  ctx.clearRect(0, 0, detectCanvas.width, detectCanvas.height);
}

function drawHighlightOnOverlay(resultCanvas) {
  const detectCanvas = document.getElementById('camDetectCanvas');
  if (!detectCanvas || !resultCanvas) return;

  const ctx = detectCanvas.getContext('2d');
  detectCanvas.width = detectCanvas.clientWidth || 1;
  detectCanvas.height = detectCanvas.clientHeight || 1;

  ctx.clearRect(0, 0, detectCanvas.width, detectCanvas.height);
  ctx.drawImage(resultCanvas, 0, 0, detectCanvas.width, detectCanvas.height);
}

function _playIntro() {
  _introTimers.forEach(t => clearTimeout(t));
  _introTimers = [];

  INTRO_MESSAGES.forEach(({ id, msg, type, delay, dur }) => {
    _introTimers.push(setTimeout(() => _showCamToast(id, msg, type, dur), delay));
  });
}

function _stopIntro() {
  _introTimers.forEach(t => clearTimeout(t));
  _introTimers = [];
}

function initCameraOverlay() {
  const camEl = document.getElementById('cam');
  if (!camEl) return;
  _buildOverlayDOM(camEl);

  setTimeout(_positionFrame, 80);
  setTimeout(_positionFrame, 300);

  window.addEventListener('resize', _positionFrame);
  _setOverlayState('idle');
  _playIntro();
}

function destroyCameraOverlay() {
  _stopIntro();
  _hideAllToasts();
  Object.values(_toastTimers).forEach(t => clearTimeout(t));
  window.removeEventListener('resize', _positionFrame);
}

// ── jscanify helpers ──────────────────────────────────────────────────────────
function tryHighlightPaper(sourceCanvas) {
  console.log('🔍 tryHighlightPaper çağrıldı');
  try {
    const highlighted = scanner.highlightPaper(sourceCanvas, {
      color: 'rgb(0,255,120)',
      thickness: 4
    });
    console.log('✅ tryHighlightPaper sonuç:', highlighted ? 'başarılı' : 'null');
    return highlighted || null;
  } catch (e) {
    console.error('❌ tryHighlightPaper hata:', e);
    return null;
  }
}

async function buildProcessedFileFromCanvas(scannedCanvas, fallbackFileName = 'camera_scanned.jpg') {
  console.log('🔄 buildProcessedFileFromCanvas başladı:', fallbackFileName);
  
  if (!scannedCanvas) {
    console.log('❌ scannedCanvas null');
    return null;
  }

  try {
    console.log('  ✓ normalizeDocumentOrientationSmart...');
    let normalizedCanvas = normalizeDocumentOrientationSmart(scannedCanvas);
    console.log('  ✓ normalized:', normalizedCanvas.width, 'x', normalizedCanvas.height);
    
    console.log('  ✓ upscaleCanvas hemen yapılıyor...');
    let upscaledCanvas = upscaleCanvas(normalizedCanvas, 1800);
    console.log('  ✓ upscaled:', upscaledCanvas.width, 'x', upscaledCanvas.height);
    
    console.log('  ✓ enhanceDocumentCanvas...');
    const enhancedCanvas = enhanceDocumentCanvas(upscaledCanvas);
    console.log('  ✓ enhanced:', enhancedCanvas.width, 'x', enhancedCanvas.height);

    console.log('  ✓ canvasToBlob...');
    const blob = await canvasToBlob(enhancedCanvas, 'image/jpeg', 0.95);
    console.log('  ✓ blob:', blob?.size, 'bytes');
    
    if (!blob) {
      console.log('❌ blob null');
      return null;
    }

    console.log('  ✓ canvasToBase64...');
    enhancedB64 = canvasToBase64(enhancedCanvas, 'image/jpeg', 0.95);
    console.log('  ✓ enhancedB64 length:', enhancedB64?.length);
    
    const file = new File([blob], fallbackFileName, { type: 'image/jpeg' });
    console.log('✅ DONE:', file.name, file.size, 'bytes\n');
    return file;
  } catch (e) {
    console.error('❌ exception:', e);
    return null;
  }
}

async function processSelectedFileWithJscanify(file) {
  try {
    await waitForOpenCvAndScanner();
  } catch (e) {
    err.style.display = 'block';
    err.textContent = 'jscanify hazır değil: ' + e.message;
    return;
  }

  try {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);

    await new Promise((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = reject;
      img.src = objectUrl;
    });

    console.log('📸 Image yüklendi:', img.naturalWidth, 'x', img.naturalHeight);

    // Image'ı canvas'a çevir
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = img.naturalWidth;
    imgCanvas.height = img.naturalHeight;
    imgCanvas.getContext('2d').drawImage(img, 0, 0);

    console.log('🖼️ imgCanvas oluşturuldu');

    const scannedCanvas = tryExtractPaper(imgCanvas);
    console.log('🔍 tryExtractPaper sonuç:', scannedCanvas ? 'başarılı' : 'başarısız');

    if (scannedCanvas) {
      console.log('✅ Crop başarılı!');
      processedFile = await buildProcessedFileFromCanvas(scannedCanvas, 'selected_scanned_enhanced.jpg');
      tabEnhanced.disabled = false;
      if (activeTab === 'enhanced') showTab('enhanced');
    } else {
      console.log('⚠️ Crop başarısız, orijinal enhance ediliyor');
      // ✅ NORMALIZE + UPSCALE + ENHANCE YAP ÖNCE
      const normalizedCanvas = normalizeDocumentOrientationSmart(imgCanvas);
      const upscaledCanvas = upscaleCanvas(normalizedCanvas, 1800);
      const enhancedCanvas = enhanceDocumentCanvas(upscaledCanvas);
      
      const blob = await canvasToBlob(enhancedCanvas, 'image/jpeg', 0.95);
      enhancedB64 = canvasToBase64(enhancedCanvas, 'image/jpeg', 0.95);
      processedFile = new File([blob], 'selected_enhanced.jpg', { type: 'image/jpeg' });
      
      console.log('✅ Orijinal enhance edildi:', processedFile.name);
      tabEnhanced.disabled = false;
      showTab('enhanced');
    }

    URL.revokeObjectURL(objectUrl);
  } catch (e) {
    console.error('❌ processSelectedFileWithJscanify exception:', e);
    enhancedB64 = null;
    processedFile = null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KAMERA
// ═��═════════════════════════════════════════════════════════════════════════════
function openChoiceMenu() { choiceSheet.classList.add('open'); }
function closeChoiceMenu() { choiceSheet.classList.remove('open'); }

function openCameraSheet() {
  closeChoiceMenu();
  cameraSheet.classList.add('open');
  cam.classList.add('active');
  startCamera();
}

async function closeCameraSheet() {
  cameraSheet.classList.remove('open');
  cam.classList.remove('active');
  await stopCamera();
}

async function startCamera() {
  try {
    await waitForOpenCvAndScanner();
    await stopCamera();

    sInfo.textContent = 'Kamera başlatılıyor...';

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    });

    video.srcObject = stream;
    await new Promise(resolve => {
      video.onloadedmetadata = async () => {
        await video.play();
        try { await tryAutoFocus(); } catch (_) {}
        resolve();
      };
    });

    initCameraOverlay();

    scanning = true;
    scanAttempts = 0;
    stableDetections = 0;
    scanInProgress = false;
    lastScanTime = 0;

    sInfo.textContent = 'Belgeyi kadraja getir';
    autoDetectDocument();
  } catch (e) {
    sInfo.textContent = 'Kamera hatası: ' + e.message;
  }
}

async function stopCamera() {
  scanning = false;
  scanAttempts = 0;
  stableDetections = 0;
  scanInProgress = false;

  destroyCameraOverlay();

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject = null;
}

async function autoDetectDocument() {
  if (!scanning || !stream) {
    console.log('⚠️ autoDetectDocument: scanning=', scanning, 'stream=', !!stream);
    return;
  }

  try {
    const now = Date.now();

    if (scanInProgress) {
      setTimeout(autoDetectDocument, 80);
      return;
    }

    if (now - lastScanTime < LIVE_SCAN_INTERVAL) {
      setTimeout(autoDetectDocument, 80);
      return;
    }

    if (!video.videoWidth || !video.videoHeight) {
      setTimeout(autoDetectDocument, 100);
      return;
    }

    scanInProgress = true;
    lastScanTime = now;

    const srcW = video.videoWidth;
    const srcH = video.videoHeight;

    const scale = Math.min(LIVE_W / srcW, LIVE_H / srcH);
    liveCanvas.width = Math.max(1, Math.round(srcW * scale));
    liveCanvas.height = Math.max(1, Math.round(srcH * scale));

    liveCtx.drawImage(video, 0, 0, liveCanvas.width, liveCanvas.height);

    const highlightedCanvas = tryHighlightPaper(liveCanvas);
    const extractedCanvas = tryExtractPaper(liveCanvas);

    if (highlightedCanvas && extractedCanvas) {
      drawHighlightOnOverlay(highlightedCanvas);
    } else {
      clearDetectCanvas();
    }

    if (extractedCanvas) {
      stableDetections++;
      console.log('📍 Belge bulundu! stableDetections:', stableDetections, '/', STABLE_FRAMES_REQUIRED);
      _setOverlayState('detecting');

      if (stableDetections >= STABLE_FRAMES_REQUIRED) {
        console.log('🎬 === STABILIZATION BAŞARILI === capturePhoto çağrılıyor');
        scanning = false;
        _setOverlayState('detected');
        _showCamToast('scan', 'Belge yakalanıyor', 'ok', 900);
        sInfo.textContent = 'Belge bulundu — çekiliyor...';

        await new Promise(r => setTimeout(r, 180));
        await capturePhoto();
        console.log('✅ capturePhoto tamamlandı');
        scanInProgress = false;
        return;
      }

      sInfo.textContent = `Belge bulundu — sabit tut ${stableDetections}/${STABLE_FRAMES_REQUIRED}`;
    } else {
      stableDetections = 0;
      scanAttempts++;
      _setOverlayState('idle');

      if (scanAttempts >= MAX_SCAN_ATTEMPTS) {
        console.log('⚠️ Belge bulunamadı (MAX_SCAN_ATTEMPTS:', MAX_SCAN_ATTEMPTS, ')');
        sInfo.textContent = 'Belge bulunamadı — Manuel Çek butonunu kullan';
        scanAttempts = 0;
      }
    }

    scanInProgress = false;
    setTimeout(autoDetectDocument, 100);
  } catch (e) {
    console.error('❌ autoDetectDocument exception:', e);
    scanInProgress = false;
    if (scanning) setTimeout(autoDetectDocument, 250);
  }
}

async function capturePhoto() {
  console.log('🎥 === capturePhoto başladı ===');
  
  if (!stream) {
    console.log('❌ stream yok');
    return;
  }

  console.log('✓ stream var, canvas oluşturuluyor...');
  const canvas = document.createElement('canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  console.log('✓ Canvas:', canvas.width, 'x', canvas.height);

  const originalBlob = await canvasToBlob(canvas, 'image/jpeg', 0.95);
  const originalFile = new File([originalBlob], 'camera.jpg', { type: 'image/jpeg' });

  console.log('✓ Dosya oluşturuldu:', originalFile.name, originalFile.size, 'bytes');

  drop._droppedFile = originalFile;
  setFile(originalFile, true);

  let scannedCanvas = null;
  try {
    console.log('🔍 tryExtractPaper çağrılıyor...');
    scannedCanvas = tryExtractPaper(canvas);
    console.log('📊 tryExtractPaper sonuç:', scannedCanvas ? '✅ başarılı' : '❌ null');
  } catch (e) {
    console.error('❌ tryExtractPaper exception:', e);
    scannedCanvas = null;
  }

  if (scannedCanvas) {
    console.log('📸 Crop başarılı! buildProcessedFileFromCanvas çağrılıyor (crop)');
    processedFile = await buildProcessedFileFromCanvas(scannedCanvas, 'camera_scanned_enhanced.jpg');
    console.log('✅ processedFile oluşturuldu (crop):', processedFile?.name);
    tabEnhanced.disabled = false;
    showTab('enhanced');
  } else {
    console.log('⚠️ Crop başarısız! buildProcessedFileFromCanvas çağrılıyor (orijinal)');
    processedFile = await buildProcessedFileFromCanvas(canvas, 'camera_enhanced.jpg');
    console.log('✅ processedFile oluşturuldu (orijinal):', processedFile?.name);
    tabEnhanced.disabled = false;
    showTab('enhanced');
  }

  console.log('🎬 Kamera kapatılıyor...');
  await closeCameraSheet();
  console.log('⏳ 300ms sonra runAnalyze çağrılacak');
  setTimeout(() => runAnalyze(), 300);
  console.log('🎥 === capturePhoto bitti ===\n');
}
// ── Analyze ───────────────────────────────────────────────────────────────────
async function runAnalyze() {
  err.style.display = 'none';
  bar.style.display = 'block';
  analyzeBtn.disabled = true;
  addPhotoBtn.disabled = true;
  clearBtn.disabled = true;

  const originalFile = drop._droppedFile || fileEl.files?.[0];
  const finalFile = processedFile || originalFile;

  if (!finalFile) {
    err.style.display = 'block';
    err.textContent = 'Lütfen bir görsel seç.';
    bar.style.display = 'none';
    analyzeBtn.disabled = false;
    addPhotoBtn.disabled = false;
    clearBtn.disabled = false;
    return;
  }

  ocrDot.className = 'dot loading';
  ocrStatus.textContent = 'OCR işleniyor...';

  try {
    const fd = new FormData();
    fd.append('file', finalFile);
    fd.append('auto_scan', 'false');
    fd.append('ocr_variant', 'enhanced');
    fd.append('ui_variant', 'enhanced');

    const res = await fetchWithTimeout('/analyze', { method: 'POST', body: fd }, OCR_TIMEOUT_MS);
    const data = await res.json();

    if (!res.ok) {
      ocrDot.className = 'dot off';
      ocrStatus.textContent = 'Hata';
      err.style.display = 'block';
      err.textContent = JSON.stringify(data, null, 2);
      return;
    }

    const abone = data.abone_bilgileri || {};
    const odeme = data.odeme || {};

    setBox('firma', data.firma_ismi || data.firma_adi || null);
    setBox('tur', data.fatura_turu || null);
    setBox('sozlesme', abone.sozlesme_no || data.sozlesme_no || null);
    setBox('tesisat', abone.tesisat_no || null);
    setBox('musteri', abone.musteri_no || data.musteri_no || null);
    setBox('fatura_no', abone.fatura_no || null);
    setBox('belge_no', abone.belge_no || null);
    setBox('son_odeme', odeme.son_odeme_tarihi || data.son_odeme_tarihi || null);
    setBox('tutar', odeme.tutar || data.tutar || null);
    setBox('ocr_time', formatDur(data.ocr_duration_seconds));
    setBox('document_id', data.document_id != null ? String(data.document_id) : null);

    ocrDot.className = 'dot';
    ocrStatus.textContent = 'Tamam ✓';

    if (enhancedB64) {
      tabEnhanced.disabled = false;
      showTab('enhanced');
    }

    out.textContent = JSON.stringify(data, null, 2);

    const rawText = data.ocr_raw_text || '';
    const rawSection  = document.getElementById('rawTextSection');
    const jsonSection = document.getElementById('jsonSection');

    if (rawText.trim()) {
      document.getElementById('rawText').textContent = rawText;
      rawSection.style.display = 'block';
    } else {
      rawSection.style.display = 'none';
    }

    jsonSection.style.display = 'block';
    debugWrap.style.display = 'block';
    copyBtn.disabled = false;

  } catch (e) {
    ocrDot.className = 'dot off';
    if (e.name === 'AbortError') {
      ocrStatus.textContent = 'Zaman aşımı';
      err.style.display = 'block';
      err.textContent = 'İstek zaman aşımına uğradı (90s). Sunucu meşgul olabilir, tekrar dene.';
    } else {
      ocrStatus.textContent = 'Hata';
      err.style.display = 'block';
      err.textContent = String(e);
    }
  } finally {
    bar.style.display = 'none';
    analyzeBtn.disabled = false;
    addPhotoBtn.disabled = false;
    clearBtn.disabled = false;
  }
}

// ── Events ────────────────────────────────────────────────────────────────────
addPhotoBtn.addEventListener('click', openChoiceMenu);
choiceClose.addEventListener('click', closeChoiceMenu);
choiceCamera.addEventListener('click', openCameraSheet);
choiceFile.addEventListener('click', () => { closeChoiceMenu(); fileEl.click(); });

sClose.addEventListener('click', closeCameraSheet);
cameraSheet.addEventListener('click', e => { if (e.target === cameraSheet) closeCameraSheet(); });
choiceSheet.addEventListener('click', e => { if (e.target === choiceSheet) closeChoiceMenu(); });

sSnap.addEventListener('click', capturePhoto);
analyzeBtn.addEventListener('click', runAnalyze);

fileEl.addEventListener('change', async () => {
  const f = fileEl.files?.[0];
  drop._droppedFile = f || null;
  processedFile = null;
  enhancedB64 = null;

  setFile(f, true);

  if (f) {
    await processSelectedFileWithJscanify(f);
    if (processedFile) showTab('enhanced');
  }
});

clearBtn.addEventListener('click', resetUI);

['dragenter','dragover'].forEach(ev => {
  drop.addEventListener(ev, e => {
    e.preventDefault();
    e.stopPropagation();
    drop.classList.add('drag');
  });
});

['dragleave','drop'].forEach(ev => {
  drop.addEventListener(ev, e => {
    e.preventDefault();
    e.stopPropagation();
    drop.classList.remove('drag');
  });
});

drop.addEventListener('drop', async e => {
  const f = e.dataTransfer.files?.[0];
  if (!f) return;

  if (!/\.(jpg|jpeg|png|webp)$/i.test(f.name)) {
    err.style.display = 'block';
    err.textContent = 'Desteklenmeyen dosya. JPG / PNG / WEBP yükle.';
    return;
  }

  drop._droppedFile = f;
  processedFile = null;
  enhancedB64 = null;

  setFile(f, true);
  await processSelectedFileWithJscanify(f);
  if (processedFile) showTab('enhanced');
});

copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(out.textContent);
    copyBtn.textContent = 'Kopyalandı ✓';
    setTimeout(() => { copyBtn.textContent = 'JSON Kopyala'; }, 1200);
  } catch (e) {
    err.style.display = 'block';
    err.textContent = 'Kopyalama başarısız: ' + String(e);
  }
});

tabOrig.addEventListener('click', () => showTab('orig'));
tabEnhanced.addEventListener('click', () => { if (!tabEnhanced.disabled) showTab('enhanced'); });

resetUI();
