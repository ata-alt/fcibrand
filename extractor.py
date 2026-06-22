"""
Swatch Extractor
=================
Fully language-agnostic — works for Italian, German, English, French catalogs.
No hardcoded section heading keywords anywhere.

Pipeline:
  1. Auto-detect swatch page (scores every page — no page=0 assumption)
  2. OpenCV HoughCircles with is_swatch_code label filter
  3. Gemini 2.0 Flash fallback if OpenCV undershoots threshold
"""

import io, re, json, base64, logging
from dataclasses import dataclass, field
from typing import Optional

import cv2, fitz, numpy as np, pdfplumber
from PIL import Image

try:
    from logger_config import setup_logger
    logger = setup_logger("extractor")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extractor")

DPI     = 250
SCALE   = DPI / 72.0
PADDING = 8

# ── Swatch code recogniser ────────────────────────────────────────────────────
# Language-agnostic: identifies product codes by shape, not by catalogue keywords.

_STOP = {
    'Made','Order','Being','Note','View','More',
    'Frame','Struttura','Gestell','Structure',
    'Seat','Sedile','Sitz','Assise',
    'Dimensions','Dimensioni','Abmessungen',
    'Metal','Metallo','Fabric','Tessuto','Stoff','Tissu',
    'Leather','Similpelle','Kunstleder','Cuir',
    'Wood','Legno','Holz','Bois',
    'Base','Top','Sedia','Poltr','Tavol','Stuhl','Tisch',
    'Sofa','Bett','Chaise','Table',
}

def _is_swatch_code(text: str) -> bool:
    """True if text looks like a swatch/finish code rather than prose."""
    t = text.strip().rstrip('.')
    if len(t) < 2 or len(t) > 8:           return False
    if t in _STOP:                          return False
    if ',' in t or '.' in t:               return False   # dimension values: H78,5
    if re.match(r'^CB\d+', t):             return False   # SKU codes: CB2348
    if re.match(r'^[HS][HX]?\d+', t):     return False   # H97, SH65
    if re.match(r'^\d+$', t):             return False   # page numbers: 180
    if re.search(r'\d', t):               return True    # P15, T3H, P38M, P151
    if t.isupper() and 2 <= len(t) <= 5:  return True    # SKZ, SLA, SLB
    if t[0].isupper() and t[1:].islower() and 3 <= len(t) <= 6:
        return True                                       # Cros, Harry, Vero
    return False


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class SwatchResult:
    label:       str
    category:    str
    image_bytes: bytes
    image_b64:   str = field(default="")
    width:       int = 0
    height:      int = 0
    confidence:  str = "opencv"

    def __post_init__(self):
        self.image_b64 = base64.b64encode(self.image_bytes).decode()
        img = Image.open(io.BytesIO(self.image_bytes))
        self.width, self.height = img.size


# ── Page auto-detection ───────────────────────────────────────────────────────

def _score_page(pdf_bytes: bytes, page_num: int) -> float:
    """
    Score page for swatch likelihood.
    Signal: consistent-radius circles + short product codes.
    No language keywords used.
    """
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix  = doc[page_num].get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20, param1=55, param2=20, minRadius=10, maxRadius=35
    )

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        words = pdf.pages[page_num].extract_words()

    total_words   = len(words)
    swatch_tokens = sum(1 for w in words if _is_swatch_code(w['text']))

    # Hard-disqualify full-bleed photo pages
    if total_words < 8:
        return -999.0

    score = float(swatch_tokens * 20)  # strongest signal

    if circles is not None:
        radii     = circles[0][:, 2]
        r_std     = float(np.std(radii))
        n         = len(circles[0])
        # Low radius variance = swatch grid; high variance = photo
        if r_std < 3:   score += n * 5
        elif r_std < 6: score += n * 2
        else:           score -= n * 1.0

    if total_words > 300:
        score -= 30

    return score


def find_swatch_page(pdf_bytes: bytes) -> int:
    """Return page number most likely to contain swatches."""
    n = len(fitz.open(stream=pdf_bytes, filetype="pdf"))
    logger.info(f"Scanning {n} pages for swatch page...")
    scores = []
    for i in range(n):
        s = _score_page(pdf_bytes, i)
        scores.append((s, i))
        logger.info(f"  Page {i}: score={s:.0f}")
    best_score, best_page = max(scores)
    logger.info(f"Swatch page: {best_page} (score={best_score:.0f})")
    return best_page


# ── Render ────────────────────────────────────────────────────────────────────

def _render(pdf_bytes: bytes, page_num: int):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Category from heading above circle ───────────────────────────────────────

_CAT_KW = {
    'frame':        ['frame','struttura','gestell','structure','base','metal',
                     'metallo','stahl','acier','acero'],
    'fabric':       ['fabric','tessuto','stoff','tissu','tejido','upholstery'],
    'faux_leather': ['faux','leather','similpelle','kunstleder','simili','cuir',
                     'ecopelle','cuero'],
    'wood':         ['wood','legno','holz','bois','madera','oak','walnut','noce'],
    'lacquer':      ['lacquer','lacca','lack','laque','laca','laccato'],
}

def _category(cy_px: float, words: list, roi_y1: int, roi_y2: int) -> str:
    cy_pts = cy_px / SCALE
    above  = [w for w in words if w['top'] < cy_pts and cy_pts - w['top'] < 130]
    for w in sorted(above, key=lambda x: cy_pts - x['top']):
        tok = w['text'].lower()
        for cat, kws in _CAT_KW.items():
            if any(kw in tok for kw in kws):
                return cat
    # Positional fallback: thirds of ROI
    span = max(roi_y2 - roi_y1, 1)
    rel  = (cy_px - roi_y1) / span
    if rel < 0.33:   return 'frame'
    elif rel < 0.66: return 'fabric'
    return 'faux_leather'


# ── Main extraction ───────────────────────────────────────────────────────────

def _opencv_extract(img_rgb, img_bgr, words) -> list[SwatchResult]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=55, param1=55, param2=22, minRadius=25, maxRadius=50
    )
    if circles is None:
        logger.warning("No circles found")
        return []

    label_tokens = [w for w in words if _is_swatch_code(w['text'])]
    logger.info(f"Swatch label candidates: {[w['text'] for w in label_tokens]}")

    # Derive ROI from circles that actually have labels
    # (avoids committing to a y-range before we know which circles are real)
    def find_label(cx, cy, used):
        cx_pts, cy_pts = cx / SCALE, cy / SCALE
        best, bd = None, float('inf')
        for w in label_tokens:
            if id(w) in used: continue
            lx = (w['x0'] + w['x1']) / 2
            ly = w['top']
            if cy_pts < ly < cy_pts + 55:
                d = abs(lx - cx_pts)
                if d < bd and d < 55:
                    bd, best = d, w
        return best

    results  = []
    used_ids = set()
    matched_ys = []

    for (cx, cy, r) in sorted(np.round(circles[0]).astype(int), key=lambda x: x[0]):
        token = find_label(int(cx), int(cy), used_ids)
        if not token:
            continue
        used_ids.add(id(token))
        matched_ys.append(int(cy))

        roi_y1 = min(matched_ys) - 100 if matched_ys else 0
        roi_y2 = max(matched_ys) + 100 if matched_ys else img_bgr.shape[0]
        cat = _category(int(cy), words, roi_y1, roi_y2)

        x1 = max(0,               int(cx) - r - PADDING)
        y1 = max(0,               int(cy) - r - PADDING)
        x2 = min(img_rgb.shape[1], int(cx) + r + PADDING)
        y2 = min(img_rgb.shape[0], int(cy) + r + PADDING)

        buf = io.BytesIO()
        Image.fromarray(img_rgb[y1:y2, x1:x2]).save(buf, format="PNG")

        results.append(SwatchResult(
            label=token['text'], category=cat,
            image_bytes=buf.getvalue(), confidence="opencv",
        ))

    logger.info(f"OpenCV matched {len(results)} swatches")
    return results


# ── Gemini fallback ───────────────────────────────────────────────────────────

def _gemini_extract(img_rgb: np.ndarray, gemini_key: str) -> list[SwatchResult]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="PNG")
        response = model.generate_content([
            Image.open(io.BytesIO(buf.getvalue())),
            "Find all color/material swatch circles on this furniture catalog page. "
            "Return ONLY JSON array, no markdown:\n"
            '[{"label":"P15","category":"frame","x1":100,"y1":200,"x2":155,"y2":255}]\n'
            "category: frame | fabric | faux_leather | wood | lacquer | other"
        ])
        raw        = re.sub(r"```(?:json)?|```", "", response.text).strip()
        detections = json.loads(raw)

        results = []
        for d in detections:
            x1 = max(0, d['x1'] - PADDING);  y1 = max(0, d['y1'] - PADDING)
            x2 = min(img_rgb.shape[1], d['x2'] + PADDING)
            y2 = min(img_rgb.shape[0], d['y2'] + PADDING)
            if (x2-x1) < 10 or (y2-y1) < 10: continue
            b = io.BytesIO()
            Image.fromarray(img_rgb[y1:y2, x1:x2]).save(b, format="PNG")
            results.append(SwatchResult(
                label=d.get('label','unknown'), category=d.get('category','other'),
                image_bytes=b.getvalue(), confidence="gemini",
            ))
        logger.info(f"Gemini returned {len(results)} swatches")
        return results
    except Exception as e:
        logger.error(f"Gemini fallback failed: {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def extract_swatches(
    pdf_bytes:    bytes,
    page_num:     Optional[int] = None,
    expected_min: int  = 3,
    gemini_key:   Optional[str] = None,
) -> list[SwatchResult]:
    """
    Extract swatch images from any furniture catalog PDF.

    Args:
        pdf_bytes:    Raw PDF bytes
        page_num:     Explicit page (0-indexed). None = auto-detect.
        expected_min: Trigger Gemini fallback if OpenCV finds fewer.
        gemini_key:   Free from aistudio.google.com
    """
    if page_num is None:
        page_num = find_swatch_page(pdf_bytes)
    else:
        logger.info(f"Using explicit page: {page_num}")

    img_rgb, img_bgr = _render(pdf_bytes, page_num)
    logger.info(f"Page {page_num} rendered: {img_bgr.shape[1]}x{img_bgr.shape[0]}px")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        words = pdf.pages[page_num].extract_words()
    logger.info(f"Text tokens: {len(words)}")

    results = _opencv_extract(img_rgb, img_bgr, words)

    if len(results) < expected_min and gemini_key:
        logger.info(f"OpenCV got {len(results)} < {expected_min} — Gemini fallback")
        results = _gemini_extract(img_rgb, gemini_key)

    return results
