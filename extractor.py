"""
Swatch Extractor — Core Logic
==============================
Pipeline:
  1. PyMuPDF  → render PDF page to image
  2. pdfplumber → extract text positions (section anchors + swatch labels)
  3. OpenCV HoughCircles → detect swatch circles in ROI
  4. Label matching → assign codes to circles
  5. If matched < expected threshold → Gemini Flash fallback
  6. PIL → crop + return individual swatch images
"""

import io
import os
import re
import json
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DPI = 250
SCALE = DPI / 72.0
PADDING = 8

# Words that appear on catalog pages but are NOT swatch labels
STOP_WORDS = {
    'Frame','Struttura','Seat','Sedile','Dimensions','Dimensioni',
    'metal','metallo','fabric','tessuto','faux','leather','similpelle',
    'Made','to','Order','Finishes','Finish','Options','Color','Colour',
    'Colors','Colours','Structure','Upholstery','Base','Top','Wood',
    'Lacquer','Stone','Glass','180','181','182','183','184',
}

# Section heading keywords to find ROI boundaries
SECTION_START_KEYWORDS = ['Frame','Struttura','Structure','Finish','Finishes','Base']
SECTION_END_KEYWORDS   = ['Dimensions','Dimensioni','Size','Sizes','Technical']

# Known y-band definitions (in PDF pts, relative to section start top)
# These are relative offsets — computed dynamically per page
# fallback if no section header found: scan full page
SWATCH_BANDS = {
    'frame':        (0,   80),   # within ~80pts of Frame header
    'fabric':       (100, 175),  # fabric section comes after
    'faux_leather': (175, 250),  # faux leather after fabric
}

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SwatchResult:
    label: str
    category: str
    image_bytes: bytes          # PNG bytes of cropped swatch
    image_b64: str = field(default="")
    width: int = 0
    height: int = 0
    confidence: str = "opencv"  # "opencv" | "gemini"

    def __post_init__(self):
        self.image_b64 = base64.b64encode(self.image_bytes).decode()
        img = Image.open(io.BytesIO(self.image_bytes))
        self.width, self.height = img.size


# ── Step 1: Render ────────────────────────────────────────────────────────────

def render_page(pdf_bytes: bytes, page_num: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (img_rgb_np, img_bgr_np) at DPI resolution."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np, img_bgr


# ── Step 2: Text extraction ────────────────────────────────────────────────────

def extract_text_layout(pdf_bytes: bytes, page_num: int = 0):
    """Returns (section_start_top, section_end_top, label_tokens)."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[page_num]
        words = page.extract_words()

    section_start = None
    section_end = None

    for w in words:
        if section_start is None and w['text'] in SECTION_START_KEYWORDS:
            section_start = w['top']
        if section_end is None and w['text'] in SECTION_END_KEYWORDS:
            if section_start and w['top'] > section_start:
                section_end = w['top']

    # Fallback: use 30% → 85% of page height
    if section_start is None:
        page_height = pdf.pages[page_num].height if hasattr(pdf, 'pages') else 800
        section_start = page_height * 0.30
    if section_end is None:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf2:
            section_end = pdf2.pages[page_num].height * 0.85

    label_tokens = [
        w for w in words
        if w['top'] > section_start
        and w['top'] < section_end
        and w['text'] not in STOP_WORDS
        and len(w['text']) >= 2
        and not w['text'].isdigit()
    ]

    return section_start, section_end, label_tokens


# ── Step 3 + 4: OpenCV detection + label matching ─────────────────────────────

def detect_and_match(
    img_bgr: np.ndarray,
    img_rgb: np.ndarray,
    section_start: float,
    section_end: float,
    label_tokens: list,
) -> list[SwatchResult]:

    ROI_y1 = max(0, int((section_start - 30) * SCALE))
    ROI_y2 = min(img_bgr.shape[0], int((section_end + 10) * SCALE))
    roi = img_bgr[ROI_y1:ROI_y2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=55,
        param1=55,
        param2=22,
        minRadius=25,
        maxRadius=50,
    )

    if circles is None:
        logger.warning("HoughCircles found no circles in ROI")
        return []

    c = np.round(circles[0]).astype(int)

    # Build dynamic y-bands relative to section_start
    band_width = (section_end - section_start) / 3
    bands = {
        'frame':        (section_start,              section_start + band_width),
        'fabric':       (section_start + band_width, section_start + band_width * 2),
        'faux_leather': (section_start + band_width * 2, section_end),
    }

    # Filter circles into bands
    valid = []
    for (cx, cy_roi, r) in c:
        cy_page = cy_roi + ROI_y1
        cy_pts = cy_page / SCALE
        for cat, (lo, hi) in bands.items():
            if lo < cy_pts < hi:
                valid.append((cx, cy_page, r, cat))
                break

    # Match to labels
    def find_label(cx_px, cy_px, used_ids):
        cx_pts, cy_pts = cx_px / SCALE, cy_px / SCALE
        best, best_d = None, float('inf')
        for w in label_tokens:
            if id(w) in used_ids:
                continue
            lx = (w['x0'] + w['x1']) / 2
            ly = w['top']
            if cy_pts < ly < cy_pts + 50:
                d = abs(lx - cx_pts)
                if d < best_d and d < 55:
                    best_d, best = d, w
        return best

    results = []
    used_ids = set()

    for (cx, cy_page, r, cat) in sorted(valid, key=lambda x: x[0]):
        token = find_label(cx, cy_page, used_ids)
        if not token:
            continue  # discard unmatched — likely noise/icons
        used_ids.add(id(token))

        x1 = max(0, cx - r - PADDING)
        y1 = max(0, cy_page - r - PADDING)
        x2 = min(img_rgb.shape[1], cx + r + PADDING)
        y2 = min(img_rgb.shape[0], cy_page + r + PADDING)

        crop_np = img_rgb[y1:y2, x1:x2]
        buf = io.BytesIO()
        Image.fromarray(crop_np).save(buf, format="PNG")

        results.append(SwatchResult(
            label=token['text'],
            category=cat,
            image_bytes=buf.getvalue(),
            confidence="opencv",
        ))

    logger.info(f"OpenCV matched {len(results)} swatches")
    return results


# ── Step 5: Gemini fallback ────────────────────────────────────────────────────

def gemini_fallback(png_bytes: bytes, gemini_key: str) -> list[dict]:
    """
    Use Gemini 2.0 Flash to detect swatch bboxes when OpenCV undershoots.
    Returns list of {label, category, x1, y1, x2, y2}.
    Free tier: 1500 req/day, 15 req/min.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        img = Image.open(io.BytesIO(png_bytes))
        response = model.generate_content([
            img,
            """This is a furniture catalog spec page. Find all color/material swatch circles.
Return ONLY a JSON array, no markdown:
[{"label":"P15","category":"frame","x1":100,"y1":200,"x2":155,"y2":255}]
category must be one of: frame, fabric, faux_leather, wood, lacquer, other."""
        ])

        raw = re.sub(r"```(?:json)?|```", "", response.text).strip()
        return json.loads(raw)

    except Exception as e:
        logger.error(f"Gemini fallback failed: {e}")
        return []


def run_gemini_fallback(
    pdf_bytes: bytes,
    img_rgb: np.ndarray,
    page_num: int,
    gemini_key: str,
) -> list[SwatchResult]:
    """Render page, call Gemini, crop results."""
    pix_buf = io.BytesIO()
    Image.fromarray(img_rgb).save(pix_buf, format="PNG")
    png_bytes = pix_buf.getvalue()

    detections = gemini_fallback(png_bytes, gemini_key)
    results = []

    for d in detections:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        # Add padding
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(img_rgb.shape[1], x2 + PADDING)
        y2 = min(img_rgb.shape[0], y2 + PADDING)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        crop_np = img_rgb[y1:y2, x1:x2]
        buf = io.BytesIO()
        Image.fromarray(crop_np).save(buf, format="PNG")

        results.append(SwatchResult(
            label=d.get('label', 'unknown'),
            category=d.get('category', 'other'),
            image_bytes=buf.getvalue(),
            confidence="gemini",
        ))

    logger.info(f"Gemini returned {len(results)} swatches")
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_swatches(
    pdf_bytes: bytes,
    page_num: int = 0,
    expected_min: int = 3,
    gemini_key: Optional[str] = None,
) -> list[SwatchResult]:
    """
    Full pipeline. Returns list of SwatchResult objects.

    Args:
        pdf_bytes:    Raw PDF file bytes
        page_num:     Which page to process (0-indexed)
        expected_min: If OpenCV finds fewer than this, trigger Gemini fallback
        gemini_key:   GEMINI_API_KEY (free at aistudio.google.com); None = no fallback
    """
    logger.info(f"Extracting swatches from page {page_num}")

    img_rgb, img_bgr = render_page(pdf_bytes, page_num)
    section_start, section_end, label_tokens = extract_text_layout(pdf_bytes, page_num)
    results = detect_and_match(img_bgr, img_rgb, section_start, section_end, label_tokens)

    # Trigger Gemini fallback if under threshold
    if len(results) < expected_min and gemini_key:
        logger.info(f"OpenCV got {len(results)} < {expected_min}, triggering Gemini fallback")
        results = run_gemini_fallback(pdf_bytes, img_rgb, page_num, gemini_key)

    return results
