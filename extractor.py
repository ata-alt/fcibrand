"""
Swatch Extractor
=================
Language-agnostic. Works for Italian, German, English, French catalogs.
Handles any page position — auto-detects swatch page via:
  - Rows of 3+ uniform circles (swatch grid fingerprint)
  - Unique product codes present (P15, GTB, SKZ etc.)
  Both signals must be strong — prevents caption pages from winning.

Pipeline:
  1. Auto-detect swatch page
  2. OpenCV HoughCircles on full-res render
  3. Label matching via is_swatch_code filter
  4. Gemini 2.0 Flash fallback if OpenCV undershoots
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
LOW     = 150 / 72.0   # low-res scale used for page scanning only


# ── Swatch code recogniser ────────────────────────────────────────────────────

_STOP = {
    # English common words that pass TitleCase filter
    'The','And','For','With','Its','Are','Was','Has','Had','Not',
    'Here','There','This','That','From','Also','Some','Such',
    'More','Each','Only','Both','Same','They','Their','Will','Been',
    # Italian/French/German common words
    'Qui','Con','Per','Nel','Una','Uno','Del','Dei','Delle','Che',
    'Tavolo','Tavola','Piano','Sedia','Sedie','Poltrona','Seduta',
    'Non','Solo','Piede','Piedi','Bronzo','Satin','Brass',
    'Legs','Canna','Please','You','Covering','Stitching',
    'Bett','Tisch','Stuhl','Sofa',
    # Furniture model/product names that match TitleCase but are NOT codes
    'Flair','Vully','Atlas','Gemini','Tuka','Etienne','Mid',
    'Semi','Easy','Cross','Lama','Nova','Luna','Star',
    # Section headings
    'Frame','Struttura','Gestell','Structure','Base','Basamento',
    'Seat','Sedile','Sitz','Assise','Top',
    'Dimensions','Dimensioni','Abmessungen',
    'Metal','Metallo','Fabric','Tessuto','Stoff','Tissu',
    'Leather','Similpelle','Kunstleder','Cuir',
    'Wood','Legno','Holz','Bois','Glass','Vetro',
    'Stone','Pietra','Ceramic','Ceramica',
    'Made','Order','Note','Notes','View',
    # Geographic / company noise
    'ITALY','NC','BA','CA','NY','UK','USA',
    'Road','House','City','State','Model','Phone',
    # Technical drawing / schematic labels
    'FRONT','DEPTH','FRONTALE','LATERALE',
    'Legend','Legenda','Ratio','Scala','Scale',
    'POUF','Pouf','Back','Front','Depth',
}

# Typographic + ASCII quote chars used for inch measurements
_QUOTE_CHARS = {'"', '\u201c', '\u201d', '\u2019', '\u2018'}

def _is_swatch_code(text: str) -> bool:
    """True if text looks like a finish/material code rather than prose."""
    t = text.strip().rstrip('.')
    if len(t) < 2 or len(t) > 8:           return False
    if t in _STOP:                          return False
    if not re.match(r'^[A-Za-z0-9.]+$', t): return False   # must be alphanumeric
    if any(c in _QUOTE_CHARS for c in t):   return False   # inch dims: 37”
    if ',' in t or ':' in t or '%' in t:   return False
    if '.' in t:                                            # allow Cod.XX only
        if re.match(r'^[A-Za-z]{2,4}\.\d{1,2}$', t): pass
        else: return False
    if re.match(r'^CB\d+', t):             return False   # SKUs: CB2348
    if re.match(r'^[HS][HX]?\d+', t):     return False   # H97, SH65
    if re.match(r'^\d+$', t):             return False   # page numbers
    if re.search(r'\d', t):               return True    # P15, T3H, Cod.50
    if t.isupper() and 2 <= len(t) <= 4:  return True    # GTG, SKZ (not ITALY)
    if t[0].isupper() and t[1:].islower() and 3 <= len(t) <= 6:
        return True                                        # Cros, Harry
    return False


def _is_scoring_code(text: str) -> bool:
    """
    Stricter version used only for PAGE SCORING.
    Requires digit in code OR ≤4 char all-uppercase.
    Filters out place/company names (Parchi, Calia, Inc) and prose (Cros, Harry).
    """
    if not _is_swatch_code(text): return False
    t = text.strip().rstrip('.')
    if re.search(r'\d', t):               return True   # P15, T3H, Cod.50
    if t.isupper() and len(t) <= 4:       return True   # GTG, SKZ (not ITALY=5)
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


# ── Page scoring ──────────────────────────────────────────────────────────────

def _count_swatch_rows(circles_np) -> int:
    """Count rows of 3+ circles — the fingerprint of a swatch grid."""
    if circles_np is None or len(circles_np) == 0:
        return 0
    c = sorted(circles_np.tolist(), key=lambda x: x[1])
    rows, cur = [], [c[0]]
    for ci in c[1:]:
        if ci[1] - cur[-1][1] > 25:
            rows.append(cur); cur = [ci]
        else:
            cur.append(ci)
    rows.append(cur)
    return len([r for r in rows if len(r) >= 3])


def _score_page(pdf_bytes: bytes, page_num: int) -> float:
    """
    Score page for swatch likelihood.
    Both signals required: circle rows AND unique product codes.
    Caption pages have codes but few circle rows → score lower.
    Photo pages have circle rows but no codes → score lower.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(LOW, LOW))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20, param1=55, param2=20, minRadius=10, maxRadius=35
    )

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        words = pdf.pages[page_num].extract_words()

    total_words = len(words)
    if total_words < 8:
        return -999.0   # full-bleed photo page

    unique_codes = list(set(w['text'] for w in words if _is_scoring_code(w['text'])))
    if len(unique_codes) < 3:
        return -999.0   # need ≥3 distinct codes — cover/address pages disqualified

    swatch_rows  = _count_swatch_rows(circles[0] if circles is not None else None)
    row_bonus    = min(swatch_rows, 6) * 25    # cap benefit at 6 rows
    row_penalty  = max(0, swatch_rows - 6) * 15  # penalise schematics (10-15+ rows)
    score = float(row_bonus - row_penalty + len(unique_codes) * 15)

    return score


# Minimum score for a page to be considered a swatch page.
# Pages below this are product photography or pure text — skip extraction entirely.
MIN_SWATCH_SCORE = 50


class NoSwatchPageError(ValueError):
    """Raised when no page in the PDF looks like a swatch/spec page."""
    pass


def find_swatch_page(pdf_bytes: bytes, gemini_key: str | None = None) -> int:
    """
    Scan all pages and return the one most likely to contain swatches.
    Raises NoSwatchPageError if no page scores above MIN_SWATCH_SCORE.
    """
    n = len(fitz.open(stream=pdf_bytes, filetype="pdf"))
    logger.info(f"Scanning {n} pages for swatch page...")

    # Stage 1: OpenCV scores every page (free, eliminates obvious non-candidates)
    scores = []
    for i in range(n):
        s = _score_page(pdf_bytes, i)
        scores.append((s, i))
        logger.info(f"  Page {i}: score={s:.0f}")

    best_score, best_page = max(scores)

    # Stage 2: Filter to candidates above minimum threshold
    candidates = [(s, i) for s, i in scores if s >= MIN_SWATCH_SCORE]
    if not candidates:
        raise NoSwatchPageError(
            f"No swatch spec page found in this PDF "
            f"(best page {best_page} scored {best_score:.0f}, minimum is {MIN_SWATCH_SCORE}). "
            f"This PDF appears to contain only product photography or text pages."
        )

    # Stage 3: Decision
    if len(candidates) == 1:
        # Only one candidate — no ambiguity, skip Gemini
        winner = candidates[0][1]
        logger.info(f"Swatch page: {winner} (single candidate, score={candidates[0][0]:.0f})")
        return winner

    # Multiple candidates: use Gemini to make the final visual call
    if gemini_key:
        logger.info(
            f"{len(candidates)} candidates above threshold — asking Gemini to pick "
            f"(pages {[i for _, i in sorted(candidates, key=lambda x: -x[0])[:3]]})"
        )
        gemini_choice = _gemini_pick_page(pdf_bytes, candidates, gemini_key)
        if gemini_choice is not None:
            logger.info(f"Swatch page: {gemini_choice} (Gemini + OpenCV hybrid)")
            return gemini_choice
        logger.info("Gemini pick failed — falling back to OpenCV best score")

    # No Gemini key or Gemini failed: use highest OpenCV score
    winner = best_page
    logger.info(f"Swatch page: {winner} (OpenCV score={best_score:.0f})")
    return winner


# ── Render ────────────────────────────────────────────────────────────────────

def _render(pdf_bytes: bytes, page_num: int):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Category assignment ───────────────────────────────────────────────────────

_CAT_KW = {
    'frame':        ['frame','struttura','gestell','structure','base','basamento',
                     'metal','metallo','stahl','acier','acero'],
    'fabric':       ['fabric','tessuto','stoff','tissu','tejido','upholstery'],
    'faux_leather': ['faux','leather','similpelle','kunstleder','simili','cuir',
                     'ecopelle','cuero'],
    'wood':         ['wood','legno','holz','bois','madera','oak','walnut','noce'],
    'lacquer':      ['lacquer','lacca','lack','laque','laca','laccato'],
    'ceramic':      ['ceramic','ceramica','keramik','céramique'],
    'glass':        ['glass','vetro','glas','verre','vidrio'],
    'glass_stone':  ['stone','pietra','stein','pierre'],
}

def _category(cy_px: float, words: list, roi_y1: int, roi_y2: int) -> str:
    cy_pts = cy_px / SCALE
    above  = [w for w in words if w['top'] < cy_pts and cy_pts - w['top'] < 130]
    for w in sorted(above, key=lambda x: cy_pts - x['top']):
        tok = w['text'].lower()
        for cat, kws in _CAT_KW.items():
            if any(kw in tok for kw in kws):
                return cat
    span = max(roi_y2 - roi_y1, 1)
    rel  = (cy_px - roi_y1) / span
    if rel < 0.33:   return 'frame'
    elif rel < 0.66: return 'fabric'
    return 'faux_leather'


# ── OpenCV extraction ─────────────────────────────────────────────────────────

def _opencv_extract(img_rgb, img_bgr, words, gemini_labels=None) -> list:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=55, param1=55, param2=22, minRadius=25, maxRadius=50
    )
    if circles is None:
        logger.warning("No circles found")
        return []

    # Use Gemini-identified labels if available, otherwise fall back to regex
    if gemini_labels is not None:
        label_tokens = [w for w in words if w['text'] in gemini_labels]
        logger.info(f"Swatch label candidates (Gemini): {[w['text'] for w in label_tokens]}")
    else:
        label_tokens = [w for w in words if _is_swatch_code(w['text'])]
        logger.info(f"Swatch label candidates (regex): {[w['text'] for w in label_tokens]}")

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

    results     = []
    used_ids    = set()
    matched_ys  = []

    for (cx, cy, r) in sorted(np.round(circles[0]).astype(int), key=lambda x: x[0]):
        # ── Reject text/noise circles ──────────────────────────
        # Real swatches: low white% OR uniform color (low std)
        # Text circles: high white% AND high contrast (dark lines on white paper)
        x1c = max(0,                int(cx) - r - 4)
        y1c = max(0,                int(cy) - r - 4)
        x2c = min(img_rgb.shape[1], int(cx) + r + 4)
        y2c = min(img_rgb.shape[0], int(cy) + r + 4)
        crop_g    = np.array(Image.fromarray(img_rgb[y1c:y2c, x1c:x2c]).convert('L'))
        white_pct = float((crop_g > 220).sum()) / max(crop_g.size, 1)
        std_val   = float(crop_g.std())
        if white_pct > 0.65 and std_val > 42:
            continue   # text on white paper, not a color swatch

        token = find_label(int(cx), int(cy), used_ids)
        if not token:
            continue
        used_ids.add(id(token))
        matched_ys.append(int(cy))

        roi_y1 = min(matched_ys) - 100
        roi_y2 = max(matched_ys) + 100
        cat    = _category(int(cy), words, roi_y1, roi_y2)

        x1 = max(0,                int(cx) - r - PADDING)
        y1 = max(0,                int(cy) - r - PADDING)
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


# ── Gemini label assist ───────────────────────────────────────────────────────
#
# Instead of asking Gemini for pixel bounding boxes (spatial reasoning = weak),
# we ask it to READ THE TEXT CODES off the page (text reading = strong).
# OpenCV still does all the spatial/circle detection work.
#
# With Gemini key:   label_tokens = Gemini-identified codes
# Without Gemini key: label_tokens = _is_swatch_code() regex (current behaviour)
#
# This replaces the old "fallback" concept entirely.

def _gemini_read_labels(img_rgb: np.ndarray, gemini_key: str) -> list[str] | None:
    """
    Ask Gemini to identify swatch code labels visible on the page.
    Returns a list of code strings, or None if the call fails.

    This is a TEXT READING task, not a spatial/coordinate task.
    Much more reliable than asking Gemini for bounding boxes.
    Gemini 2.0 Flash is sufficient — Pro not needed for this.
    """
    try:
        from google import genai as ggenai
        client = ggenai.Client(api_key=gemini_key)

        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="PNG")
        image_bytes = buf.getvalue()

        prompt = (
            "This is a furniture catalog spec page. "
            "Find ALL finish/material/color codes shown as labels near swatch samples.\n"
            "These are short product codes like: P15, P38M, SKZ, SLA, GTG, Cod.50, Cod.03, "
            "T3H, P2C, GMA, MTO etc.\n"
            "Return ONLY a JSON array of the exact codes you can read. No explanation, no markdown.\n"
            "Example: [\"P15\", \"P151\", \"P38M\", \"SKZ\", \"Cod.50\"]"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                ggenai.types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )

        raw    = re.sub(r"```(?:json)?|```", "", response.text).strip()
        labels = json.loads(raw)

        if isinstance(labels, list) and all(isinstance(lb, str) for lb in labels):
            # Deduplicate preserving order
            seen, unique = set(), []
            for lb in labels:
                if lb not in seen:
                    seen.add(lb)
                    unique.append(lb)
            logger.info(f"Gemini identified {len(unique)} labels: {unique}")
            return unique

        logger.warning("Gemini returned unexpected format — falling back to regex")
        return None

    except Exception as e:
        logger.warning(f"Gemini label read failed ({e}) — using regex labels")
        return None


# ── Gemini page picker ────────────────────────────────────────────────────────

def _gemini_pick_page(
    pdf_bytes: bytes,
    candidates: list[tuple[float, int]],   # [(score, page_num), ...]
    gemini_key: str,
) -> int | None:
    """
    Send up to 3 candidate pages to Gemini as images in one API call.
    Ask: "which page is the material/finish swatch spec page?"
    Returns the winning page_num, or None if the call fails.

    This is a VISUAL UNDERSTANDING question — Gemini is reliable for this.
    One API call regardless of how many candidates.
    """
    try:
        from google import genai as ggenai
        client = ggenai.Client(api_key=gemini_key)

        # Sort by OpenCV score descending, take top 3
        top = sorted(candidates, key=lambda x: -x[0])[:3]
        page_nums = [pn for _, pn in top]

        # Render each candidate at low resolution
        parts = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, pn in enumerate(page_nums, start=1):
            pix = doc[pn].get_pixmap(matrix=fitz.Matrix(LOW, LOW))
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            buf = io.BytesIO()
            Image.fromarray(img_np).save(buf, format="PNG")
            parts.append(ggenai.types.Part.from_bytes(
                data=buf.getvalue(), mime_type="image/png"
            ))
            parts.append(f"(Image {i} = catalog page {pn})")

        prompt = (
            f"I am showing you {len(page_nums)} pages from a furniture catalog.\n"
            "One of them is the material/finish options spec page that shows:\n"
            "  - Color/finish sample swatches (circles or squares)\n"
            "  - Short product codes like P15, SKZ, Cod.50, GTG, T3H\n"
            "  - Section labels like Frame, Fabric, Leather, Ceramic, Glass\n\n"
            f"Which image number (1 to {len(page_nums)}) is the swatch spec page?\n"
            "If none of them are a swatch spec page, answer 0.\n"
            "Answer with a single digit only — nothing else."
        )
        parts.append(prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=parts,
        )

        answer = response.text.strip()
        # Extract the first digit from the answer
        digit_match = re.search(r"\d", answer)
        if not digit_match:
            logger.warning(f"Gemini page pick: unexpected answer '{answer}'")
            return None

        choice = int(digit_match.group())
        if choice == 0:
            logger.info("Gemini page pick: answered 0 (no swatch page)")
            return None
        if choice < 1 or choice > len(page_nums):
            logger.warning(f"Gemini page pick: out-of-range answer '{choice}'")
            return None

        selected = page_nums[choice - 1]
        logger.info(
            f"Gemini page pick: chose image {choice} = catalog page {selected} "
            f"(from candidates {page_nums})"
        )
        return selected

    except Exception as e:
        logger.warning(f"Gemini page pick failed ({e}) — using OpenCV best score")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_swatches(
    pdf_bytes:    bytes,
    page_num:     Optional[int] = None,
    expected_min: int  = 3,
    gemini_key:   Optional[str] = None,
) -> list:
    """
    Extract swatch images from any furniture catalog PDF.

    Args:
        pdf_bytes:    Raw PDF bytes
        page_num:     Explicit 0-indexed page, or None for auto-detect
        expected_min: (unused — kept for API compatibility)
        gemini_key:   Free from aistudio.google.com
    """
    if page_num is None:
        page_num = find_swatch_page(pdf_bytes, gemini_key=gemini_key)
    else:
        logger.info(f"Using explicit page: {page_num}")

    img_rgb, img_bgr = _render(pdf_bytes, page_num)
    logger.info(f"Page {page_num} rendered: {img_bgr.shape[1]}x{img_bgr.shape[0]}px")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        words = pdf.pages[page_num].extract_words()
    logger.info(f"Text tokens: {len(words)}")

    # If Gemini key provided: ask Gemini to READ the label codes (text reading task).
    # OpenCV still handles all circle detection and pixel coordinates.
    # This is far more reliable than asking Gemini for bounding boxes.
    gemini_labels = None
    if gemini_key:
        gemini_labels = _gemini_read_labels(img_rgb, gemini_key)

    results = _opencv_extract(img_rgb, img_bgr, words, gemini_labels=gemini_labels)
    return results
