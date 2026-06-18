import io
import base64
import logging
import requests
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(
    source_url:  str,
    target_size: int   = 1000,
    crop_bbox:   tuple = None,
    remove_bg:   bool  = False,
    padding:     int   = 20,
    sharpen:     bool  = True
) -> str:
    """
    FCI product image processing pipeline.
    Returns base64 encoded JPEG string.
    """

    # ── Step 1: Download source URL ──────────────────────────
    logger.info(f"Downloading image from: {source_url}")
    response = requests.get(source_url, timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    logger.info(f"Downloaded — original size: {img.size}, mode: {img.mode}")

    # ── Step 2: Open with Pillow ──────────────────────────────
    img = img.convert("RGBA")

    # ── Step 3: Optional crop to bbox ────────────────────────
    if crop_bbox:
        x1, y1, x2, y2 = crop_bbox
        img = img.crop((x1, y1, x2, y2))
        logger.info(f"Cropped to bbox: {crop_bbox} — new size: {img.size}")

    # ── Step 4: Optional rembg background removal ─────────────
    if remove_bg:
        try:
            from rembg import remove
            logger.info("Removing background with rembg...")
            img = remove(img)
            logger.info("Background removed")
        except ImportError:
            raise Exception("rembg not installed — add it to requirements.txt")

    # ── Step 5: Resize with LANCZOS ───────────────────────────
    usable = target_size - (padding * 2)
    orig_w, orig_h = img.size
    ratio = min(usable / orig_w, usable / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    logger.info(f"Resized to: {new_w}x{new_h} (ratio: {ratio:.3f})")

    # ── Step 6: Sharpen after resize ──────────────────────────
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
        logger.info("Sharpening applied")

    # ── Step 7: Paste on white canvas with padding ────────────
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2

    if img.mode == "RGBA":
        canvas.paste(img, (offset_x, offset_y), mask=img.split()[3])
    else:
        img = img.convert("RGB")
        canvas.paste(img, (offset_x, offset_y))

    logger.info(f"Pasted on {target_size}x{target_size} white canvas")

    # ── Step 8: Save as JPEG q90, 300 DPI ─────────────────────
    output_buffer = io.BytesIO()
    canvas.save(
        output_buffer,
        format="JPEG",
        quality=90,
        dpi=(300, 300),
        optimize=True
    )
    output_buffer.seek(0)

    encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    size_kb = len(output_buffer.getvalue()) / 1024
    logger.info(f"Done — output size: {size_kb:.1f} KB")

    # ── Step 9: Return base64 ─────────────────────────────────
    return encoded
