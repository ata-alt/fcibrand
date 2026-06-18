import io
import base64
from PIL import Image, ImageFilter, ImageChops
from logger_config import setup_logger

logger = setup_logger("processor")


def trim_whitespace(img: Image.Image, threshold: int = 240) -> Image.Image:
    """
    Trims existing white/near-white borders from image.
    threshold: how aggressively to trim
      255 = pure white only
      240 = near-white too (recommended for product images)
      220 = more aggressive
    """
    rgb  = img.convert("RGB")
    bg   = Image.new("RGB", rgb.size, (threshold, threshold, threshold))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()

    if bbox:
        logger.info(f"  Trimmed whitespace | bbox: {bbox} | "
                    f"before: {img.size} | "
                    f"after: {(bbox[2]-bbox[0], bbox[3]-bbox[1])}")
        return img.crop(bbox)
    else:
        logger.info("  No whitespace to trim — image is all white or uniform")
        return img


def get_orientation(w: int, h: int) -> str:
    """
    Returns orientation based on aspect ratio.
      landscape  → width > height
      portrait   → height > width
      square     → width == height
    """
    if w > h:
        return "landscape"
    elif h > w:
        return "portrait"
    else:
        return "square"


def get_padding(w: int, h: int,
                landscape_padding: int,
                portrait_padding:  int,
                square_padding:    int) -> tuple:
    """
    Returns (padding, orientation) based on image dimensions.
    """
    orientation = get_orientation(w, h)
    if orientation == "landscape":
        return landscape_padding, orientation
    elif orientation == "portrait":
        return portrait_padding, orientation
    else:
        return square_padding, orientation


def process_image(
    image_input:        str,
    target_size:        int  = 1200,
    landscape_padding:  int  = 6,    # ← tight padding for landscape
    portrait_padding:   int  = 10,   # ← slightly more for portrait
    square_padding:     int  = 10,   # ← same as portrait for square
    sharpen:            bool = True,
    trim_white:         bool = True
) -> str:
    """
    FCI product image processing pipeline.
    Accepts base64 encoded image string.
    Returns base64 encoded JPEG string.
    Padding is applied dynamically based on image orientation.
    """

    logger.info("=" * 60)
    logger.info("NEW IMAGE PROCESSING REQUEST STARTED")
    logger.info("=" * 60)
    logger.debug(f"Params — target_size: {target_size}, "
                 f"landscape_padding: {landscape_padding}, "
                 f"portrait_padding: {portrait_padding}, "
                 f"square_padding: {square_padding}, "
                 f"sharpen: {sharpen}, trim_white: {trim_white}")

    # ── Step 1: Decode base64 ─────────────────────────────────
    logger.info("STEP 1 — Decoding base64 image...")
    try:
        image_bytes = base64.b64decode(image_input)
        img         = Image.open(io.BytesIO(image_bytes))
        logger.info(f"STEP 1 — OK | size: {img.size} | mode: {img.mode} | "
                    f"input: {len(image_bytes)/1024:.1f} KB")
    except Exception as e:
        logger.error(f"STEP 1 — FAILED | {str(e)}")
        raise Exception(f"Invalid base64 image: {str(e)}")

    # ── Step 2: Convert to RGBA ───────────────────────────────
    logger.info("STEP 2 — Converting to RGBA...")
    try:
        img = img.convert("RGBA")
        logger.info(f"STEP 2 — OK | mode: {img.mode}")
    except Exception as e:
        logger.error(f"STEP 2 — FAILED | {str(e)}")
        raise

    # ── Step 3: Auto-trim existing white space ────────────────
    if trim_white:
        logger.info("STEP 3 — Trimming existing whitespace...")
        try:
            size_before = img.size
            img         = trim_whitespace(img, threshold=240)
            size_after  = img.size
            logger.info(f"STEP 3 — OK | {size_before} → {size_after}")
        except Exception as e:
            logger.error(f"STEP 3 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 3 — SKIPPED | trim_white is false")

    # ── Step 4: Detect orientation + apply dynamic padding ────
    logger.info("STEP 4 — Detecting orientation...")
    try:
        trimmed_w, trimmed_h = img.size
        padding, orientation = get_padding(
            trimmed_w, trimmed_h,
            landscape_padding,
            portrait_padding,
            square_padding
        )
        logger.info(f"STEP 4 — OK | orientation: {orientation} | "
                    f"size: {trimmed_w}x{trimmed_h} | "
                    f"padding applied: {padding}px")
    except Exception as e:
        logger.error(f"STEP 4 — FAILED | {str(e)}")
        raise

    # ── Step 5: Resize with LANCZOS ───────────────────────────
    logger.info("STEP 5 — Resizing with LANCZOS...")
    try:
        usable         = target_size - (padding * 2)
        orig_w, orig_h = img.size
        ratio          = min(usable / orig_w, usable / orig_h)

        # Never upscale beyond original resolution
        ratio = min(ratio, 1.0)

        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"STEP 5 — OK | {orig_w}x{orig_h} → {new_w}x{new_h} | "
                    f"ratio: {ratio:.3f} | "
                    f"{'kept original (no upscale)' if ratio == 1.0 else 'downscaled'}")
    except Exception as e:
        logger.error(f"STEP 5 — FAILED | {str(e)}")
        raise

    # ── Step 6: Sharpen after resize ──────────────────────────
    if sharpen:
        logger.info("STEP 6 — Applying sharpening filter...")
        try:
            img = img.filter(ImageFilter.SHARPEN)
            logger.info("STEP 6 — OK | sharpening applied")
        except Exception as e:
            logger.error(f"STEP 6 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 6 — SKIPPED | sharpen is false")

    # ── Step 7: Paste on white canvas ─────────────────────────
    logger.info("STEP 7 — Pasting on white canvas...")
    try:
        canvas   = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2

        if img.mode == "RGBA":
            canvas.paste(img, (offset_x, offset_y), mask=img.split()[3])
        else:
            img = img.convert("RGB")
            canvas.paste(img, (offset_x, offset_y))

        logger.info(f"STEP 7 — OK | canvas: {target_size}x{target_size} | "
                    f"image: {new_w}x{new_h} | "
                    f"white padding — top: {offset_y}px, bottom: {offset_y}px, "
                    f"left: {offset_x}px, right: {offset_x}px")
    except Exception as e:
        logger.error(f"STEP 7 — FAILED | {str(e)}")
        raise

    # ── Step 8: Save as JPEG q90, 150 DPI ─────────────────────
    logger.info("STEP 8 — Saving as JPEG q90, 150 DPI...")
    try:
        output_buffer = io.BytesIO()
        canvas.save(
            output_buffer,
            format   = "JPEG",
            quality  = 90,
            dpi      = (150, 150),
            optimize = True
        )
        output_buffer.seek(0)
        size_kb = len(output_buffer.getvalue()) / 1024
        logger.info(f"STEP 8 — OK | output size: {size_kb:.1f} KB")
    except Exception as e:
        logger.error(f"STEP 8 — FAILED | {str(e)}")
        raise

    # ── Step 9: Encode to base64 ──────────────────────────────
    logger.info("STEP 9 — Encoding to base64...")
    encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    logger.info(f"STEP 9 — OK | base64 length: {len(encoded)} chars")

    logger.info("=" * 60)
    logger.info("REQUEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return encoded
