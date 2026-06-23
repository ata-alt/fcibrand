from PIL import Image
import io
import base64
from logger_config import setup_logger

logger = setup_logger("canvas_filler")


def fill_canvas(
    image_input: str,
    target_size: int   = 1200,
    focus_x:     float = 0.5,
    focus_y:     float = 0.5,
) -> str:
    """
    Removes white padding from a processed canvas image by zooming
    content to fill the full target_size x target_size square.

    Accepts base64 encoded JPEG string (output of processor.py).
    Returns base64 encoded JPEG string at same target_size and 72 DPI.

    focus_x / focus_y : normalised focal point (0.0–1.0)
                        default 0.5, 0.5 = centre (correct for product photography)
    """

    logger.info("=" * 60)
    logger.info("CANVAS FILLER STARTED")
    logger.info("=" * 60)

    # ── Step 1: Decode base64 ─────────────────────────────────
    logger.info("FILL STEP 1 — Decoding base64 image...")
    try:
        image_bytes = base64.b64decode(image_input)
        img         = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"FILL STEP 1 — OK | size: {img.size} | mode: {img.mode}")
    except Exception as e:
        logger.error(f"FILL STEP 1 — FAILED | {str(e)}")
        raise Exception(f"Invalid base64 image: {str(e)}")

    # ── Step 2: Detect white padding via corner sampling ───────
    logger.info("FILL STEP 2 — Detecting white padding...")
    try:
        w, h       = img.size
        sample     = 5
        corners    = [
            img.getpixel((0,     0)),
            img.getpixel((w-1,   0)),
            img.getpixel((0,     h-1)),
            img.getpixel((w-1,   h-1)),
        ]
        avg_r = sum(c[0] for c in corners) // 4
        avg_g = sum(c[1] for c in corners) // 4
        avg_b = sum(c[2] for c in corners) // 4
        is_white_padding = (avg_r > 240 and avg_g > 240 and avg_b > 240)
        logger.info(f"FILL STEP 2 — corner avg RGB: ({avg_r},{avg_g},{avg_b}) | "
                    f"white padding detected: {is_white_padding}")
    except Exception as e:
        logger.error(f"FILL STEP 2 — FAILED | {str(e)}")
        raise

    if not is_white_padding:
        logger.info("FILL STEP 2 — No white padding found, returning image as-is")
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=90,
                 dpi=(72, 72), optimize=True)
        output_buffer.seek(0)
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    # ── Step 3: Find content bbox (non-white region) ───────────
    logger.info("FILL STEP 3 — Finding content bounding box...")
    try:
        from PIL import ImageChops
        bg   = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        diff = diff.point(lambda x: 0 if x < 10 else 255)
        bbox = diff.convert("L").getbbox()

        if bbox is None:
            logger.info("FILL STEP 3 — Entirely white image, returning as-is")
            output_buffer = io.BytesIO()
            img.save(output_buffer, format="JPEG", quality=90,
                     dpi=(72, 72), optimize=True)
            output_buffer.seek(0)
            return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        # Small safety margin
        margin = 4
        bbox = (
            max(0,   bbox[0] - margin),
            max(0,   bbox[1] - margin),
            min(w,   bbox[2] + margin),
            min(h,   bbox[3] + margin),
        )
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        logger.info(f"FILL STEP 3 — OK | bbox: {bbox} | "
                    f"content: {content_w}x{content_h}")
    except Exception as e:
        logger.error(f"FILL STEP 3 — FAILED | {str(e)}")
        raise

    # ── Step 4: Crop to content region ────────────────────────
    logger.info("FILL STEP 4 — Cropping to content region...")
    try:
        content = img.crop(bbox)
        logger.info(f"FILL STEP 4 — OK | cropped: {content.size}")
    except Exception as e:
        logger.error(f"FILL STEP 4 — FAILED | {str(e)}")
        raise

    # ── Step 5: Scale to fill target_size ─────────────────────
    logger.info("FILL STEP 5 — Scaling to fill canvas...")
    try:
        scale    = max(target_size / content_w, target_size / content_h)
        scaled_w = int(content_w * scale)
        scaled_h = int(content_h * scale)
        content  = content.resize((scaled_w, scaled_h), Image.LANCZOS)
        logger.info(f"FILL STEP 5 — OK | {content_w}x{content_h} → "
                    f"{scaled_w}x{scaled_h} | scale: {scale:.3f}")
    except Exception as e:
        logger.error(f"FILL STEP 5 — FAILED | {str(e)}")
        raise

    # ── Step 6: Crop to target_size anchored at focal point ────
    logger.info(f"FILL STEP 6 — Cropping to {target_size}x{target_size} "
                f"at focus ({focus_x}, {focus_y})...")
    try:
        focal_px = int(scaled_w * focus_x)
        focal_py = int(scaled_h * focus_y)

        left = focal_px - target_size // 2
        top  = focal_py - target_size // 2

        # clamp so crop never exceeds scaled image bounds
        left = max(0, min(left, scaled_w - target_size))
        top  = max(0, min(top,  scaled_h - target_size))

        final = content.crop((left, top,
                               left + target_size,
                               top  + target_size))
        logger.info(f"FILL STEP 6 — OK | crop box: "
                    f"({left}, {top}, {left+target_size}, {top+target_size})")
    except Exception as e:
        logger.error(f"FILL STEP 6 — FAILED | {str(e)}")
        raise

    # ── Step 7: Save as JPEG q90, 72 DPI ──────────────────────
    logger.info("FILL STEP 7 — Saving as JPEG q90, 72 DPI...")
    try:
        output_buffer = io.BytesIO()
        final.save(output_buffer, format="JPEG", quality=90,
                   dpi=(72, 72), optimize=True)
        output_buffer.seek(0)
        size_kb = len(output_buffer.getvalue()) / 1024
        logger.info(f"FILL STEP 7 — OK | output size: {size_kb:.1f} KB")
    except Exception as e:
        logger.error(f"FILL STEP 7 — FAILED | {str(e)}")
        raise

    # ── Step 8: Encode to base64 ──────────────────────────────
    encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    logger.info(f"FILL STEP 8 — OK | base64 length: {len(encoded)} chars")

    logger.info("=" * 60)
    logger.info("CANVAS FILLER COMPLETED")
    logger.info("=" * 60)

    return encoded
