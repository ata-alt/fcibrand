import io
import base64
from PIL import Image, ImageFilter
from logger_config import setup_logger

logger = setup_logger("processor")

def process_image(
    image_input:  str,
    target_size:  int  = 1200,
    padding:      int  = 10,
    sharpen:      bool = True
) -> str:
    """
    FCI product image processing pipeline.
    Accepts base64 encoded image string.
    Returns base64 encoded JPEG string.
    """

    logger.info("=" * 60)
    logger.info("NEW IMAGE PROCESSING REQUEST STARTED")
    logger.info("=" * 60)
    logger.debug(f"Params — target_size: {target_size}, "
                 f"padding: {padding}, sharpen: {sharpen}")

    # ── Step 1: Decode base64 ─────────────────────────────────
    logger.info("STEP 1 — Decoding base64 image...")
    try:
        image_bytes = base64.b64decode(image_input)
        img         = Image.open(io.BytesIO(image_bytes))
        logger.info(f"STEP 1 — OK | size: {img.size} | mode: {img.mode} | "
                    f"input bytes: {len(image_bytes)/1024:.1f} KB")
    except Exception as e:
        logger.error(f"STEP 1 — FAILED | {str(e)}")
        raise Exception(f"Invalid base64 image: {str(e)}")

    # ── Step 2: Convert to RGBA ───────────────────────────────
    logger.info("STEP 2 — Converting to RGBA...")
    try:
        img = img.convert("RGBA")
        logger.info(f"STEP 2 — OK | mode is now: {img.mode}")
    except Exception as e:
        logger.error(f"STEP 2 — FAILED | {str(e)}")
        raise

    # ── Step 3: Resize with LANCZOS ───────────────────────────
    logger.info("STEP 3 — Resizing with LANCZOS...")
    try:
        usable         = target_size - (padding * 2)
        orig_w, orig_h = img.size
        ratio          = min(usable / orig_w, usable / orig_h)

        # Never upscale beyond original resolution
        ratio = min(ratio, 1.0)

        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"STEP 3 — OK | {orig_w}x{orig_h} → {new_w}x{new_h} | "
                    f"ratio: {ratio:.3f} | "
                    f"{'kept original size (no upscale)' if ratio == 1.0 else 'downscaled'}")
    except Exception as e:
        logger.error(f"STEP 3 — FAILED | {str(e)}")
        raise

    # ── Step 4: Sharpen after resize ──────────────────────────
    if sharpen:
        logger.info("STEP 4 — Applying sharpening filter...")
        try:
            img = img.filter(ImageFilter.SHARPEN)
            logger.info("STEP 4 — OK | sharpening applied")
        except Exception as e:
            logger.error(f"STEP 4 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 4 — SKIPPED | sharpen is false")

    # ── Step 5: Paste on white canvas with padding ────────────
    logger.info("STEP 5 — Pasting on white canvas...")
    try:
        canvas   = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2

        if img.mode == "RGBA":
            canvas.paste(img, (offset_x, offset_y), mask=img.split()[3])
        else:
            img = img.convert("RGB")
            canvas.paste(img, (offset_x, offset_y))

        logger.info(f"STEP 5 — OK | canvas: {target_size}x{target_size} | "
                    f"image: {new_w}x{new_h} | "
                    f"offset: ({offset_x}, {offset_y}) | "
                    f"white padding — top: {offset_y}px, bottom: {offset_y}px, "
                    f"left: {offset_x}px, right: {offset_x}px")
    except Exception as e:
        logger.error(f"STEP 5 — FAILED | {str(e)}")
        raise

    # ── Step 6: Save as JPEG q90, 150 DPI ─────────────────────
    logger.info("STEP 6 — Saving as JPEG q90, 150 DPI...")
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
        logger.info(f"STEP 6 — OK | output size: {size_kb:.1f} KB")
    except Exception as e:
        logger.error(f"STEP 6 — FAILED | {str(e)}")
        raise

    # ── Step 7: Encode to base64 ──────────────────────────────
    logger.info("STEP 7 — Encoding to base64...")
    encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    logger.info(f"STEP 7 — OK | base64 length: {len(encoded)} chars")

    logger.info("=" * 60)
    logger.info("REQUEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return encoded
