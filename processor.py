import io
import base64
from PIL import Image, ImageFilter
from logger_config import setup_logger

logger = setup_logger("processor")

def process_image(
    image_input:  str,
    target_size:  int   = 1200,
    crop_bbox:    tuple = None,
    remove_bg:    bool  = False,
    padding:      int   = 20,
    sharpen:      bool  = True
) -> str:
    """
    FCI product image processing pipeline.
    Accepts base64 encoded image string.
    Returns base64 encoded JPEG string.
    """

    logger.info("=" * 60)
    logger.info("NEW IMAGE PROCESSING REQUEST STARTED")
    logger.info("=" * 60)
    logger.debug(f"Params — target_size: {target_size}, crop_bbox: {crop_bbox}, "
                 f"remove_bg: {remove_bg}, padding: {padding}, sharpen: {sharpen}")

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

    # ── Step 3: Optional crop ─────────────────────────────────
    if crop_bbox:
        logger.info(f"STEP 3 — Cropping to bbox: {crop_bbox}...")
        try:
            x1, y1, x2, y2 = crop_bbox
            img = img.crop((x1, y1, x2, y2))
            logger.info(f"STEP 3 — OK | new size after crop: {img.size}")
        except Exception as e:
            logger.error(f"STEP 3 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 3 — SKIPPED | no crop_bbox provided")

    # ── Step 4: Optional background removal ───────────────────
    if remove_bg:
        logger.info("STEP 4 — Removing background with rembg...")
        try:
            from rembg import remove
            img = remove(img)
            logger.info(f"STEP 4 — OK | size after rembg: {img.size}")
        except ImportError:
            logger.error("STEP 4 — FAILED | rembg not installed")
            raise Exception("rembg not installed — add it to requirements.txt")
        except Exception as e:
            logger.error(f"STEP 4 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 4 — SKIPPED | remove_bg is false")

    # ── Step 5: Resize with LANCZOS ───────────────────────────
    logger.info("STEP 5 — Resizing with LANCZOS...")
    try:
        usable        = target_size - (padding * 2)
        orig_w, orig_h = img.size
        ratio         = min(usable / orig_w, usable / orig_h)
        new_w         = int(orig_w * ratio)
        new_h         = int(orig_h * ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"STEP 5 — OK | {orig_w}x{orig_h} → {new_w}x{new_h} "
                    f"(ratio: {ratio:.3f}, usable area: {usable}px)")
    except Exception as e:
        logger.error(f"STEP 5 — FAILED | {str(e)}")
        raise

    # ── Step 6: Sharpen ───────────────────────────────────────
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
                    f"image offset: ({offset_x}, {offset_y})")
    except Exception as e:
        logger.error(f"STEP 7 — FAILED | {str(e)}")
        raise

    # ── Step 8: Save as JPEG ──────────────────────────────────
    logger.info("STEP 8 — Saving as JPEG q90, 300 DPI...")
    try:
        output_buffer = io.BytesIO()
        canvas.save(
            output_buffer,
            format   = "JPEG",
            quality  = 90,
            dpi      = (72, 72),
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
