import io
import base64
from PIL import Image, ImageFilter, ImageChops
from logger_config import setup_logger

logger = setup_logger("processor")


def detect_background_color(img: Image.Image, sample_size: int = 5) -> tuple:
    """
    Samples corners of the image to detect background color.
    Returns average RGB of corner pixels.
    
    Samples a small patch from each corner:
    ┌──────────────────────────┐
    │ ▓▓                   ▓▓ │  ← sample these 4 corners
    │                         │
    │                         │
    │ ▓▓                   ▓▓ │
    └──────────────────────────┘
    """
    rgb = img.convert("RGB")
    w, h = rgb.size

    # Sample patches from all 4 corners
    corners = [
        rgb.crop((0,         0,          sample_size, sample_size)),  # top-left
        rgb.crop((w-sample_size, 0,       w,           sample_size)),  # top-right
        rgb.crop((0,         h-sample_size, sample_size, h)),          # bottom-left
        rgb.crop((w-sample_size, h-sample_size, w,     h)),            # bottom-right
    ]

    # Get average color of each corner patch
    r_vals, g_vals, b_vals = [], [], []
    for corner in corners:
        pixels = list(corner.getdata())
        r_vals.extend([p[0] for p in pixels])
        g_vals.extend([p[1] for p in pixels])
        b_vals.extend([p[2] for p in pixels])

    avg_r = int(sum(r_vals) / len(r_vals))
    avg_g = int(sum(g_vals) / len(g_vals))
    avg_b = int(sum(b_vals) / len(b_vals))

    logger.info(f"  Detected background color: RGB({avg_r}, {avg_g}, {avg_b})")
    return (avg_r, avg_g, avg_b)


def trim_whitespace(img: Image.Image, tolerance: int = 30) -> Image.Image:
    """
    Trims background from image using smart corner color detection.
    
    tolerance: how much color variation to allow
      10  = very tight, only pixels very close to background color
      30  = recommended, handles gradients and shadows
      50  = aggressive, may trim product edges on very light products
    """
    # Detect actual background color from corners
    bg_color = detect_background_color(img)

    rgb = img.convert("RGB")

    # Build reference background using detected color
    bg = Image.new("RGB", rgb.size, bg_color)

    # Find difference between image and detected background
    diff = ImageChops.difference(rgb, bg)

    # Apply tolerance — pixels within tolerance treated as background
    # Enhance the difference to make non-background stand out
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(diff)
    diff     = enhancer.enhance(2.0)

    bbox = diff.getbbox()

    if bbox:
        # Add tiny safety margin to avoid cutting product edges
        margin = 2
        w, h   = img.size
        bbox   = (
            max(0,   bbox[0] - margin),
            max(0,   bbox[1] - margin),
            min(w,   bbox[2] + margin),
            min(h,   bbox[3] + margin),
        )
        logger.info(f"  Trimmed background | bbox: {bbox} | "
                    f"before: {img.size} | "
                    f"after: {(bbox[2]-bbox[0], bbox[3]-bbox[1])}")
        return img.crop(bbox)
    else:
        logger.info("  Nothing to trim — uniform background")
        return img


def get_padding_for_orientation(w: int, h: int,
                                landscape_padding: int,
                                portrait_padding:  int,
                                square_padding:    int) -> tuple:
    """Returns (padding, orientation) based on dimensions."""
    if w > h:
        return landscape_padding, "landscape"
    elif h > w:
        return portrait_padding,  "portrait"
    else:
        return square_padding,    "square"


def process_image(
    image_input:        str,
    target_size:        int  = 1200,
    landscape_padding:  int  = 0,
    portrait_padding:   int  = 1,
    square_padding:     int  = 1,
    sharpen:            bool = True,
    trim_white:         bool = True,
    trim_tolerance:     int  = 50    # ← increase from 30 to 50
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

    # ── Step 3: Smart trim background ────────────────────────
    if trim_white:
        logger.info("STEP 3 — Smart trimming background...")
        try:
            size_before = img.size
            img         = trim_whitespace(img, tolerance=trim_tolerance)  # ← use param
            size_after  = img.size
            logger.info(f"STEP 3 — OK | {size_before} → {size_after} | "
                        f"removed: {size_before[0]-size_after[0]}px W, "
                        f"{size_before[1]-size_after[1]}px H | "
                        f"tolerance used: {trim_tolerance}")
        except Exception as e:
            logger.error(f"STEP 3 — FAILED | {str(e)}")
            raise
    else:
        logger.info("STEP 3 — SKIPPED | trim_white is false")

    # ── Step 4: Detect orientation + dynamic padding ──────────
    logger.info("STEP 4 — Detecting orientation...")
    try:
        trimmed_w, trimmed_h = img.size
        padding, orientation = get_padding_for_orientation(
            trimmed_w, trimmed_h,
            landscape_padding,
            portrait_padding,
            square_padding
        )
        logger.info(f"STEP 4 — OK | orientation: {orientation} | "
                    f"trimmed size: {trimmed_w}x{trimmed_h} | "
                    f"padding: {padding}px")
    except Exception as e:
        logger.error(f"STEP 4 — FAILED | {str(e)}")
        raise

    # ── Step 5: Resize with LANCZOS ───────────────────────────
    # ✅ Upscaling is NOW allowed after trim
    # Product should fill the canvas after background is removed
    logger.info("STEP 5 — Resizing with LANCZOS...")
    try:
        usable         = target_size - (padding * 2)
        orig_w, orig_h = img.size
        ratio          = min(usable / orig_w, usable / orig_h)
        # No ratio cap — upscale allowed to fill canvas after trim

        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        action = "upscaled" if ratio > 1.0 else "downscaled"
        logger.info(f"STEP 5 — OK | {orig_w}x{orig_h} → {new_w}x{new_h} | "
                    f"ratio: {ratio:.3f} | {action}")
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
                    f"padding — top: {offset_y}px, bottom: {offset_y}px, "
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
