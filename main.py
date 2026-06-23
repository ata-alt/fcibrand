import os
import io
import zipfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from processor import process_image
from canvas_filler import fill_canvas
from extractor import extract_swatches, NoSwatchPageError
from logger_config import setup_logger

logger = setup_logger("main")

app = FastAPI(title="FCI Image Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── image_type → trim_white mapping ───────────────────────
IMAGE_TYPE_MAP = {
    "studio_clean":       True,   # real alpha or rembg, reliable trim
    "studio_clean_light": False,  # skip trim, avoid light-edge clipping
    "studio_shadow":      False,  # skip trim, preserve shadow
    "lifestyle":          False,  # skip trim, background intentional
}

# image types where white padding will exist after processor
# → canvas filler runs automatically on these
NEEDS_FILL = {"studio_clean_light", "studio_shadow", "lifestyle"}

# ── Request model ──────────────────────────────────────────
class ProcessRequest(BaseModel):
    image:               str
    studio_clean:        Optional[bool] = None
    studio_clean_light:  Optional[bool] = None
    studio_shadow:       Optional[bool] = None
    lifestyle:           Optional[bool] = None
    target_size:         int  = Field(default=1200, ge=100, le=5000)
    landscape_padding:   int  = Field(default=0,    ge=0,   le=200)
    portrait_padding:    int  = Field(default=1,    ge=0,   le=200)
    square_padding:      int  = Field(default=1,    ge=0,   le=200)
    sharpen:             bool = True

# ── Health check ───────────────────────────────────────────
@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "ok", "service": "fci-image-processor"}

# ── Main processing endpoint ───────────────────────────────
@app.post("/process")
def process(req: ProcessRequest):
    try:
        # ── Determine which image_type field is True ───────────
        image_type = None
        for field in ["studio_clean", "studio_clean_light", "studio_shadow", "lifestyle"]:
            if getattr(req, field) is True:
                image_type = field
                break

        if image_type is None:
            raise ValueError(
                "No image_type field is True. "
                "Expected exactly one of: studio_clean, studio_clean_light, "
                "studio_shadow, lifestyle to be true."
            )

        trim_white  = IMAGE_TYPE_MAP[image_type]
        should_fill = image_type in NEEDS_FILL

        logger.info(f"Received request — image_type: {image_type} | "
                    f"trim_white: {trim_white} | "
                    f"canvas_fill: {should_fill} | "
                    f"target_size: {req.target_size} | "
                    f"landscape_padding: {req.landscape_padding} | "
                    f"portrait_padding: {req.portrait_padding} | "
                    f"square_padding: {req.square_padding} | "
                    f"sharpen: {req.sharpen}")

        # ── Stage 1: Process image ─────────────────────────────
        base64_image = process_image(
            image_input       = req.image,
            target_size       = req.target_size,
            landscape_padding = req.landscape_padding,
            portrait_padding  = req.portrait_padding,
            square_padding    = req.square_padding,
            sharpen           = req.sharpen,
            trim_white        = trim_white,
        )

        # ── Stage 2: Fill canvas (remove white padding) ────────
        if should_fill:
            logger.info("STAGE 2 — Running canvas filler...")
            base64_image = fill_canvas(
                image_input = base64_image,
                target_size = req.target_size,
                focus_x     = 0.5,
                focus_y     = 0.5,
            )
            logger.info("STAGE 2 — Canvas filler done")
        else:
            logger.info("STAGE 2 — SKIPPED | studio_clean handles its own trim")

        return {
            "success":       True,
            "image":         base64_image,
            "format":        "JPEG",
            "size":          req.target_size,
            "dpi":           72,
            "image_type":    image_type,
            "trim_white":    trim_white,
            "canvas_filled": should_fill,
        }

    except Exception as e:
        logger.error(f"Request failed — {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Swatch extraction — returns ZIP of PNGs ───────────────
@app.post("/extract")
async def extract_to_zip(
    file:         UploadFile    = File(..., description="Furniture catalog PDF"),
    page:         Optional[int] = Form(None, description="Page number (None = auto-detect)"),
    expected_min: int           = Form(3,    description="Min swatches before Gemini fallback"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large (max 50MB)")

    try:
        swatches = extract_swatches(
            pdf_bytes    = pdf_bytes,
            page_num     = page,
            expected_min = expected_min,
            gemini_key   = os.getenv("GEMINI_API_KEY"),
        )
    except NoSwatchPageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Swatch extraction failed — {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    if not swatches:
        raise HTTPException(status_code=404, detail="No swatches detected on the identified page. Try specifying a page number manually.")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sw in swatches:
            zf.writestr(f"{sw.category}__{sw.label}.png", sw.image_bytes)
        manifest = {
            "source":  file.filename,
            "page":    page,
            "total":   len(swatches),
            "swatches": [
                {
                    "label":      sw.label,
                    "category":   sw.category,
                    "file":       f"{sw.category}__{sw.label}.png",
                    "size":       f"{sw.width}x{sw.height}",
                    "confidence": sw.confidence,
                }
                for sw in swatches
            ],
        }
        import json
        zf.writestr("_manifest.json", json.dumps(manifest, indent=2))

    zip_buf.seek(0)
    zip_name = file.filename.replace(".pdf", "_swatches.zip")
    logger.info(f"Swatch ZIP — {len(swatches)} swatches from '{file.filename}' page {page}")

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )

# ── Swatch extraction — returns JSON with base64 images ───
@app.post("/extract/json")
async def extract_to_json(
    file:         UploadFile    = File(...),
    page:         Optional[int] = Form(None),
    expected_min: int           = Form(3),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()

    try:
        swatches = extract_swatches(
            pdf_bytes    = pdf_bytes,
            page_num     = page,
            expected_min = expected_min,
            gemini_key   = os.getenv("GEMINI_API_KEY"),
        )
    except NoSwatchPageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Swatch extraction failed — {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Swatch JSON — {len(swatches)} swatches from '{file.filename}' page {page}")

    return JSONResponse({
        "source":  file.filename,
        "page":    page,
        "total":   len(swatches),
        "swatches": [
            {
                "label":          sw.label,
                "category":       sw.category,
                "width":          sw.width,
                "height":         sw.height,
                "confidence":     sw.confidence,
                "image_b64":      sw.image_b64,
                "image_data_uri": f"data:image/png;base64,{sw.image_b64}",
            }
            for sw in swatches
        ],
    })

# ── Log viewer ─────────────────────────────────────────────
@app.get("/logs", response_class=PlainTextResponse)
def view_logs(lines: int = 100):
    log_file = "/tmp/logs/fci_processor.log"
    try:
        if not os.path.exists(log_file):
            return "No log file found yet — make a request first."
        with open(log_file, "r") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading log file: {str(e)}"

# ── Clear logs ─────────────────────────────────────────────
@app.delete("/logs")
def clear_logs():
    log_file = "/tmp/logs/fci_processor.log"
    try:
        open(log_file, "w").close()
        logger.info("Log file cleared")
        return {"success": True, "message": "Logs cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
