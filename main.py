import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional
from processor import process_image
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
        # OpenClaw sends booleans — exactly one should be True
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

        trim_white = IMAGE_TYPE_MAP[image_type]

        logger.info(f"Received request — image_type: {image_type} | "
                    f"trim_white: {trim_white} | "
                    f"target_size: {req.target_size} | "
                    f"landscape_padding: {req.landscape_padding} | "
                    f"portrait_padding: {req.portrait_padding} | "
                    f"square_padding: {req.square_padding} | "
                    f"sharpen: {req.sharpen}")

        base64_image = process_image(
            image_input       = req.image,
            target_size       = req.target_size,
            landscape_padding = req.landscape_padding,
            portrait_padding  = req.portrait_padding,
            square_padding    = req.square_padding,
            sharpen           = req.sharpen,
            trim_white        = trim_white,
        )

        return {
            "success":    True,
            "image":      base64_image,
            "format":     "JPEG",
            "size":       req.target_size,
            "dpi":        72,
            "image_type": image_type,
            "trim_white": trim_white
        }

    except Exception as e:
        logger.error(f"Request failed — {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
