import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from processor import process_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FCI Image Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ──────────────────────────────────────────
class ProcessRequest(BaseModel):
    image:       str
    target_size: int = 1200        # ← changed from 1000 to 1200
    crop_bbox:   Optional[List[int]] = None
    remove_bg:   bool = False
    padding:     int = 20
    sharpen:     bool = True

# ── Health check ───────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "service": "fci-image-processor"}

# ── Main processing endpoint ───────────────────────────────
@app.post("/process")
def process(req: ProcessRequest):
    try:
        logger.info("Received image processing request")

        base64_image = process_image(
            image_input = req.image,
            target_size = req.target_size,
            crop_bbox   = tuple(req.crop_bbox) if req.crop_bbox else None,
            remove_bg   = req.remove_bg,
            padding     = req.padding,
            sharpen     = req.sharpen
        )

        return {
            "success": True,
            "image":   base64_image,
            "format":  "JPEG",
            "size":    req.target_size,
            "dpi":     300
        }

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
