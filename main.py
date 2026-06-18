import io
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from processor import process_image

app = FastAPI(title="FCI Image Processor")

# Allow your JS server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ──────────────────────────────────────────
class ProcessRequest(BaseModel):
    source_url:  str
    target_size: int = 1000
    crop_bbox:   Optional[List[int]] = None  # [x1, y1, x2, y2]
    remove_bg:   bool = False
    padding:     int = 20
    sharpen:     bool = True

# ── Health check (Railway uses this to confirm app is live) ──
@app.get("/")
def health_check():
    return {"status": "ok", "service": "fci-image-processor"}

# ── Main processing endpoint ───────────────────────────────
@app.post("/process")
def process(req: ProcessRequest):
    try:
        base64_image = process_image(
            source_url  = req.source_url,
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

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
