import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
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

# ── Request model ──────────────────────────────────────────
class ProcessRequest(BaseModel):
    image:               str
    target_size:         int  = 1200
    landscape_padding:   int  = 6     # ← landscape gets tight padding
    portrait_padding:    int  = 10    # ← portrait gets slightly more
    square_padding:      int  = 10    # ← square same as portrait
    sharpen:             bool = True
    trim_white:          bool = True

# ── Health check ───────────────────────────────────────────
@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "ok", "service": "fci-image-processor"}

# ── Main processing endpoint ───────────────────────────────
@app.post("/process")
def process(req: ProcessRequest):
    try:
        logger.info(f"Received request — target_size: {req.target_size}, "
                    f"landscape_padding: {req.landscape_padding}, "
                    f"portrait_padding: {req.portrait_padding}, "
                    f"sharpen: {req.sharpen}, trim_white: {req.trim_white}")

        base64_image = process_image(
            image_input       = req.image,
            target_size       = req.target_size,
            landscape_padding = req.landscape_padding,
            portrait_padding  = req.portrait_padding,
            square_padding    = req.square_padding,
            sharpen           = req.sharpen,
            trim_white        = req.trim_white
        )

        return {
            "success": True,
            "image":   base64_image,
            "format":  "JPEG",
            "size":    req.target_size,
            "dpi":     150
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
