import os
import logging
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

class ProcessRequest(BaseModel):
    image:       str
    target_size: int  = 1200
    padding:     int  = 10
    sharpen:     bool = True

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
                    f"padding: {req.padding}, sharpen: {req.sharpen}")

        base64_image = process_image(
            image_input = req.image,
            target_size = req.target_size,
            padding     = req.padding,
            sharpen     = req.sharpen
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

# ── Log viewer endpoint ────────────────────────────────────
@app.get("/logs", response_class=PlainTextResponse)
def view_logs(lines: int = 100):
    """
    View last N lines of the log file.
    Usage: /logs        → last 100 lines
           /logs?lines=50  → last 50 lines
    """
    log_file = "/tmp/logs/fci_processor.log"
    try:
        if not os.path.exists(log_file):
            return "No log file found yet — make a request first."

        with open(log_file, "r") as f:
            all_lines = f.readlines()

        last_n = all_lines[-lines:]
        return "".join(last_n)

    except Exception as e:
        return f"Error reading log file: {str(e)}"

# ── Clear logs endpoint ────────────────────────────────────
@app.delete("/logs")
def clear_logs():
    log_file = "/tmp/logs/fci_processor.log"
    try:
        open(log_file, "w").close()
        logger.info("Log file cleared")
        return {"success": True, "message": "Logs cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
