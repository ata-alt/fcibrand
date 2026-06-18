import logging
import os
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up logger that writes to both:
    - Console (stdout) → visible in Railway dashboard permanently
    - Log file         → /tmp/fci_processor.log (current session)
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ── Formatter ─────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    # ── Handler 1: Console (stdout) ───────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Handler 2: File ───────────────────────────────────────
    # /tmp is the only writable directory on Railway
    log_dir  = "/tmp/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "fci_processor.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)   # DEBUG level for file — more detail
    file_handler.setFormatter(formatter)

    # ── Attach handlers ───────────────────────────────────────
    if not logger.handlers:  # avoid duplicate handlers on reload
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
