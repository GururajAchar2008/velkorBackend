"""
utils/logger.py
Central logging utility for Velkor AI Backend.
"""

import logging
import logging.handlers
import os
from config import Config

os.makedirs("logs", exist_ok=True)

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

_root = logging.getLogger()
_root.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

if not _root.handlers:
    console = logging.StreamHandler()
    console.setFormatter(_formatter)
    _root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_formatter)
    _root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_provider(provider: str, model: str, response_time: float):
    logger = get_logger("providers")
    logger.info(
        "Provider=%s Model=%s ResponseTime=%.2fs",
        provider,
        model,
        response_time,
    )


def log_latency(response_time: float, kind: str = "chat"):
    """Log chat latency against the 2–3s target. Image gen is excluded."""
    if kind != "chat":
        return
    logger = get_logger("latency")
    target = Config.CHAT_LATENCY_TARGET_S
    if response_time > target:
        logger.warning(
            "Chat latency %.2fs exceeded %.1fs target",
            response_time,
            target,
        )
    else:
        logger.info(
            "Chat latency %.2fs (target %.1fs)",
            response_time,
            target,
        )


def log_upload(filename: str, size: int):
    logger = get_logger("uploads")
    logger.info(
        "Uploaded file=%s Size=%d bytes",
        filename,
        size,
    )


def log_rag(source: str, chunks: int):
    logger = get_logger("rag")
    logger.info(
        "Source=%s Chunks=%d",
        source,
        chunks,
    )


def log_search(query: str, results: int):
    logger = get_logger("search")
    logger.info("Query=%r Results=%d", query[:120], results)


def log_error(location: str, exc: Exception):
    logger = get_logger(location)
    logger.exception(str(exc))
