"""
files/parsers/image.py

Image parser for Velkor AI.
This version extracts metadata and prepares images for OCR/Vision.
"""

from PIL import Image
from typing import Dict
import io
import logging

logger = logging.getLogger(__name__)


def extract_image(file_stream) -> Dict:
    """
    Extract image metadata.

    OCR and Vision model integration will be added later.
    """
    try:
        data = file_stream.read()
        img = Image.open(io.BytesIO(data))

        metadata = {
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
        }

        return {
            "success": True,
            "text": "",
            "metadata": metadata,
            "image_bytes": data,
            "error": None,
        }

    except Exception as e:
        logger.exception("Image extraction failed")

        return {
            "success": False,
            "text": "",
            "metadata": {},
            "image_bytes": None,
            "error": str(e),
        }


def is_supported_image(filename: str) -> bool:
    filename = filename.lower()

    return filename.endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".tiff",
        )
    )
