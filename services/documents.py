"""
services/documents.py

Universal file parsing service for Velkor AI.

Delegates to the files/extractor pipeline so every supported format is
fully extracted into text that reaches the AI model:

  - PDF (per-page text + metadata)
  - Word / Excel / PowerPoint (.docx .doc .xlsx .xlsm .xls .pptx .ppt)
  - ZIP / TAR archives (recursive text extraction)
  - Images (metadata + optional OCR)
  - Code & plain-text files (30+ formats)
"""

import base64
from pathlib import Path
from typing import Dict, Any

from files.extractor import extract_file, extract_file_path
from utils.logger import get_logger

logger = get_logger(__name__)


def _run_ocr(image_bytes: bytes) -> str:
    """
    Best-effort OCR via pytesseract (only if the system has Tesseract).

    Returns extracted text, or empty string when unavailable.
    """
    try:
        import io

        from PIL import Image

        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


class DocumentService:
    """
    Extracts complete text content from uploaded files for the AI model.
    """

    def parse_file(self, file_path) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path.name}", "text": ""}

        ext = path.suffix.lower()
        logger.info("Parsing uploaded document: %s (extension: %s)", path.name, ext)

        try:
            result = extract_file_path(str(path))
        except Exception as e:
            logger.error("Failed to parse file %s: %s", path.name, str(e))
            return {"success": False, "error": str(e), "text": ""}

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Failed to parse file"),
                "text": "",
            }

        metadata = result.get("metadata", {}) or {}
        text = result.get("text", "") or ""
        image_payload = self._image_payload(result)

        if not text and image_payload:
            text = self._image_text(image_payload)

        return {
            "success": True,
            "text": text,
            "type": ext.lstrip(".") or "file",
            "metadata": metadata,
            "image": image_payload,
        }

    def parse_upload(self, file_storage) -> Dict[str, Any]:
        """Parse a Flask FileStorage object directly."""
        try:
            result = extract_file(file_storage)
        except Exception as e:
            logger.error("Failed to parse upload: %s", str(e))
            return {"success": False, "error": str(e), "text": ""}

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Failed to parse file"),
                "text": "",
            }

        metadata = result.get("metadata", {}) or {}
        text = result.get("text", "") or ""
        image_payload = self._image_payload(result)

        if not text and image_payload:
            text = self._image_text(image_payload)

        return {
            "success": True,
            "text": text,
            "type": "upload",
            "metadata": metadata,
            "image": image_payload,
        }

    def _image_payload(self, result) -> Dict[str, Any] | None:
        image_bytes = result.get("image_bytes")
        if not image_bytes:
            return None

        meta = result.get("metadata", {}) or {}
        return {
            "format": meta.get("format", ""),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "ocr_text": _run_ocr(image_bytes),
            "base64": base64.b64encode(image_bytes).decode("ascii"),
        }

    def _image_text(self, image_payload: Dict[str, Any]) -> str:
        lines = [
            f"[Uploaded image: {image_payload['format']} "
            f"{image_payload['width']}x{image_payload['height']}px]"
        ]
        if image_payload.get("ocr_text"):
            lines.append("Extracted text from the image:")
            lines.append(image_payload["ocr_text"])
        else:
            lines.append(
                "The image is attached as context. "
                "Describe what you can infer; OCR was not available on the server."
            )
        return "\n".join(lines)


document_service = DocumentService()
