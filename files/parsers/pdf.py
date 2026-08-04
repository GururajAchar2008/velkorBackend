"""
files/parsers/pdf.py

PDF extraction utility for Velkor AI.
"""

from typing import Dict, List
import logging
import PyPDF2

logger = logging.getLogger(__name__)


def extract_pdf(file_stream) -> Dict:
    """
    Extract text and metadata from a PDF.

    Returns:
        {
            "success": bool,
            "text": str,
            "pages": list[str],
            "metadata": dict,
            "error": str | None
        }
    """
    try:
        reader = PyPDF2.PdfReader(file_stream)

        pages: List[str] = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")

        text = "\n\n".join(pages)

        meta = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                meta[str(k)] = str(v)

        return {
            "success": True,
            "text": text.strip(),
            "pages": pages,
            "metadata": meta,
            "error": None,
        }

    except Exception as e:
        logger.exception("PDF extraction failed")
        return {
            "success": False,
            "text": "",
            "pages": [],
            "metadata": {},
            "error": str(e),
        }


def extract_pdf_preview(file_stream, max_chars: int = 2000) -> str:
    """
    Return only the first part of a PDF for previews.
    """
    result = extract_pdf(file_stream)
    if not result["success"]:
        return ""
    return result["text"][:max_chars]
