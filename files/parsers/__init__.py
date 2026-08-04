"""
files/parsers package.

Each parser returns a standardized dict:
    {
        "success": bool,
        "text": str,
        "metadata": dict,
        "error": str | None,
        ...
    }
"""

from .pdf import extract_pdf
from .office import (
    extract_docx,
    extract_xlsx,
    extract_pptx,
)
from .image import extract_image
from .code import extract_code
from .archive import extract_archive

__all__ = [
    "extract_pdf",
    "extract_docx",
    "extract_xlsx",
    "extract_pptx",
    "extract_image",
    "extract_code",
    "extract_archive",
]
