"""
files/extractor.py

Universal file extractor for Velkor AI.
"""

import os

from .parsers.pdf import extract_pdf
from .parsers.office import (
    extract_docx,
    extract_xlsx,
    extract_pptx,
)
from .parsers.image import extract_image
from .parsers.archive import extract_archive
from .parsers.code import extract_code


PDF = {".pdf"}
DOCX = {".docx"}
XLSX = {".xlsx", ".xlsm"}
PPTX = {".pptx"}
IMAGE = {
    ".png", ".jpg", ".jpeg",
    ".webp", ".bmp", ".gif", ".tiff"
}
ARCHIVE = {".zip"}

CODE = {
    ".py",".js",".ts",".jsx",".tsx",
    ".java",".c",".cpp",".cc",".cs",
    ".go",".rs",".php",".rb",".swift",
    ".kt",".scala",".sql",".json",".xml",
    ".yaml",".yml",".html",".css",".md",
    ".txt",".sh",".bat"
}


def extract_file(file_storage):
    """
    Flask FileStorage -> standardized extraction result.
    """

    filename = file_storage.filename or "unknown"

    ext = os.path.splitext(filename)[1].lower()

    stream = file_storage.stream

    stream.seek(0)

    if ext in PDF:
        return extract_pdf(stream)

    if ext in DOCX:
        return extract_docx(stream)

    if ext in XLSX:
        return extract_xlsx(stream)

    if ext in PPTX:
        return extract_pptx(stream)

    if ext in IMAGE:
        return extract_image(stream)

    if ext in ARCHIVE:
        return extract_archive(stream)

    if ext in CODE:
        return extract_code(stream, filename)

    return {
        "success": False,
        "text": "",
        "metadata": {},
        "error": f"Unsupported file type: {ext}",
    }
