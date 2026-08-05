"""
files/extractor.py

Universal file extractor for Velkor AI.
"""

import io
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
DOCX = {".docx", ".doc"}
XLSX = {".xlsx", ".xlsm", ".xls"}
PPTX = {".pptx", ".ppt"}
IMAGE = {
    ".png", ".jpg", ".jpeg",
    ".webp", ".bmp", ".gif", ".tiff", ".svg",
    ".heic", ".heif",
}
ARCHIVE = {".zip", ".tar", ".gz", ".tgz"}

CODE = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".cc", ".h", ".hpp", ".cs",
    ".go", ".rs", ".php", ".rb", ".swift",
    ".kt", ".scala", ".sql", ".json", ".xml",
    ".yaml", ".yml", ".html", ".htm", ".css", ".md",
    ".txt", ".sh", ".bat", ".csv", ".tsv",
    ".log", ".ini", ".toml", ".cfg", ".conf",
    ".env", ".tex", ".rst", ".rtf", ".vtt", ".srt",
}

TEXT_EXTENSIONS = CODE


def _ext(filepath):
    return os.path.splitext(filepath)[1].lower()


def extract_file_path(filepath):
    """Extract content from a file on disk (path-based API)."""
    filename = os.path.basename(filepath)
    ext = _ext(filename)

    with open(filepath, "rb") as f:
        data = f.read()

    stream = io.BytesIO(data)
    return dispatch(stream, filename, ext)


def extract_text(filepath):
    """Alias used by upload routes."""
    return extract_file_path(filepath)


def dispatch(stream, filename: str, ext: str):
    """Route a stream to the right parser based on file extension."""

    result = None

    if ext in PDF:
        result = extract_pdf(stream)

    elif ext in DOCX:
        result = extract_docx(stream)

    elif ext in XLSX:
        result = extract_xlsx(stream)

    elif ext in PPTX:
        result = extract_pptx(stream)

    elif ext in IMAGE:
        result = extract_image(stream)

    elif ext in ARCHIVE:
        result = extract_archive(stream)

    elif ext in CODE:
        result = extract_code(stream, filename)

    if result is not None:
        if not result.get("success") and ext in {".doc", ".xls", ".ppt"}:
            stream.seek(0)
            fallback = extract_code(stream, filename)
            if fallback.get("success"):
                return fallback
        return result

    return {
        "success": False,
        "text": "",
        "metadata": {},
        "error": f"Unsupported file type: {ext}",
    }


def extract_file(file_storage):
    """Flask FileStorage -> standardized extraction result."""
    filename = file_storage.filename or "unknown"
    ext = _ext(filename)
    stream = file_storage.stream
    stream.seek(0)
    return dispatch(stream, filename, ext)
