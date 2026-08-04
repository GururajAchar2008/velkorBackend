"""
files/parsers/archive.py

ZIP archive extractor for Velkor AI.
"""

import io
import zipfile
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".php",
    ".json", ".xml", ".html", ".css", ".sql", ".yaml", ".yml",
    ".csv"
}


def _is_supported(filename: str) -> bool:
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in SUPPORTED_TEXT_EXTENSIONS)


def extract_archive(file_stream) -> Dict:
    """
    Extract supported text files from a ZIP archive.

    Returns:
    {
        success,
        text,
        files,
        metadata,
        error
    }
    """

    try:
        data = file_stream.read()
        archive = zipfile.ZipFile(io.BytesIO(data))

        extracted: List[str] = []
        file_count = 0

        for member in archive.infolist():

            if member.is_dir():
                continue

            if not _is_supported(member.filename):
                continue

            try:
                with archive.open(member) as f:
                    content = f.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                extracted.append(
                    f"\n========== {member.filename} ==========\n"
                )

                extracted.append(content)

                file_count += 1

            except Exception as e:
                logger.warning(
                    "Could not read %s : %s",
                    member.filename,
                    e,
                )

        return {
            "success": True,
            "text": "\n".join(extracted),
            "files": file_count,
            "metadata": {
                "archive_entries": len(archive.infolist())
            },
            "error": None,
        }

    except Exception as e:
        logger.exception("ZIP extraction failed")

        return {
            "success": False,
            "text": "",
            "files": 0,
            "metadata": {},
            "error": str(e),
        }
