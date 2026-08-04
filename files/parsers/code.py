"""
files/parsers/code.py

Universal source-code parser for Velkor AI.
"""

from typing import Dict
import logging
import os

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React JSX",
    ".tsx": "React TSX",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".sql": "SQL",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".md": "Markdown",
    ".txt": "Plain Text",
    ".sh": "Shell",
    ".bat": "Batch",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".log": "Log File",
    ".ini": "INI",
    ".toml": "TOML",
    ".cfg": "Config",
    ".conf": "Config",
    ".env": "Environment File",
    ".tex": "LaTeX",
    ".rst": "reStructuredText",
    ".rtf": "Rich Text",
    ".vtt": "Subtitles",
    ".srt": "Subtitles",
}


def extract_code(file_stream, filename: str) -> Dict:
    """
    Extract source code while preserving formatting.
    """

    try:
        ext = os.path.splitext(filename)[1].lower()

        language = LANGUAGE_MAP.get(ext, "Unknown")

        content = file_stream.read().decode(
            "utf-8",
            errors="ignore"
        )

        content = (
            content
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        wrapped = (
            f"Filename: {filename}\n"
            f"Language: {language}\n\n"
            f"{content}"
        )

        return {
            "success": True,
            "text": wrapped,
            "metadata": {
                "language": language,
                "filename": filename,
                "extension": ext,
                "characters": len(content),
                "lines": content.count("\n") + 1,
            },
            "error": None,
        }

    except Exception as e:
        logger.exception("Code extraction failed")

        return {
            "success": False,
            "text": "",
            "metadata": {},
            "error": str(e),
        }
