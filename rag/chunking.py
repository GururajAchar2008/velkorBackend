"""
rag/chunking.py
Split long documents into overlapping chunks for RAG.
"""

from typing import List

from config import Config


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap if overlap is not None else Config.CHUNK_OVERLAP
    overlap = max(0, min(overlap, chunk_size // 2))

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Prefer breaking on paragraph/sentence boundaries
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep)
                if idx > chunk_size // 3:
                    end = start + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return [c for c in chunks if c]
