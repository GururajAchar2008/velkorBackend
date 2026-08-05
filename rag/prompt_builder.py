"""
rag/prompt_builder.py
Assemble ranked document context for the model prompt.
"""

from typing import List

from config import Config


def build_document_context(chunks: List[str], max_chars: int = None) -> str:
    max_chars = max_chars or Config.MAX_CONTEXT_CHARS
    if not chunks:
        return ""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        block = f"[Chunk {i}]\n{chunk}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)
