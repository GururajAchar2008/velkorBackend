"""
rag/embeddings.py
Placeholder for future embedding / vector-DB integration.
Lexical ranking is used at launch; this module keeps the extension point.
"""

from typing import List


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Not wired at launch — returns empty vectors."""
    return [[] for _ in texts]
