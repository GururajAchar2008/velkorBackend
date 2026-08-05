"""
rag/ranking.py
Lightweight lexical ranking of chunks against a query.
"""

import re
from typing import List, Tuple

from config import Config


_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def score_chunk(query: str, chunk: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    c = _tokens(chunk)
    if not c:
        return 0.0
    overlap = len(q & c)
    # Prefer denser matches and earlier occurrences of query terms
    density = overlap / max(len(q), 1)
    bonus = 0.0
    lower = chunk.lower()
    for term in q:
        pos = lower.find(term)
        if pos >= 0:
            bonus += 1.0 / (1.0 + pos / 200.0)
    return density * 10.0 + bonus + overlap


def rank_chunks(
    query: str,
    chunks: List[str],
    top_k: int = None,
) -> List[str]:
    top_k = top_k or Config.TOP_K_CHUNKS
    if not chunks:
        return []
    scored: List[Tuple[float, int, str]] = []
    for i, chunk in enumerate(chunks):
        scored.append((score_chunk(query, chunk), -i, chunk))
    scored.sort(reverse=True)
    # If all scores are zero, still return leading chunks
    if scored[0][0] <= 0:
        return chunks[:top_k]
    return [c for _, _, c in scored[:top_k]]
