"""
services/title_service.py
Auto-generate short conversation titles.
"""

import re
from typing import List, Dict, Optional

from providers.router import router
from utils.logger import get_logger

logger = get_logger(__name__)


class TitleService:
    def from_heuristic(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return "New chat"
        # Prefer first sentence-ish
        for sep in (".", "?", "!", "\n"):
            idx = text.find(sep)
            if 8 <= idx <= 60:
                text = text[:idx]
                break
        if len(text) > 56:
            text = text[:53].rstrip() + "…"
        return text[0].upper() + text[1:] if text else "New chat"

    def generate(self, messages: List[Dict], use_model: bool = False) -> str:
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        title = self.from_heuristic(query)
        if not use_model or not query:
            return title

        try:
            prompt = (
                "Generate a very short chat title (max 6 words) for this user message. "
                "Return ONLY the title, no quotes or punctuation fluff.\n\n"
                f"Message: {query[:400]}"
            )
            response = router.generate(
                messages=[{"role": "user", "content": prompt}],
                timeout=15,
            )
            if response.success and response.reply:
                cleaned = re.sub(r'["\']', "", response.reply).strip().split("\n")[0]
                cleaned = re.sub(r"\s+", " ", cleaned)[:60]
                if len(cleaned) >= 3:
                    return cleaned
        except Exception as e:
            logger.warning("Title model failed: %s", e)
        return title


title_service = TitleService()
