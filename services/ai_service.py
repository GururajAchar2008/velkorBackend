"""
services/ai_service.py

Main AI orchestration service for Velkor AI.
Handles provider routing, search integration, and prompt assembly.
"""

import time
from providers.router import router
from services.search_service import search_service
from services.prompt_service import prompt_service
from config import Config
from utils.logger import get_logger, log_provider

logger = get_logger(__name__)

SEARCH_HINTS = [
    "latest", "today", "current", "news", "recent",
    "price", "compare", "vs", "2025", "2026"
]

class AIService:

    def needs_search(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in SEARCH_HINTS)

    def chat(
        self,
        messages,
        file_context: str = "",
        system_prompt: str = None,
    ):
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break

        web_context = ""
        if self.needs_search(query):
            result = search_service.search(query)
            if result.get("success"):
                web_context = "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('url', '')}"
                    for r in result.get("results", [])
                )

        prompt = prompt_service.build(
            messages=messages,
            file_context=file_context,
            web_context=web_context,
            system_prompt=system_prompt,
        )

        final_messages = [
            {
                "role": "system",
                "content": prompt
            }
        ]

        start = time.time()

        # Calls the unified router (NVIDIA NIM primary -> OpenRouter fallback)
        response = router.generate(
            messages=final_messages,
            system_prompt=None,
            timeout=Config.REQUEST_TIMEOUT,
        )

        elapsed = time.time() - start

        log_provider(
            response.provider,
            response.model,
            elapsed
        )

        return response

ai_service = AIService()