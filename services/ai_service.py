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
        research_mode: bool = False,
    ):
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break

        web_context = ""
        sources = []
        should_search = research_mode or self.needs_search(query)
        if should_search:
            result = search_service.search(query)
            if result.get("success"):
                web_context = "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('url', '')}"
                    for r in result.get("results", [])
                )
                sources = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                    }
                    for r in result.get("results", [])
                    if r.get("url")
                ]

        prompt = prompt_service.build(
            messages=messages,
            file_context=file_context,
            web_context=web_context,
            system_prompt=system_prompt,
            research_mode=research_mode,
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

        response.research_performed = bool(sources)
        response.sources = sources

        return response

ai_service = AIService()