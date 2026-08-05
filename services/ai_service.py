"""
services/ai_service.py

Main AI orchestration for Velkor AI.
Research search runs ONLY when research_mode is explicitly enabled.
"""

import time
from typing import Callable, Optional

from providers.router import router
from services.search_service import search_service
from services.prompt_service import prompt_service
from services.rag_service import rag_service
from services.memory_service import memory_service
from services.cach_service import cache_service
from config import Config
from utils.logger import get_logger, log_search

logger = get_logger(__name__)


class AIService:
    def chat(
        self,
        messages,
        file_context: str = "",
        system_prompt: str = None,
        research_mode: bool = False,
        device_id: str = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        stream: bool = False,
    ):
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break

        # Memory update from latest user message
        if device_id and query:
            memory_service.extract_and_update(device_id, query)
        memory_block = memory_service.get_context_block(device_id)

        # RAG — only most relevant chunks
        rag_result = rag_service.build_context(query, file_context)
        ranked_docs = rag_result.get("context") or ""

        # Research — toggle only, no auto-detect
        web_context = ""
        sources = []
        if research_mode:
            cache_key = f"search:{query.strip().lower()[:200]}"
            cached = cache_service.get(cache_key)
            if cached:
                web_context = cached.get("web_context", "")
                sources = cached.get("sources", [])
            else:
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
                    log_search(query, len(sources))
                    cache_service.set(
                        cache_key,
                        {"web_context": web_context, "sources": sources},
                        ttl=180,
                    )

        prompt = prompt_service.build(
            messages=messages,
            file_context=ranked_docs,
            web_context=web_context,
            system_prompt=system_prompt,
            research_mode=research_mode,
            memory_context=memory_block,
        )

        final_messages = [{"role": "system", "content": prompt}]
        # Providers need at least one user turn
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        final_messages.append({
            "role": "user",
            "content": last_user or "Please respond based on the conversation context above.",
        })
        start = time.time()

        if stream:
            response = router.generate_stream(
                messages=final_messages,
                system_prompt=None,
                timeout=Config.REQUEST_TIMEOUT,
                on_chunk=on_chunk,
                should_stop=should_stop,
            )
        else:
            response = router.generate(
                messages=final_messages,
                system_prompt=None,
                timeout=Config.REQUEST_TIMEOUT,
            )

        elapsed = time.time() - start
        if not response.response_time:
            response.response_time = elapsed

        response.research_performed = bool(research_mode and sources)
        response.sources = sources
        response.rag_used = bool(rag_result.get("rag_used"))

        return response


ai_service = AIService()
