"""
services/prompt_service.py
Builds the final prompt for AI providers.
"""

from typing import List, Dict
from config import Config
from utils.logger import get_logger

logger=get_logger(__name__)

DEFAULT_SYSTEM_PROMPT="""You are Velkor AI, a helpful, accurate and professional AI assistant.
Prefer uploaded document context when relevant.
Use live search context for current events.
If information is uncertain, clearly say so.
"""

RESEARCH_INSTRUCTIONS="""You are in RESEARCH MODE. The user enabled live web research.
Base your answer on the LIVE SEARCH CONTEXT below and treat it as the latest,
most reliable information. Always cite your sources by listing the source titles
and URLs at the end of your answer. If the search context does not contain the
answer, say so clearly and give your best general answer instead of guessing."""

class PromptService:

    def __init__(self):
        self.max_chars=Config.MAX_CONTEXT_CHARS

    def _history(self,messages:List[Dict])->str:
        lines=[]
        for m in messages:
            role=m.get("role","user").upper()
            content=m.get("content","")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def build(
        self,
        messages:List[Dict],
        file_context:str="",
        web_context:str="",
        system_prompt:str|None=None,
        research_mode:bool=False,
    )->str:

        parts=[]
        if research_mode:
            parts.append(RESEARCH_INSTRUCTIONS)
        parts.append(system_prompt or DEFAULT_SYSTEM_PROMPT)

        if web_context:
            parts.append("=== LIVE SEARCH CONTEXT ===")
            parts.append(web_context[:self.max_chars])
        elif research_mode:
            parts.append("=== LIVE SEARCH CONTEXT ===")
            parts.append("(No live search results could be retrieved. Answer from your own knowledge and clearly note that live results were unavailable.)")

        if file_context:
            parts.append("=== DOCUMENT CONTEXT ===")
            parts.append(file_context[:self.max_chars])

        parts.append("=== CONVERSATION ===")
        parts.append(self._history(messages))

        prompt="\n\n".join(parts)

        logger.info("Prompt built (%d chars)",len(prompt))

        return prompt

prompt_service=PromptService()
