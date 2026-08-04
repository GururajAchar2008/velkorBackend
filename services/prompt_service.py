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
    )->str:

        parts=[]
        parts.append(system_prompt or DEFAULT_SYSTEM_PROMPT)

        if web_context:
            parts.append("=== LIVE SEARCH CONTEXT ===")
            parts.append(web_context[:self.max_chars])

        if file_context:
            parts.append("=== DOCUMENT CONTEXT ===")
            parts.append(file_context[:self.max_chars])

        parts.append("=== CONVERSATION ===")
        parts.append(self._history(messages))

        prompt="\n\n".join(parts)

        logger.info("Prompt built (%d chars)",len(prompt))

        return prompt

prompt_service=PromptService()
