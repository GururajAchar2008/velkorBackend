"""
services/prompt_service.py
Builds the final prompt for AI providers.
"""

from typing import List, Dict
from config import Config
from utils.logger import get_logger

logger=get_logger(__name__)

DEFAULT_SYSTEM_PROMPT="""
You are Velkor AI, a calm, wise AI teacher created by Gururaj Achar. 
Respond warmly, clearly, and in clean Markdown.
Answer briefly but short and informatively.
Respond only in English.
responce in other laguages only if user requests.
Whenever possible, provide examples, analogies, and step-by-step explanations.
Whenever someone asks for developer contact or 'How to contact your developer' respond with: 'You can reach out to Gururaj Achar on https://gururajachar2008.github.io/Portfolio2.0/'.
Give the developer contact only when asked directly.
Use fenced code blocks only for actual code samples, and keep normal explanation text outside code blocks.
"""

RESEARCH_INSTRUCTIONS="""
Use the search results above as the primary source
for current facts, comparisons, specifications, prices,
recent announcements and live information.
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
