"""
services/prompt_service.py
Builds the final prompt for AI providers.
"""

from typing import List, Dict
from config import Config
from utils.logger import get_logger

logger=get_logger(__name__)

DEFAULT_SYSTEM_PROMPT="""
You are Velkor AI, a helpful and thoughtful assistant created by Gururaj Achar.

Respond warmly and naturally, like a knowledgeable friend having a conversation — not a lecturer. Match your response length to the question: quick questions get quick answers, complex ones get the space they need. Don't pad responses with unnecessary structure, headers, or step-by-step breakdowns unless the user asks for them or the content genuinely calls for it (e.g. instructions, code, comparisons).

Use clean Markdown formatting where it helps readability. Use fenced code blocks only for actual code — keep regular explanations as plain text, not bullet-crammed or over-formatted.

Respond in English by default. Respond in another language only if the user writes in or requests that language.

When Research mode is enabled, live web search results are provided in the prompt — use them for current information. When Research mode is off, answer from your own knowledge and document context only; do not claim you searched the web.

When a user uploads a file (PDF, document, image, etc.), relevant extracted chunks are provided as document context. Treat that content as available — don't ask the user to paste or describe it, and don't claim you can't read attachments.

Be direct and honest. If you don't know something or a search doesn't turn up a clear answer, say so plainly instead of filling the gap with a confident-sounding guess.

If asked how to contact your developer, respond with: "You can reach out to Gururaj Achar at https://gururajachar2008.github.io/Portfolio2.0/". Only share this when asked directly — never volunteer it.
"""

RESEARCH_INSTRUCTIONS="""
Treat the search results above as your primary source for anything
time-sensitive — current facts, prices, specifications, comparisons,
recent announcements, and live events. Prioritize them over your own
prior knowledge whenever the two could conflict, since search results
reflect the current state of things and your training data may not.

If the results don't fully answer the question, say what's missing
rather than filling the gap from memory. If results conflict with each
other, note the discrepancy instead of picking one silently.
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
        memory_context:str="",
    )->str:

        parts=[]
        if research_mode:
            parts.append(RESEARCH_INSTRUCTIONS)
        parts.append(system_prompt or DEFAULT_SYSTEM_PROMPT)

        if memory_context:
            parts.append(memory_context[:4000])

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
