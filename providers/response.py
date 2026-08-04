from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AIResponse:
    """
    Standard response object returned by every AI provider.

    Every provider (NVIDIA, OpenRouter, etc.)
    MUST return this object.

    This allows the router to work without
    caring which provider actually generated
    the response.
    """

    success: bool

    reply: str

    provider: str

    model: str

    status_code: int = 200

    error: Optional[str] = None

    response_time: float = 0.0

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    raw: Optional[Dict[str, Any]] = None

    fallback_used: bool = False

    rag_used: bool = False

    cached: bool = False

    retries: int = 0

    def to_dict(self):
        return {
            "success": self.success,
            "reply": self.reply,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "error": self.error,
            "response_time": self.response_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "fallback_used": self.fallback_used,
            "rag_used": self.rag_used,
            "cached": self.cached,
            "retries": self.retries,
        }