from .nvidia import NvidiaProvider
from .openrouter import OpenRouterProvider
from .response import AIResponse

RETRY_STATUS={408,429,500,502,503,504}

class AIRouter:
    """
    Primary: NVIDIA NIM
    Fallback: OpenRouter
    """

    def __init__(self):
        self.primary=NvidiaProvider()
        self.fallback=OpenRouterProvider()

    def generate(self,messages,system_prompt=None,timeout=60)->AIResponse:
        first=self.primary.generate(
            messages=messages,
            system_prompt=system_prompt,
            timeout=timeout,
        )

        if first.success:
            return first

        if first.status_code not in RETRY_STATUS:
            return first

        second=self.fallback.generate(
            messages=messages,
            system_prompt=system_prompt,
            timeout=timeout,
        )

        if second.success:
            second.fallback_used=True
            return second

        return AIResponse(
            success=False,
            reply="",
            provider="router",
            model="none",
            status_code=second.status_code or first.status_code,
            error=f"Primary failed: {first.error} | Fallback failed: {second.error}",
        )

router=AIRouter()
