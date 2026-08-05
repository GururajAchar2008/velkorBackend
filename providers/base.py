from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Iterator, Callable

from .response import AIResponse


class AIProvider(ABC):
    """
    Base class for every AI provider.

    All providers MUST inherit this class.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict],
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ) -> AIResponse:
        """
        Generate a chat completion.

        Must always return AIResponse.
        """
        pass

    def generate_stream(
        self,
        messages: List[Dict],
        timeout: int = 60,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        """
        Stream a chat completion. Default falls back to non-streaming
        and emits the full reply as a single chunk.
        """
        response = self.generate(
            messages=messages,
            timeout=timeout,
            system_prompt=system_prompt,
        )
        if response.success and response.reply and on_chunk:
            if not (should_stop and should_stop()):
                on_chunk(response.reply)
        return response

    @abstractmethod
    def health_check(self) -> bool:
        """
        Returns True if the provider is reachable.
        """
        pass
