from abc import ABC, abstractmethod
from typing import List, Dict, Optional

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

    @abstractmethod
    def health_check(self) -> bool:
        """
        Returns True if the provider is reachable.
        """
        pass