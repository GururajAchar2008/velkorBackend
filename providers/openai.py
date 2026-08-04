"""
providers/openai.py

OpenAI premium provider for Velkor AI.

Used only as a last-resort premium tier by the router.
No requests are made unless OPENAI_API_KEY is configured.
"""

import os
import time
import requests
from typing import List, Dict, Optional

from .base import AIProvider
from .response import AIResponse
from config import Config


class OpenAIProvider(AIProvider):

    def __init__(self):
        super().__init__(
            api_key=os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY),
            model=os.getenv("OPENAI_MODEL", Config.OPENAI_MODEL),
        )
        self.base_url = os.getenv(
            "OPENAI_BASE_URL",
            Config.OPENAI_BASE_URL,
        ).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def health_check(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        messages: List[Dict],
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ) -> AIResponse:

        if not self.api_key:
            return AIResponse(
                success=False,
                reply="",
                provider=self.provider_name,
                model=self.model,
                status_code=500,
                error="OPENAI_API_KEY missing",
            )

        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                return AIResponse(
                    success=False,
                    reply="",
                    provider=self.provider_name,
                    model=self.model,
                    status_code=resp.status_code,
                    error=resp.text,
                    response_time=elapsed,
                )

            data = resp.json()
            usage = data.get("usage", {})

            return AIResponse(
                success=True,
                reply=data["choices"][0]["message"]["content"],
                provider=self.provider_name,
                model=data.get("model", self.model),
                response_time=elapsed,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                raw=data,
            )

        except requests.Timeout:
            return AIResponse(
                success=False,
                reply="",
                provider=self.provider_name,
                model=self.model,
                status_code=408,
                error="Request timed out",
            )
        except Exception as e:
            return AIResponse(
                success=False,
                reply="",
                provider=self.provider_name,
                model=self.model,
                status_code=500,
                error=str(e),
            )
