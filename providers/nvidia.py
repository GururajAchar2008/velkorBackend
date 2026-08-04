import os
import time
import requests
from typing import List, Dict, Optional

from .base import AIProvider
from .response import AIResponse
from config import Config


class NvidiaProvider(AIProvider):
    API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self):
        super().__init__(
            api_key=os.getenv("NVIDIA_API_KEY", Config.NVIDIA_API_KEY),
            model=os.getenv("NVIDIA_MODEL", Config.NVIDIA_MODEL),
        )

    @property
    def provider_name(self) -> str:
        return "NVIDIA NIM"

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
                error="NVIDIA_API_KEY missing",
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
            "stream": False,
        }

        start = time.time()
        try:
            resp = requests.post(
                self.API_URL,
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