import json
import os
import time
from typing import List, Dict, Optional, Callable

import requests

from .base import AIProvider
from .response import AIResponse
from config import Config


class OpenRouterProvider(AIProvider):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        super().__init__(
            api_key=os.getenv("OPENROUTER_API_KEY", Config.OPENROUTER_API_KEY),
            model=os.getenv("OPENROUTER_MODEL", Config.OPENROUTER_MODEL),
        )

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def health_check(self) -> bool:
        return bool(self.api_key)

    def _build_payload(self, messages, system_prompt, stream: bool):
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)
        return {
            "model": self.model,
            "messages": payload_messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "stream": stream,
        }

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://gururajachar2008.github.io/Velkor/",
            "X-Title": "Velkor AI",
        }

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
                error="OPENROUTER_API_KEY missing",
            )

        payload = self._build_payload(messages, system_prompt, stream=False)
        start = time.time()

        try:
            r = requests.post(
                self.API_URL,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            elapsed = time.time() - start

            if r.status_code != 200:
                return AIResponse(
                    success=False,
                    reply="",
                    provider=self.provider_name,
                    model=self.model,
                    status_code=r.status_code,
                    error=r.text,
                    response_time=elapsed,
                )

            data = r.json()
            usage = data.get("usage", {})

            return AIResponse(
                success=True,
                reply=data["choices"][0]["message"]["content"] or "",
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

    def generate_stream(
        self,
        messages: List[Dict],
        timeout: int = 60,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        if not self.api_key:
            return AIResponse(
                success=False,
                reply="",
                provider=self.provider_name,
                model=self.model,
                status_code=500,
                error="OPENROUTER_API_KEY missing",
            )

        payload = self._build_payload(messages, system_prompt, stream=True)
        start = time.time()
        full_text = []

        try:
            with requests.post(
                self.API_URL,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
                stream=True,
            ) as r:
                if r.status_code != 200:
                    return AIResponse(
                        success=False,
                        reply="",
                        provider=self.provider_name,
                        model=self.model,
                        status_code=r.status_code,
                        error=r.text,
                        response_time=time.time() - start,
                    )

                for line in r.iter_lines(decode_unicode=True):
                    if should_stop and should_stop():
                        break
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                    else:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        full_text.append(piece)
                        if on_chunk:
                            on_chunk(piece)

            elapsed = time.time() - start
            reply = "".join(full_text)
            if not reply and not (should_stop and should_stop()):
                return AIResponse(
                    success=False,
                    reply="",
                    provider=self.provider_name,
                    model=self.model,
                    status_code=500,
                    error="Empty streamed response",
                    response_time=elapsed,
                )

            return AIResponse(
                success=True,
                reply=reply,
                provider=self.provider_name,
                model=self.model,
                response_time=elapsed,
            )

        except requests.Timeout:
            return AIResponse(
                success=False,
                reply="".join(full_text),
                provider=self.provider_name,
                model=self.model,
                status_code=408,
                error="Request timed out",
            )
        except Exception as e:
            return AIResponse(
                success=False,
                reply="".join(full_text),
                provider=self.provider_name,
                model=self.model,
                status_code=500,
                error=str(e),
            )
