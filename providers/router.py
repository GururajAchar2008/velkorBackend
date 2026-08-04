"""
providers/router.py

Smart provider router for Velkor AI.

Routing order (cost-first):
    1. NVIDIA NIM    (Primary / Free)     - skipped while rate-limited
    2. OpenRouter    (Fallback / Low-cost)
    3. OpenAI        (Premium / Last-resort) - only when OPENAI_API_KEY set

NVIDIA's free tier is rate-limited (~40 RPM). The router tracks a sliding
60-second window plus a cooldown after a 429, and automatically sends
overflow traffic to OpenRouter so users never wait behind a rate limit.
"""

import time
from collections import deque

from .nvidia import NvidiaProvider
from .openrouter import OpenRouterProvider
from .openai import OpenAIProvider
from .response import AIResponse
from config import Config


RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class _RateGuard:
    """
    Sliding-window rate limiter per provider.

    Tracks how many requests were sent in the last 60 seconds and adds a
    temporary cooldown whenever the provider reports a 429 (rate limit).
    """

    WINDOW_SECONDS = 60

    def __init__(self, rpm_limit: int, cooldown_seconds: int = 60):
        self.rpm_limit = max(rpm_limit, 1)
        self.cooldown_seconds = cooldown_seconds
        self._timestamps: deque = deque()
        self._cooldown_until = 0.0

    def _prune(self):
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > self.WINDOW_SECONDS:
            self._timestamps.popleft()

    def available(self) -> bool:
        now = time.monotonic()
        if now < self._cooldown_until:
            return False
        self._prune()
        return len(self._timestamps) < self.rpm_limit

    def note_request(self):
        self._prune()
        self._timestamps.append(time.monotonic())

    def note_rate_limited(self):
        self._cooldown_until = time.monotonic() + self.cooldown_seconds


class AIRouter:
    """
    Primary: NVIDIA NIM
    Fallback: OpenRouter
    Premium: OpenAI (optional)
    """

    def __init__(self):
        self.primary = NvidiaProvider()
        self.fallback = OpenRouterProvider()
        self.premium = OpenAIProvider()

        self.nvidia_guard = _RateGuard(
            Config.NVIDIA_RPM_LIMIT,
            Config.NVIDIA_COOLDOWN_SECONDS,
        )

    def _providers_in_order(self):
        order = [self.primary, self.fallback]
        if self.premium.health_check():
            order.append(self.premium)

        if not self.primary.health_check():
            order = [p for p in order if p is not self.primary]

        return order

    def _attempt(self, provider, messages, system_prompt, timeout):
        if provider is self.primary:
            if not self.nvidia_guard.available():
                return None
            self.nvidia_guard.note_request()

        return provider.generate(
            messages=messages,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    def generate(self, messages, system_prompt=None, timeout=60) -> AIResponse:
        errors = []
        last_status = 500

        for provider in self._providers_in_order():
            response = self._attempt(provider, messages, system_prompt, timeout)

            if response is None:
                errors.append(
                    f"{provider.provider_name}: skipped (rate limit reached)"
                )
                continue

            if response.success:
                if provider is not self.primary:
                    response.fallback_used = True
                return response

            errors.append(
                f"{provider.provider_name}: {response.error or response.status_code}"
            )
            last_status = response.status_code

            if provider is self.primary and response.status_code == 429:
                self.nvidia_guard.note_rate_limited()
                # Stop hitting NVIDIA for the cooldown window; keep descending.
                continue

        return AIResponse(
            success=False,
            reply="",
            provider="router",
            model="none",
            status_code=last_status,
            error=" | ".join(errors),
        )


router = AIRouter()
