"""
providers/router.py

Smart provider router for Velkor AI.

Routing order:
    1. NVIDIA NIM    (Primary / Free)     - skipped while rate-limited
    2. OpenRouter    (Fallback / Low-cost)
    3. OpenAI        (Premium / Last-resort) - only when OPENAI_API_KEY set

Falls back on ANY primary-provider error (rate limit, timeout, server
error, malformed response). Switching is invisible to the user.
"""

import time
from collections import deque
from typing import Optional, Callable

from .nvidia import NvidiaProvider
from .openrouter import OpenRouterProvider
from .openai import OpenAIProvider
from .response import AIResponse
from config import Config
from utils.logger import get_logger, log_provider, log_latency

logger = get_logger(__name__)

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class _RateGuard:
    """Sliding-window rate limiter per provider."""

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
    """Primary: NVIDIA NIM. Fallback: OpenRouter. Premium: OpenAI (optional)."""

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

    def _attempt_stream(
        self,
        provider,
        messages,
        system_prompt,
        timeout,
        on_chunk,
        should_stop,
    ):
        if provider is self.primary:
            if not self.nvidia_guard.available():
                return None
            self.nvidia_guard.note_request()

        return provider.generate_stream(
            messages=messages,
            system_prompt=system_prompt,
            timeout=timeout,
            on_chunk=on_chunk,
            should_stop=should_stop,
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
                log_provider(response.provider, response.model, response.response_time)
                log_latency(response.response_time, kind="chat")
                return response

            errors.append(
                f"{provider.provider_name}: {response.error or response.status_code}"
            )
            last_status = response.status_code

            if provider is self.primary and response.status_code == 429:
                self.nvidia_guard.note_rate_limited()
                continue

            # Any primary error → continue to fallback
            logger.warning(
                "Provider %s failed (%s); trying next",
                provider.provider_name,
                response.error or response.status_code,
            )

        return AIResponse(
            success=False,
            reply="",
            provider="router",
            model="none",
            status_code=last_status,
            error=" | ".join(errors),
        )

    def generate_stream(
        self,
        messages,
        system_prompt=None,
        timeout=60,
        on_chunk: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        errors = []
        last_status = 500

        for provider in self._providers_in_order():
            response = self._attempt_stream(
                provider,
                messages,
                system_prompt,
                timeout,
                on_chunk,
                should_stop,
            )

            if response is None:
                errors.append(
                    f"{provider.provider_name}: skipped (rate limit reached)"
                )
                continue

            if response.success:
                if provider is not self.primary:
                    response.fallback_used = True
                log_provider(response.provider, response.model, response.response_time)
                log_latency(response.response_time, kind="chat")
                return response

            errors.append(
                f"{provider.provider_name}: {response.error or response.status_code}"
            )
            last_status = response.status_code

            if provider is self.primary and response.status_code == 429:
                self.nvidia_guard.note_rate_limited()
                continue

            logger.warning(
                "Stream provider %s failed (%s); trying next",
                provider.provider_name,
                response.error or response.status_code,
            )

        return AIResponse(
            success=False,
            reply="",
            provider="router",
            model="none",
            status_code=last_status,
            error=" | ".join(errors),
        )


router = AIRouter()
