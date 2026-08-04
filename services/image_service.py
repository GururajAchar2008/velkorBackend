"""
services/image_service.py

NVIDIA NIM hosted image generation and editing for Velkor AI.

Every prompt is checked against a strict content-safety policy before it
reaches the model. Requests that reference explicit sexual content, minors,
self-harm, hate speech, violence, weapons, drugs, illegal acts, terrorism or
personal data are rejected with a clear policy message. This is a best-effort
compliance layer that complements the model provider's own safety systems;
operators remain responsible for meeting the laws of their jurisdiction.
"""

import base64
import time
from typing import Dict, Any, Tuple

import requests

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

API_BASE = "https://integrate.api.nvidia.com/v1"

BLOCKED_CATEGORIES = {
    "explicit sexual content": [
        "nude", "naked", "porn", "xxx", "erotic", "nsfw", "sex",
        "sexual", "hentai", "boobs", "penis", "vagina", "strip", "explicit",
    ],
    "minors": ["child", "minor", "kid", "teen", "underage", "schoolgirl", "schoolboy"],
    "self-harm": ["self harm", "suicide", "cutting myself", "kill myself"],
    "violence and gore": ["gore", "blood", "mutilat", "torture", "murder", "dead body", "corpse"],
    "hate speech": ["nazi", "racist", "racism", "kkk", "white power", "hate speech"],
    "illegal activity": ["illegal", "steal", "stolen", "fraud", "counterfeit", "credit card number"],
    "weapons": ["gun", "rifle", "pistol", "bomb", "explosive", "knife", "weapon", "ammo"],
    "drugs": ["cocaine", "heroin", "meth", "marijuana", "mdma", "illegal drugs"],
    "terrorism": ["terrorist", "isis", "bombing a", "attack plan"],
    "personal data": ["passport", "aadhaar", "ssn", "pan card", "identity card", "driver licence"],
}


class ImageSafetyError(Exception):
    """Raised when a prompt violates policy or the provider fails."""

    def __init__(self, message: str, category: str = "policy"):
        self.category = category
        super().__init__(message)


class ImageService:

    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        text = (prompt or "").lower()
        if not text.strip():
            return False, "Please describe the image you want to create."
        for category, terms in BLOCKED_CATEGORIES.items():
            for term in terms:
                if term in text:
                    return False, self._policy_message(category)
        return True, ""

    def _policy_message(self, category: str) -> str:
        return (
            "This request was blocked by Velkor AI's content-safety policy. "
            "Velkor only generates appropriate, lawful, family-friendly images. "
            f"(Blocked category: {category})"
        )

    def _require_key(self) -> str:
        if not Config.NVIDIA_API_KEY:
            raise ImageSafetyError(
                "Image generation is not configured on the server.",
                "config",
            )
        return Config.NVIDIA_API_KEY

    def _sanitize_prompt(self, prompt: str) -> str:
        return (prompt or "").strip()[: Config.IMAGE_MAX_PROMPT_CHARS]

    def generate(self, prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
        ok, reason = self.validate_prompt(prompt)
        if not ok:
            raise ImageSafetyError(reason, "policy")
        key = self._require_key()

        start = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/images/generations",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": Config.IMAGE_GEN_MODEL,
                    "prompt": self._sanitize_prompt(prompt),
                    "response_format": "b64_json",
                    "size": size,
                },
                timeout=Config.REQUEST_TIMEOUT,
            )
        except requests.Timeout:
            raise ImageSafetyError("Image generation timed out. Please try again.", "provider")
        except Exception as e:
            logger.error("Image generation request failed: %s", e)
            raise ImageSafetyError("Image generation failed. Please try again.", "provider")

        elapsed = time.time() - start
        if resp.status_code != 200:
            logger.error("NVIDIA image generation failed: %s %s", resp.status_code, resp.text)
            raise ImageSafetyError(
                "The image provider returned an error. Please try again.",
                "provider",
            )

        payload = resp.json()
        b64 = self._extract_b64(payload)
        if not b64:
            raise ImageSafetyError("The image provider returned an empty result.", "provider")

        return {
            "success": True,
            "image_b64": b64,
            "mime": "image/png",
            "model": Config.IMAGE_GEN_MODEL,
            "provider": "NVIDIA NIM",
            "response_time": elapsed,
        }

    def edit(self, prompt: str, image_bytes: bytes, image_filename: str = "image.png") -> Dict[str, Any]:
        ok, reason = self.validate_prompt(prompt)
        if not ok:
            raise ImageSafetyError(reason, "policy")
        key = self._require_key()

        start = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/images/edits",
                headers={"Authorization": f"Bearer {key}"},
                files={"image": (image_filename, image_bytes)},
                data={
                    "model": Config.IMAGE_EDIT_MODEL,
                    "prompt": self._sanitize_prompt(prompt),
                    "response_format": "b64_json",
                },
                timeout=Config.REQUEST_TIMEOUT * 2,
            )
        except requests.Timeout:
            raise ImageSafetyError("Image editing timed out. Please try again.", "provider")
        except Exception as e:
            logger.error("Image edit request failed: %s", e)
            raise ImageSafetyError("Image editing failed. Please try again.", "provider")

        elapsed = time.time() - start
        if resp.status_code != 200:
            logger.error("NVIDIA image edit failed: %s %s", resp.status_code, resp.text)
            raise ImageSafetyError(
                "The image provider returned an error. Please try again.",
                "provider",
            )

        payload = resp.json()
        b64 = self._extract_b64(payload)
        if not b64:
            raise ImageSafetyError("The image provider returned an empty result.", "provider")

        return {
            "success": True,
            "image_b64": b64,
            "mime": "image/png",
            "model": Config.IMAGE_EDIT_MODEL,
            "provider": "NVIDIA NIM",
            "response_time": elapsed,
        }

    @staticmethod
    def _extract_b64(payload: Dict[str, Any]) -> str:
        """Extract base64 from either the OpenAI-compatible format
        (data[].b64_json) or the legacy NIM format (artifacts[].base64)."""
        data = payload.get("data") or []
        if data and isinstance(data, list):
            b64 = (data[0] or {}).get("b64_json")
            if b64:
                return b64
        artifacts = payload.get("artifacts") or []
        if artifacts and isinstance(artifacts, list):
            b64 = (artifacts[0] or {}).get("base64")
            if b64:
                return b64
        return ""


image_service = ImageService()
