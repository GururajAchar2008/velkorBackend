"""
config.py
Central configuration for Velkor AI Backend.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---------- Flask ----------
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    SECRET_KEY = os.getenv("SECRET_KEY", "velkor-dev-secret")

    # ---------- AI Providers ----------
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "deepseek/deepseek-chat-v3-0324",
    )

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL = os.getenv(
        "NVIDIA_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b",
    )
    # Free-tier NVIDIA NIM is capped (~40 RPM). Router auto-falls back to
    # OpenRouter when the window is full or after a 429 cooldown.
    NVIDIA_RPM_LIMIT = int(os.getenv("NVIDIA_RPM_LIMIT", "40"))
    NVIDIA_COOLDOWN_SECONDS = int(os.getenv("NVIDIA_COOLDOWN_SECONDS", "60"))

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

    # ---------- Upload ----------
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    MAX_UPLOAD_BYTES = 100 * 1024 * 1024
    UPLOAD_FOLDER = "storage/uploads"
    TEMP_FOLDER = "storage/temp"
    MEMORY_FOLDER = "storage/memory"

    # ---------- Image generation / editing (NVIDIA NIM) ----------
    IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "flux.1-schnell")
    IMAGE_EDIT_MODEL = os.getenv("IMAGE_EDIT_MODEL", "qwen-image-edit")
    IMAGE_MAX_PROMPT_CHARS = 4000
    MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024

    # ---------- RAG ----------
    MAX_CONTEXT_CHARS = 12000
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150
    TOP_K_CHUNKS = 5

    # ---------- AI ----------
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    CHAT_LATENCY_TARGET_S = 3.0

    # ---------- Logging ----------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "logs/velkor.log"

    @classmethod
    def validate(cls):
        """Check that at least one AI provider is configured."""
        missing = []

        if not cls.NVIDIA_API_KEY and not cls.OPENROUTER_API_KEY and not cls.OPENAI_API_KEY:
            missing.extend(["NVIDIA_API_KEY", "OPENROUTER_API_KEY"])

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "providers": {
                "nvidia": bool(cls.NVIDIA_API_KEY),
                "openrouter": bool(cls.OPENROUTER_API_KEY),
                "openai": bool(cls.OPENAI_API_KEY),
            },
        }
