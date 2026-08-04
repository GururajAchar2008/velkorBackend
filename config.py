"""
config.py
Central configuration for Velkor AI Backend V2.
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
        "deepseek/deepseek-chat-v3-0324"
    )

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL = os.getenv(
        "NVIDIA_MODEL",
        "nvidia/llama-3.3-nemotron-super-49b-v1"
    )

    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

    # ---------- Upload ----------
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    UPLOAD_FOLDER = "storage/uploads"
    TEMP_FOLDER = "storage/temp"

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

    # ---------- Logging ----------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "logs/velkor.log"

    @classmethod
    def validate(cls):
        missing = []

        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")

        if not cls.NVIDIA_API_KEY:
            missing.append("NVIDIA_API_KEY")

        return {
            "valid": len(missing) == 0,
            "missing": missing
        }
