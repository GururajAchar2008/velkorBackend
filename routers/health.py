"""
routers/health.py

Health-check endpoint that reports provider availability.
"""

from flask import Blueprint, jsonify

from providers.router import router
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
health_bp = Blueprint("health", __name__)


def _payload():
    status = Config.validate()
    return {
        "status": "ok" if status["valid"] else "degraded",
        "service": "Velkor AI Backend",
        "version": "3.0",
        "config": status,
        "latency_target_s": Config.CHAT_LATENCY_TARGET_S,
        "providers": {
            "nvidia": {
                "configured": router.primary.health_check(),
                "model": Config.NVIDIA_MODEL,
                "rate_available": router.nvidia_guard.available(),
            },
            "openrouter": {
                "configured": router.fallback.health_check(),
                "model": Config.OPENROUTER_MODEL,
            },
            "openai": {
                "configured": router.premium.health_check(),
            },
            "image": {
                "configured": bool(Config.NVIDIA_API_KEY),
                "generation_model": Config.IMAGE_GEN_MODEL,
                "edit_model": Config.IMAGE_EDIT_MODEL,
            },
        },
    }


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify(_payload()), 200


@health_bp.route("/api/health", methods=["GET"])
def health_api():
    return jsonify(_payload()), 200
