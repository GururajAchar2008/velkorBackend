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


@health_bp.route("/health", methods=["GET"])
def health():
    status = Config.validate()
    return jsonify({
        "status": "ok" if status["valid"] else "degraded",
        "service": "Velkor AI / Guru JI AI Backend",
        "version": "2.0",
        "config": status,
        "providers": {
            "nvidia": {
                "configured": router.primary.health_check(),
                "rate_available": router.nvidia_guard.available(),
            },
            "openrouter": {
                "configured": router.fallback.health_check(),
            },
            "openai": {
                "configured": router.premium.health_check(),
            },
        },
    }), 200
