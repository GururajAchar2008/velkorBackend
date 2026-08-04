"""
routers/models.py

Lists the models the backend is configured to use.
"""

from flask import Blueprint, jsonify

from config import Config

models_bp = Blueprint("models", __name__)


@models_bp.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "success": True,
        "models": {
            "primary": {
                "provider": "NVIDIA NIM",
                "model": Config.NVIDIA_MODEL,
            },
            "fallback": {
                "provider": "OpenRouter",
                "model": Config.OPENROUTER_MODEL,
            },
            "premium": {
                "provider": "OpenAI",
                "model": Config.OPENAI_MODEL,
                "configured": bool(Config.OPENAI_API_KEY),
            },
        },
    }), 200
