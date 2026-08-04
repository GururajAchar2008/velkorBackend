"""
routers/chat.py

API routes for chat handling, supporting message history,
web search context, and full document uploads with the AI service.
"""

import os
import json
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from config import Config
from services.ai_service import ai_service
from services.documents import document_service
from utils.logger import get_logger

logger = get_logger(__name__)
chat_bp = Blueprint("chat", __name__)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


@chat_bp.route("/chat/test", methods=["POST"])
def chat_test():
    """Debug endpoint to test JSON parsing."""
    data = request.get_json(silent=True) or {}
    return jsonify({"received": data, "is_json": request.is_json}), 200


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint handling text messages, optional file uploads,
    and routing through the primary/fallback AI pipeline.
    """
    try:
        messages = []
        file_context = ""
        file_meta = {}
        system_prompt = None
        research_mode = False

        # Parse JSON body (don't rely on request.is_json which can be flaky)
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        system_prompt = data.get("system_prompt")
        research_mode = bool(data.get("research_mode", False))

        # Accept an optional base64 attachment in JSON payloads too.
        file_payload = data.get("file")
        if file_payload and isinstance(file_payload, dict):
            file_context = file_payload.get("text", "")
            file_meta = file_payload.get("metadata", {}) or {}

        if not messages:
            return jsonify({"success": False, "error": "No messages provided."}), 400

        response = ai_service.chat(
            messages=messages,
            file_context=file_context,
            system_prompt=system_prompt,
            research_mode=research_mode,
        )

        if not response.success:
            return jsonify({
                "success": False,
                "error": response.error or "AI providers are currently unavailable.",
                "provider": response.provider,
                "model": response.model,
            }), 502

        return jsonify({
            "success": True,
            "reply": response.reply,
            "provider": response.provider,
            "model": response.model,
            "fallback_used": response.fallback_used,
            "response_time": response.response_time,
            "total_tokens": response.total_tokens,
            "file": file_meta or None,
            "research_performed": response.research_performed,
            "sources": response.sources or [],
        }), 200

    except Exception as e:
        logger.error("Error in /chat endpoint: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500
