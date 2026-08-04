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
    import json as json_lib
    raw = request.get_data(as_text=True)
    try:
        data = json_lib.loads(raw) if raw else {}
    except Exception as e:
        data = {"parse_error": str(e), "raw": raw}
    return jsonify({"received": data, "is_json": request.is_json, "content_type": request.content_type, "raw_len": len(raw)}), 200


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint handling text messages, optional file uploads,
    and routing through the primary/fallback AI pipeline.
    """
    import json as json_lib
    try:
        # Get raw request data first
        raw_data = request.get_data(as_text=True)
        if not raw_data:
            return jsonify({"success": False, "error": "No request body provided."}), 400

        # Parse JSON with error handling
        try:
            data = json_lib.loads(raw_data)
        except json_lib.JSONDecodeError as e:
            return jsonify({"success": False, "error": f"Invalid JSON: {str(e)}"}), 400

        # Extract fields from JSON data
        messages = data.get("messages", [])
        system_prompt = data.get("system_prompt")
        research_mode = bool(data.get("research_mode", False))

        # Handle file payload
        file_payload = data.get("file")
        file_context = ""
        file_meta = {}
        if file_payload and isinstance(file_payload, dict):
            file_context = file_payload.get("text", "")
            file_meta = file_payload.get("metadata", {}) or {}

        # Validate messages
        if not messages:
            return jsonify({"success": False, "error": "No messages provided."}), 400

        # Call AI service
        response = ai_service.chat(
            messages=messages,
            file_context=file_context,
            system_prompt=system_prompt,
            research_mode=research_mode,
        )

        # Handle AI service errors
        if not response.success:
            return jsonify({
                "success": False,
                "error": response.error or "AI providers are currently unavailable.",
                "provider": response.provider,
                "model": response.model,
            }), 502

        # Return successful response
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
