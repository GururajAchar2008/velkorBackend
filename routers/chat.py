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

        if request.is_json:
            data = request.get_json(silent=True) or {}
            messages = data.get("messages", [])
            system_prompt = data.get("system_prompt")

            # Accept an optional base64 attachment in JSON payloads too.
            file_payload = data.get("file")
            if file_payload and isinstance(file_payload, dict):
                file_context = file_payload.get("text", "")
                file_meta = file_payload.get("metadata", {}) or {}
        else:
            messages_raw = request.form.get("messages")
            if messages_raw:
                try:
                    messages = json.loads(messages_raw)
                except Exception:
                    messages = []

            system_prompt = request.form.get("system_prompt")

            file = request.files.get("file")
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                file.save(file_path)

                parse_result = document_service.parse_file(file_path)

                try:
                    os.remove(file_path)
                except Exception:
                    pass

                if parse_result.get("success"):
                    file_context = parse_result.get("text", "")
                    file_meta = {
                        "name": filename,
                        "type": parse_result.get("type", ""),
                        **parse_result.get("metadata", {}),
                    }
                else:
                    return jsonify({
                        "success": False,
                        "error": f"Could not read the uploaded file: {parse_result.get('error', 'unknown error')}"
                    }), 400

        if not messages:
            return jsonify({"success": False, "error": "No messages provided."}), 400

        response = ai_service.chat(
            messages=messages,
            file_context=file_context,
            system_prompt=system_prompt,
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
        }), 200

    except Exception as e:
        logger.error("Error in /chat endpoint: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500
