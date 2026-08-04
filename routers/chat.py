"""
routes/chat.py

API routes for chat handling, supporting message history, 
web search context, and full document uploads with the new AI service.
"""

import os
import json
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.ai_service import ai_service
from services.document_service import document_service
from utils.logger import get_logger

logger = get_logger(__name__)
chat_bp = Blueprint("chat", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint handling text messages, optional file uploads, 
    and routing through the primary/fallback AI pipeline.
    """
    try:
        messages = []
        file_context = ""
        system_prompt = None

        if request.is_json:
            data = request.get_json()
            messages = data.get("messages", [])
            system_prompt = data.get("system_prompt")
        else:
            messages_raw = request.form.get("messages")
            if messages_raw:
                try:
                    messages = json.loads(messages_raw)
                except Exception:
                    messages = []
            
            system_prompt = request.form.get("system_prompt")
            
            # Handle file upload if present
            if "file" in request.files:
                file = request.files["file"]
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(file_path)

                    # Parse file content using the document service
                    parse_result = document_service.parse_file(file_path)
                    if parse_result.get("success"):
                        file_context = parse_result.get("text", "")
                    
                    # Clean up local file after extraction
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

        if not messages:
            return jsonify({"success": False, "error": "No messages provided."}), 400

        # Invoke the main AI orchestration service
        response = ai_service.chat(
            messages=messages,
            file_context=file_context,
            system_prompt=system_prompt,
        )

        return jsonify({
            "success": True,
            "reply": response.content,
            "provider": response.provider,
            "model": response.model,
            "fallback_used": getattr(response, "fallback_used", False)
        }), 200

    except Exception as e:
        logger.error("Error in /chat endpoint: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500