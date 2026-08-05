"""
routers/upload.py

API route for file uploads. Parses the uploaded file into text,
stores metadata, and returns a file context for the AI service.
"""

from flask import Blueprint, request, jsonify

from services.upload_service import upload_service
from utils.logger import get_logger

logger = get_logger(__name__)
upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload and parse a document. Returns extracted text and metadata
    that can be sent as file_context to the /api/chat endpoint.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]
    result = upload_service.process(file)
    status = result.pop("status_code", None) or (200 if result.get("success") else 400)
    return jsonify(result), status
