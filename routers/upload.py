"""
routers/upload.py

API route for file uploads. Parses the uploaded file into text,
stores metadata, and returns a file context for the AI service.
"""

import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from config import Config
from files.extractor import extract_text
from utils.logger import get_logger

logger = get_logger(__name__)
upload_bp = Blueprint("upload", __name__)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload and parse a document. Returns extracted text and metadata
    that can be sent as file_context to the /api/chat endpoint.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename."}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > Config.MAX_UPLOAD_BYTES:
        return jsonify({
            "success": False,
            "error": f"File too large. Maximum size is {Config.MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        }), 413

    try:
        # Save the file temporarily
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info("File uploaded: %s (%d bytes)", filename, file_size)

        # Extract text
        extracted = extract_text(file_path)

        return jsonify({
            "success": True,
            "filename": filename,
            "size": file_size,
            "text": extracted.get("text", ""),
            "metadata": extracted.get("metadata", {}),
            "pages": extracted.get("pages", 0),
            "preview": (extracted.get("text", "") or "")[:500],
        }), 200

    except Exception as e:
        logger.error("Error processing file %s: %s", filename, str(e))
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500
    finally:
        # Clean up the uploaded file
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
