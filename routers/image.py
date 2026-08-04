"""
routers/image.py

API routes for AI image generation and editing.
Both endpoints run prompts through a strict content-safety filter.
"""

import base64

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from config import Config
from services.image_service import image_service, ImageSafetyError
from utils.logger import get_logger

logger = get_logger(__name__)
image_bp = Blueprint("image", __name__)


@image_bp.route("/image/generate", methods=["POST"])
def generate():
    """
    Generate an image from a text prompt.
    Expects JSON: {"prompt": "...", "size": "1024x1024"}
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        size = data.get("size", "1024x1024")

        result = image_service.generate(prompt=prompt, size=size)

        return jsonify(result), 200

    except ImageSafetyError as e:
        return jsonify({
            "success": False,
            "status_code": 400,
            "error": str(e),
            "category": e.category,
        }), 400

    except Exception as e:
        logger.error("Error in /image/generate: %s", str(e))
        return jsonify({
            "success": False,
            "status_code": 500,
            "error": "Internal error while generating the image.",
        }), 500


@image_bp.route("/image/edit", methods=["POST"])
def edit():
    """
    Edit an uploaded image using a text prompt.
    Expects either JSON: {"prompt": "...", "image_base64": "..."} or
    multipart form with a "prompt" field and "image" file upload.
    """
    try:
        prompt = ""
        image_bytes = None
        image_filename = "image.png"

        if request.is_json:
            data = request.get_json(silent=True) or {}
            prompt = data.get("prompt", "")
            image_base64 = data.get("image_base64", "")
            if not image_base64:
                return jsonify({
                    "success": False,
                    "status_code": 400,
                    "error": "No image provided.",
                }), 400
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                return jsonify({
                    "success": False,
                    "status_code": 400,
                    "error": "Invalid image data.",
                }), 400
        else:
            prompt = request.form.get("prompt", "")
            file = request.files.get("image")
            if not file or not file.filename:
                return jsonify({
                    "success": False,
                    "status_code": 400,
                    "error": "No image provided.",
                }), 400
            image_filename = secure_filename(file.filename) or "image.png"
            image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "status_code": 400,
                "error": "No image provided.",
            }), 400

        if len(image_bytes) > Config.MAX_IMAGE_UPLOAD_BYTES:
            return jsonify({
                "success": False,
                "status_code": 400,
                "error": "The uploaded image is too large (max 20 MB).",
            }), 400

        result = image_service.edit(
            prompt=prompt,
            image_bytes=image_bytes,
            image_filename=image_filename,
        )

        return jsonify(result), 200

    except ImageSafetyError as e:
        return jsonify({
            "success": False,
            "status_code": 400,
            "error": str(e),
            "category": e.category,
        }), 400

    except Exception as e:
        logger.error("Error in /image/edit: %s", str(e))
        return jsonify({
            "success": False,
            "status_code": 500,
            "error": "Internal error while editing the image.",
        }), 500
