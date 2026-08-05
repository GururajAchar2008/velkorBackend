"""
routers/memory.py
Device-ID keyed memory endpoints.
"""

from flask import Blueprint, request, jsonify

from services.memory_service import memory_service
from utils.validators import validate_device_id
from utils.logger import get_logger

logger = get_logger(__name__)
memory_bp = Blueprint("memory", __name__)


@memory_bp.route("/memory", methods=["GET"])
def get_memory():
    device_id = request.args.get("device_id") or ""
    if not validate_device_id(device_id):
        return jsonify({"success": False, "error": "Invalid device_id."}), 400
    facts = memory_service.get_facts(device_id)
    return jsonify({"success": True, "device_id": device_id, "facts": facts}), 200


@memory_bp.route("/memory", methods=["POST"])
def update_memory():
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id") or ""
    if not validate_device_id(device_id):
        return jsonify({"success": False, "error": "Invalid device_id."}), 400
    facts = data.get("facts") or []
    if not isinstance(facts, list):
        return jsonify({"success": False, "error": "facts must be a list."}), 400
    updated = memory_service.add_facts(device_id, [str(f) for f in facts])
    return jsonify({"success": True, "device_id": device_id, "facts": updated}), 200


@memory_bp.route("/memory", methods=["DELETE"])
def clear_memory():
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id") or request.args.get("device_id") or ""
    if not validate_device_id(device_id):
        return jsonify({"success": False, "error": "Invalid device_id."}), 400
    memory_service.clear(device_id)
    return jsonify({"success": True}), 200
