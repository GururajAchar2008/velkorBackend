"""
routers/chat.py

API routes for chat handling, supporting message history,
research-mode web search, RAG document context, device memory,
and real-time HTTP SSE streaming.
"""

import json
import queue
import threading
from flask import Blueprint, request, jsonify, Response, stream_with_context

from services.ai_service import ai_service
from services.title_service import title_service
from utils.logger import get_logger
from utils.validators import validate_messages

logger = get_logger(__name__)
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat/stream", methods=["POST", "OPTIONS"])
@chat_bp.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    Main chat endpoint. Supports both JSON response and real-time SSE streaming.
    """
    if request.method == "OPTIONS":
        return "", 200

    try:
        raw_data = request.get_data(as_text=True)
        if not raw_data:
            return jsonify({"success": False, "error": "No request body provided."}), 400

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            return jsonify({"success": False, "error": f"Invalid JSON: {str(e)}"}), 400

        messages = data.get("messages", [])
        ok, err = validate_messages(messages)
        if not ok:
            return jsonify({"success": False, "error": err}), 400

        system_prompt = data.get("system_prompt")
        research_mode = bool(data.get("research_mode", False))
        device_id = data.get("device_id")
        stream_requested = bool(
            data.get("stream", False)
            or request.path.endswith("/stream")
            or "text/event-stream" in request.headers.get("Accept", "")
        )

        file_payload = data.get("file")
        file_context = ""
        file_meta = {}
        if file_payload and isinstance(file_payload, dict):
            file_context = file_payload.get("text", "") or ""
            file_meta = file_payload.get("metadata", {}) or {}

        if stream_requested:
            def generate_sse():
                q = queue.Queue()

                def on_chunk(piece: str):
                    if piece:
                        q.put(("chunk", piece))

                def worker():
                    try:
                        response = ai_service.chat(
                            messages=messages,
                            file_context=file_context,
                            system_prompt=system_prompt,
                            research_mode=research_mode,
                            device_id=device_id,
                            on_chunk=on_chunk,
                            stream=True,
                        )
                        title = None
                        if data.get("generate_title"):
                            title = title_service.generate(messages)

                        q.put(("done", {
                            "success": response.success,
                            "reply": response.reply,
                            "text": response.reply,
                            "provider": response.provider,
                            "model": response.model,
                            "response_time": response.response_time,
                            "total_tokens": response.total_tokens,
                            "file": file_meta or None,
                            "research_performed": response.research_performed,
                            "sources": response.sources or [],
                            "rag_used": response.rag_used,
                            "title": title,
                            "error": response.error if not response.success else None,
                        }))
                    except Exception as ex:
                        q.put(("error", str(ex)))

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                while True:
                    kind, payload = q.get()
                    if kind == "chunk":
                        yield f"data: {json.dumps({'chunk': payload})}\n\n"
                    elif kind == "done":
                        yield f"data: {json.dumps({'done': True, 'payload': payload})}\n\n"
                        break
                    elif kind == "error":
                        yield f"data: {json.dumps({'error': payload})}\n\n"
                        break

            return Response(
                stream_with_context(generate_sse()),
                mimetype="text/event-stream",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            )

        # Standard non-streaming path
        response = ai_service.chat(
            messages=messages,
            file_context=file_context,
            system_prompt=system_prompt,
            research_mode=research_mode,
            device_id=device_id,
            stream=False,
        )

        if not response.success:
            return jsonify({
                "success": False,
                "error": response.error or "AI providers are currently unavailable.",
                "provider": response.provider,
                "model": response.model,
            }), 502

        title = None
        if data.get("generate_title"):
            title = title_service.generate(messages)

        return jsonify({
            "success": True,
            "reply": response.reply,
            "text": response.reply,
            "provider": response.provider,
            "model": response.model,
            "fallback_used": response.fallback_used,
            "response_time": response.response_time,
            "total_tokens": response.total_tokens,
            "file": file_meta or None,
            "research_performed": response.research_performed,
            "sources": response.sources or [],
            "rag_used": response.rag_used,
            "title": title,
        }), 200

    except Exception as e:
        logger.error("Error in /chat endpoint: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@chat_bp.route("/title", methods=["POST"])
def title():
    try:
        data = request.get_json(silent=True) or {}
        messages = data.get("messages") or []
        text = data.get("text") or ""
        if not messages and text:
            messages = [{"role": "user", "content": text}]
        generated = title_service.generate(messages)
        return jsonify({"success": True, "title": generated}), 200
    except Exception as e:
        logger.error("Error in /title: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
