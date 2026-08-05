"""
routers/socket_events.py

Socket.IO events for chat streaming and image generation progress.
Keeps long-running work alive across free-tier HTTP timeouts.
"""

import threading
from typing import Dict, Set

from flask import request
from flask_socketio import emit

from services.ai_service import ai_service
from services.image_service import image_service, ImageSafetyError
from services.title_service import title_service
from rag.chunking import chunk_text
from utils.logger import get_logger
from utils.validators import validate_messages

logger = get_logger(__name__)

# sid -> stop flag
_stop_flags: Dict[str, threading.Event] = {}
_active_sids: Set[str] = set()


def _get_stop(sid: str) -> threading.Event:
    if sid not in _stop_flags:
        _stop_flags[sid] = threading.Event()
    return _stop_flags[sid]


def register_socket_events(socketio):
    @socketio.on("connect")
    def on_connect():
        logger.info("Socket connected: %s", request.sid)
        emit("connected", {"ok": True})

    @socketio.on("disconnect")
    def on_disconnect():
        sid = request.sid
        logger.info("Socket disconnected: %s", sid)
        flag = _stop_flags.pop(sid, None)
        if flag:
            flag.set()
        _active_sids.discard(sid)

    @socketio.on("chat:stop")
    def on_chat_stop(data=None):
        sid = request.sid
        _get_stop(sid).set()
        emit("chat:stopped", {"ok": True})

    @socketio.on("chat:start")
    def on_chat_start(data):
        sid = request.sid
        data = data or {}
        stop_flag = _get_stop(sid)
        stop_flag.clear()
        _active_sids.add(sid)

        messages = data.get("messages") or []
        ok, err = validate_messages(messages)
        if not ok:
            emit("chat:error", {"error": err})
            return

        system_prompt = data.get("system_prompt")
        research_mode = bool(data.get("research_mode", False))
        device_id = data.get("device_id")
        file_payload = data.get("file") or {}
        file_context = ""
        if isinstance(file_payload, dict):
            file_context = file_payload.get("text") or ""

        request_id = data.get("request_id") or sid

        def run():
            try:
                def on_chunk(piece: str):
                    socketio.emit(
                        "chat:chunk",
                        {"request_id": request_id, "chunk": piece},
                        to=sid,
                    )

                response = ai_service.chat(
                    messages=messages,
                    file_context=file_context,
                    system_prompt=system_prompt,
                    research_mode=research_mode,
                    device_id=device_id,
                    on_chunk=on_chunk,
                    should_stop=stop_flag.is_set,
                    stream=True,
                )

                if stop_flag.is_set():
                    socketio.emit(
                        "chat:stopped",
                        {"request_id": request_id, "partial": response.reply or ""},
                        to=sid,
                    )
                    return

                if not response.success:
                    socketio.emit(
                        "chat:error",
                        {
                            "request_id": request_id,
                            "error": response.error or "AI providers unavailable.",
                        },
                        to=sid,
                    )
                    return

                title = None
                if data.get("generate_title"):
                    title = title_service.generate(messages)

                socketio.emit(
                    "chat:done",
                    {
                        "request_id": request_id,
                        "reply": response.reply,
                        "text": response.reply,
                        "sources": response.sources or [],
                        "research_performed": response.research_performed,
                        "rag_used": response.rag_used,
                        "response_time": response.response_time,
                        "title": title,
                    },
                    to=sid,
                )
            except Exception as e:
                logger.exception("chat:start failed")
                socketio.emit(
                    "chat:error",
                    {"request_id": request_id, "error": str(e)},
                    to=sid,
                )
            finally:
                _active_sids.discard(sid)

        socketio.start_background_task(run)

    @socketio.on("image:start")
    def on_image_start(data):
        sid = request.sid
        data = data or {}
        prompt = data.get("prompt") or ""
        size = data.get("size") or "1024x1024"
        request_id = data.get("request_id") or sid

        def run():
            try:
                socketio.emit(
                    "image:progress",
                    {"request_id": request_id, "stage": "validating", "progress": 5},
                    to=sid,
                )
                ok, reason = image_service.validate_prompt(prompt)
                if not ok:
                    socketio.emit(
                        "image:error",
                        {"request_id": request_id, "error": reason},
                        to=sid,
                    )
                    return

                socketio.emit(
                    "image:progress",
                    {"request_id": request_id, "stage": "generating", "progress": 25},
                    to=sid,
                )

                # Heartbeat chunks so free-tier proxies don't idle-timeout
                def heartbeat():
                    for pct in (40, 55, 70, 85):
                        socketio.sleep(2)
                        socketio.emit(
                            "image:progress",
                            {
                                "request_id": request_id,
                                "stage": "generating",
                                "progress": pct,
                            },
                            to=sid,
                        )

                socketio.start_background_task(heartbeat)

                result = image_service.generate(prompt=prompt, size=size)
                b64 = result.get("image_b64") or ""

                # Stream image in chunks (reuse chunking helper for payload size)
                parts = chunk_text(b64, chunk_size=12000, overlap=0) if b64 else []
                total = max(len(parts), 1)
                for i, part in enumerate(parts):
                    socketio.emit(
                        "image:chunk",
                        {
                            "request_id": request_id,
                            "index": i,
                            "total": total,
                            "chunk": part,
                        },
                        to=sid,
                    )

                socketio.emit(
                    "image:done",
                    {
                        "request_id": request_id,
                        "image_b64": b64,
                        "mime": result.get("mime", "image/png"),
                        "model": result.get("model"),
                        "response_time": result.get("response_time"),
                    },
                    to=sid,
                )
            except ImageSafetyError as e:
                socketio.emit(
                    "image:error",
                    {"request_id": request_id, "error": str(e), "category": e.category},
                    to=sid,
                )
            except Exception as e:
                logger.exception("image:start failed")
                socketio.emit(
                    "image:error",
                    {"request_id": request_id, "error": str(e)},
                    to=sid,
                )

        socketio.start_background_task(run)
