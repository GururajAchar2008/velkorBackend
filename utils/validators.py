"""
utils/validators.py
Request and payload validators.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


DEVICE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,128}$")


def validate_messages(messages: Any) -> Tuple[bool, str]:
    if not isinstance(messages, list) or not messages:
        return False, "No messages provided."
    for m in messages:
        if not isinstance(m, dict):
            return False, "Invalid message format."
        if m.get("role") not in ("user", "assistant", "system"):
            return False, "Invalid message role."
        if not isinstance(m.get("content", ""), str):
            return False, "Invalid message content."
    return True, ""


def validate_device_id(device_id: Optional[str]) -> bool:
    if not device_id:
        return False
    return bool(DEVICE_ID_RE.match(device_id))


def sanitize_filename(name: str) -> str:
    name = (name or "file").strip()
    name = re.sub(r"[^\w.\- ]+", "_", name)
    return name[:200] or "file"
