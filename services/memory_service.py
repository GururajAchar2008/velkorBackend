"""
services/memory_service.py

Device-ID keyed personalization store.
No login — memory is keyed to a durable client-generated device/browser ID.
"""

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from config import Config
from utils.logger import get_logger
from utils.validators import validate_device_id

logger = get_logger(__name__)

_FACT_PATTERNS = [
    re.compile(r"(?:my name is|i'?m|i am)\s+([A-Z][a-zA-Z\-']{1,40})", re.I),
    re.compile(r"(?:i (?:work|study) (?:at|in|as)\s+)(.{3,80})", re.I),
    re.compile(r"(?:i(?:'m| am) (?:a|an)\s+)(.{3,60})", re.I),
    re.compile(r"(?:call me|prefer(?:s)?)\s+(.{2,40})", re.I),
    re.compile(r"(?:i (?:live|live in|based in)\s+)(.{3,60})", re.I),
]


class MemoryService:
    def __init__(self, folder: str = None):
        self.folder = folder or Config.MEMORY_FOLDER
        os.makedirs(self.folder, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, device_id: str) -> str:
        safe = re.sub(r"[^\w\-]", "_", device_id)[:128]
        return os.path.join(self.folder, f"{safe}.json")

    def _load(self, device_id: str) -> Dict[str, Any]:
        path = self._path(device_id)
        if not os.path.exists(path):
            return {"device_id": device_id, "facts": [], "updated_at": None}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"device_id": device_id, "facts": [], "updated_at": None}
            data.setdefault("facts", [])
            return data
        except Exception as e:
            logger.warning("Failed to load memory for %s: %s", device_id, e)
            return {"device_id": device_id, "facts": [], "updated_at": None}

    def _save(self, device_id: str, data: Dict[str, Any]) -> None:
        path = self._path(device_id)
        data["device_id"] = device_id
        data["updated_at"] = time.time()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def get_facts(self, device_id: Optional[str]) -> List[str]:
        if not validate_device_id(device_id):
            return []
        with self._lock:
            data = self._load(device_id)
        facts = data.get("facts") or []
        return [f for f in facts if isinstance(f, str) and f.strip()][:40]

    def get_context_block(self, device_id: Optional[str]) -> str:
        facts = self.get_facts(device_id)
        if not facts:
            return ""
        lines = "\n".join(f"- {f}" for f in facts)
        return (
            "=== USER MEMORY (device-scoped, no account) ===\n"
            "Use these known facts about the user when relevant. "
            "Do not invent personal details beyond this list.\n"
            f"{lines}"
        )

    def add_facts(self, device_id: Optional[str], facts: List[str]) -> List[str]:
        if not validate_device_id(device_id) or not facts:
            return self.get_facts(device_id)
        cleaned = []
        for f in facts:
            f = (f or "").strip()
            if 3 <= len(f) <= 200:
                cleaned.append(f)
        if not cleaned:
            return self.get_facts(device_id)

        with self._lock:
            data = self._load(device_id)
            existing = data.get("facts") or []
            lower = {e.lower() for e in existing}
            for f in cleaned:
                if f.lower() not in lower:
                    existing.append(f)
                    lower.add(f.lower())
            data["facts"] = existing[-40:]
            self._save(device_id, data)
            return data["facts"]

    def extract_and_update(self, device_id: Optional[str], user_text: str) -> List[str]:
        """Lightweight heuristic extraction from the latest user message."""
        if not validate_device_id(device_id) or not user_text:
            return []
        found = []
        for pattern in _FACT_PATTERNS:
            m = pattern.search(user_text)
            if m:
                fact = m.group(0).strip()
                if len(fact) > 200:
                    fact = fact[:200]
                found.append(fact)
        if found:
            self.add_facts(device_id, found)
        return found

    def clear(self, device_id: Optional[str]) -> bool:
        if not validate_device_id(device_id):
            return False
        path = self._path(device_id)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
        return False


memory_service = MemoryService()
