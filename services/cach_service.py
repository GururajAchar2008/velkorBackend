"""
services/cach_service.py
Simple TTL in-memory cache for search/RAG snippets.
"""

import threading
import time
from typing import Any, Optional


class CacheService:
    def __init__(self, default_ttl: float = 300.0, max_entries: int = 256):
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expires = item
            if time.time() > expires:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: float = None) -> None:
        ttl = self.default_ttl if ttl is None else ttl
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Drop oldest-ish entries
                for k in list(self._store.keys())[: max(1, self.max_entries // 10)]:
                    self._store.pop(k, None)
            self._store[key] = (value, time.time() + ttl)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


cache_service = CacheService()
