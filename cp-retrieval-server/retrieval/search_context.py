from __future__ import annotations

from collections import OrderedDict
import secrets
import threading
import time


class SearchContextStore:
    """Bounded, expiring, browser-owned query references; never store query text in cookies."""

    def __init__(self, max_entries=512, max_bytes=32 * 1024 * 1024, ttl_seconds=3600,
                 per_owner=32, max_query_bytes=256 * 1024, clock=time.monotonic):
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._bytes = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._per_owner = per_owner
        self._max_query_bytes = max_query_bytes
        self._clock = clock

    def _remove(self, key):
        entry = self._entries.pop(key)
        self._bytes -= entry[3]

    def _expire(self):
        now = self._clock()
        for key, entry in list(self._entries.items()):
            if entry[2] <= now:
                self._remove(key)

    def put(self, owner: str, query: str, existing: str | None = None) -> str:
        size = len(query.encode("utf-8"))
        if size > self._max_query_bytes:
            raise ValueError("query_too_large")
        with self._lock:
            self._expire()
            if existing in self._entries:
                entry = self._entries[existing]
                if entry[0] == owner and entry[1] == query:
                    self._entries.move_to_end(existing)
                    self._entries[existing] = (owner, query, self._clock() + self._ttl, size)
                    return existing
            owned = [key for key, entry in self._entries.items() if entry[0] == owner]
            while len(owned) >= self._per_owner:
                self._remove(owned.pop(0))
            while self._entries and (len(self._entries) >= self._max_entries or self._bytes + size > self._max_bytes):
                self._remove(next(iter(self._entries)))
            if size > self._max_bytes:
                raise ValueError("query_too_large")
            key = secrets.token_urlsafe(24)
            self._entries[key] = (owner, query, self._clock() + self._ttl, size)
            self._bytes += size
            return key

    def get(self, owner: str, key: str) -> str | None:
        if not isinstance(key, str) or len(key) > 64:
            return None
        with self._lock:
            self._expire()
            entry = self._entries.get(key)
            if entry is None or entry[0] != owner:
                return None
            self._entries.move_to_end(key)
            self._entries[key] = (entry[0], entry[1], self._clock() + self._ttl, entry[3])
            return entry[1]
