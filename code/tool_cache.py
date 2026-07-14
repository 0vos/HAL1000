"""Result cache for tool calls.

Cache key = sha256(json.dumps(args, sort_keys=True)) of the tool name +
arguments. Reusing a cache hit means B3 will not re-invoke the Skill,
which is helpful for repeated queries, expensive Skills, and demos.

Supports both in-memory and on-disk JSON persistence.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


class ToolCache:
    def __init__(self, max_entries: int = 256, persist_path: str | Path | None = None):
        self._entries: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                for key, value in data.get("entries", []):
                    self._entries[key] = value
            except Exception:
                # corrupt cache: start fresh
                self._entries = OrderedDict()

    @staticmethod
    def make_key(name: str, args: dict) -> str:
        encoded = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{name}::{digest[:16]}"

    def get(self, name: str, args: dict) -> dict | None:
        key = self.make_key(name, args)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            # mark as recently used
            self._entries.move_to_end(key)
            return entry

    def put(self, name: str, args: dict, result: dict) -> str:
        key = self.make_key(name, args)
        with self._lock:
            self._entries[key] = {"tool_name": name, "args": args, "result": result}
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        self._flush()
        return key

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._entries), "max_entries": self._max}

    def _flush(self) -> None:
        if not self._persist_path:
            return
        data = {"entries": list(self._entries.items())}
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._persist_path)


GLOBAL_CACHE: ToolCache | None = None


def get_global_cache() -> ToolCache:
    global GLOBAL_CACHE
    if GLOBAL_CACHE is None:
        GLOBAL_CACHE = ToolCache()
    return GLOBAL_CACHE


def reset_global_cache() -> None:
    global GLOBAL_CACHE
    GLOBAL_CACHE = None