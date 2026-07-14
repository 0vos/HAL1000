"""Tool call statistics: counts, latency, success rate, failure breakdown."""
from __future__ import annotations

import json
import statistics
import threading
from collections import defaultdict
from pathlib import Path


class ToolStats:
    def __init__(self, persist_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._success: dict[str, int] = defaultdict(int)
        self._failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._cache_hits = 0
        self._cache_misses = 0
        self._retry_attempts = 0
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._counts = defaultdict(int, data.get("counts", {}))
                self._success = defaultdict(int, data.get("success", {}))
                self._failures = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in data.get("failures", {}).items()})
                self._latencies = defaultdict(list, {k: list(v) for k, v in data.get("latencies", {}).items()})
                self._cache_hits = int(data.get("cache_hits", 0))
                self._cache_misses = int(data.get("cache_misses", 0))
                self._retry_attempts = int(data.get("retry_attempts", 0))
            except Exception:
                pass

    def record(self, name: str, status: str, latency_ms: float, error_code: str | None = None) -> None:
        with self._lock:
            self._counts[name] += 1
            if status == "success":
                self._success[name] += 1
            else:
                code = error_code or "UNKNOWN"
                self._failures[name][code] += 1
            self._latencies[name].append(float(latency_ms))
        self._flush()

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1
        self._flush()

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1
        self._flush()

    def record_retry(self) -> None:
        with self._lock:
            self._retry_attempts += 1
        self._flush()

    def snapshot(self) -> dict:
        with self._lock:
            tools = {}
            for name, total in self._counts.items():
                successes = self._success[name]
                latency = self._latencies[name]
                tools[name] = {
                    "calls": total,
                    "successes": successes,
                    "failures": total - successes,
                    "failure_rate": round(1 - successes / total, 4) if total else 0.0,
                    "avg_latency_ms": round(statistics.fmean(latency), 3) if latency else 0.0,
                    "p50_latency_ms": round(_percentile(latency, 50), 3) if latency else 0.0,
                    "p95_latency_ms": round(_percentile(latency, 95), 3) if latency else 0.0,
                    "error_codes": dict(self._failures[name]),
                }
            return {
                "total_calls": sum(self._counts.values()),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "retry_attempts": self._retry_attempts,
                "tools": tools,
            }

    def _flush(self) -> None:
        if not self._persist_path:
            return
        data = {
            "counts": dict(self._counts),
            "success": dict(self._success),
            "failures": {k: dict(v) for k, v in self._failures.items()},
            "latencies": {k: list(v) for k, v in self._latencies.items()},
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "retry_attempts": self._retry_attempts,
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._persist_path)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)