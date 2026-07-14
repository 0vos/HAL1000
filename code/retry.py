"""Retry layer for recoverable tool errors.

Some errors are worth retrying (e.g. transient FileNotFoundError from a
slow filesystem, JSON decode failure from a flaky parser). Others must
never be retried (e.g. INVALID_INPUT, UNSUPPORTED_TYPE) because they will
fail identically every time.
"""
from __future__ import annotations

import time
from typing import Callable


RETRYABLE_EXCEPTIONS = (
    # FileNotFoundError 不在此列：文件不存在重试也不会凭空出现，属于调用方逻辑错误
    ConnectionError,
    TimeoutError,
    OSError,
)

RETRYABLE_STATUS = {"timeout"}


def should_retry(exc: BaseException | None, attempts: int, max_attempts: int) -> bool:
    if attempts >= max_attempts:
        return False
    if exc is None:
        return False
    # FileNotFoundError 是 OSError 子类，但文件不存在重试也不会自动出现，不应重试
    if isinstance(exc, FileNotFoundError):
        return False
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


def should_retry_result(result: dict | None, attempts: int, max_attempts: int) -> bool:
    if attempts >= max_attempts or not isinstance(result, dict):
        return False
    if result.get("status") in RETRYABLE_STATUS:
        return True
    error = result.get("error")
    if isinstance(error, dict):
        code = error.get("code", "")
        return code in {"EXECUTION_TIMEOUT", "FILE_NOT_FOUND", "INTERNAL"}
    return False


def call_with_retry(
    func: Callable[[], dict],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Call `func` and retry if it raises a retryable exception or returns
    a result that looks recoverable. Returns (final_result, attempt_log).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    attempts: list[dict] = []
    delay = base_delay
    last_exc: BaseException | None = None
    last_result: dict | None = None
    for index in range(max_attempts):
        start = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 3)
            attempts.append({
                "attempt": index + 1,
                "outcome": "exception",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "latency_ms": elapsed,
            })
            if should_retry(exc, index + 1, max_attempts):
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            raise
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        attempts.append({
            "attempt": index + 1,
            "outcome": "result",
            "status": result.get("status") if isinstance(result, dict) else None,
            "latency_ms": elapsed,
        })
        if should_retry_result(result, index + 1, max_attempts):
            last_result = result
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue
        return result, attempts
    if last_exc is not None:
        raise last_exc
    return last_result or {"status": "error", "error": {"type": "RetryExhausted", "message": "no attempts completed"}}, attempts