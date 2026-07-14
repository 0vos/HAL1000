"""Error classification for Skill execution.

Provides:
    ErrorCode - an Enum mapping business exceptions to a stable taxonomy
    classify_exception(exc) -> (code, message) tuple
    attach_error_code(exc, code) - tag an existing exception with a code
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Tuple


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    PATH_ESCAPE = "PATH_ESCAPE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    OVERFLOW = "OVERFLOW"
    DIVISION_BY_ZERO = "DIVISION_BY_ZERO"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INTERNAL = "INTERNAL"


_CODE_ATTR = "_hal_error_code"

_ERROR_MAP: Tuple[Tuple[type, ErrorCode], ...] = (
    (FileNotFoundError, ErrorCode.FILE_NOT_FOUND),
    (PermissionError, ErrorCode.PERMISSION_DENIED),
    (TimeoutError, ErrorCode.EXECUTION_TIMEOUT),
    (ValueError, ErrorCode.INVALID_INPUT),
    (OverflowError, ErrorCode.OVERFLOW),
    (ZeroDivisionError, ErrorCode.DIVISION_BY_ZERO),
    (TypeError, ErrorCode.INVALID_INPUT),
    (KeyError, ErrorCode.INVALID_INPUT),
    (NotImplementedError, ErrorCode.UNSUPPORTED_TYPE),
)


def classify_exception(exc: BaseException) -> ErrorCode:
    explicit = getattr(exc, _CODE_ATTR, None)
    if isinstance(explicit, ErrorCode):
        return explicit
    msg = str(exc).lower()
    if "escape" in msg:
        return ErrorCode.PATH_ESCAPE
    if "timeout" in msg or "timed out" in msg:
        return ErrorCode.EXECUTION_TIMEOUT
    if "division" in msg or "zero" in msg:
        return ErrorCode.DIVISION_BY_ZERO
    if "overflow" in msg or "out of range" in msg or "too large" in msg:
        return ErrorCode.OVERFLOW
    if "not supported" in msg or "only supports" in msg or "unsupported" in msg:
        return ErrorCode.UNSUPPORTED_TYPE
    if "not found" in msg or "no such file" in msg:
        return ErrorCode.FILE_NOT_FOUND
    if "permission" in msg:
        return ErrorCode.PERMISSION_DENIED
    for exc_type, code in _ERROR_MAP:
        if isinstance(exc, exc_type):
            return code
    return ErrorCode.INTERNAL


def attach_error_code(exc: BaseException, code: ErrorCode) -> BaseException:
    setattr(exc, _CODE_ATTR, code)
    return exc


def enrich_error_payload(exc: BaseException) -> dict:
    code = classify_exception(exc)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "code": code.value,
    }