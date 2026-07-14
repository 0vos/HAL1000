"""Cross-platform restricted Python execution for B2.

This is a restricted executor, not a claim of perfect Python sandboxing. It
combines an AST allowlist, minimal builtins, a fresh subprocess, wall-clock
and memory monitoring, process-tree cleanup, and bounded output. Linux and
Windows use the same JSON-over-stdio worker protocol.
"""
from __future__ import annotations

import ast
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MEMORY_LIMIT_MB = 128
_DEFAULT_MAX_OUTPUT_CHARS = 4000
_MAX_SOURCE_CHARS = 4000
_MAX_AST_NODES = 500

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_SAFE_MODULES = {"math": math, "statistics": statistics}
_SAFE_MODULE_ATTRIBUTES = {
    "math": {
        "acos", "asin", "atan", "atan2", "ceil", "comb", "cos", "degrees",
        "e", "exp", "fabs", "factorial", "floor", "fmod", "fsum", "gcd",
        "hypot", "isclose", "isfinite", "isinf", "isnan", "lcm", "log",
        "log10", "log2", "perm", "pi", "pow", "prod", "radians", "sin",
        "sqrt", "tan", "tau", "trunc",
    },
    "statistics": {
        "fmean", "geometric_mean", "harmonic_mean", "mean", "median",
        "median_grouped", "median_high", "median_low", "mode", "multimode",
        "pstdev", "pvariance", "quantiles", "stdev", "variance",
    },
}

_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Return,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


class SandboxPolicyError(ValueError):
    """Raised when source code violates the restricted execution policy."""


class _PolicyValidator(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxPolicyError(f"forbidden syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("_"):
            raise SandboxPolicyError(f"private name access is forbidden: {node.id}")
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not isinstance(node.value, ast.Name):
            raise SandboxPolicyError("attribute access is limited to approved modules")
        module_name = node.value.id
        if node.attr not in _SAFE_MODULE_ATTRIBUTES.get(module_name, set()):
            raise SandboxPolicyError(f"forbidden attribute: {module_name}.{node.attr}")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                raise SandboxPolicyError(f"function call is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            raise SandboxPolicyError("dynamic function calls are forbidden")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise SandboxPolicyError("expanded keyword arguments are forbidden")
            self.visit(keyword.value)


class _BoundedTextBuffer(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit = limit
        self._parts: list[str] = []
        self._length = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        value = str(text)
        remaining = self.limit - self._length
        if remaining > 0:
            accepted = value[:remaining]
            self._parts.append(accepted)
            self._length += len(accepted)
        if len(value) > max(remaining, 0):
            self.truncated = True
        return len(value)

    def getvalue(self) -> str:
        value = "".join(self._parts)
        if self.truncated:
            return value + "\n... [output truncated]"
        return value


def _validate_source(source: str) -> None:
    if not isinstance(source, str) or not source.strip():
        raise SandboxPolicyError("source must be a non-empty string")
    if len(source) > _MAX_SOURCE_CHARS:
        raise SandboxPolicyError(f"source exceeds {_MAX_SOURCE_CHARS}-character limit")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxPolicyError(f"invalid Python syntax: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise SandboxPolicyError(f"source exceeds {_MAX_AST_NODES}-node AST limit")
    _PolicyValidator().visit(tree)


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item, depth + 1) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    return repr(value)


def _apply_unix_limits(timeout_seconds: float, memory_limit_mb: int) -> bool:
    try:
        import resource

        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_seconds = max(1, int(timeout_seconds) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        return True
    except (ImportError, OSError, ValueError):
        return False


def _runtime_error_code(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "OVERFLOW"
    if isinstance(exc, ZeroDivisionError):
        return "DIVISION_BY_ZERO"
    if isinstance(exc, (TypeError, ValueError, KeyError, IndexError)):
        return "INVALID_INPUT"
    if isinstance(exc, OverflowError):
        return "OVERFLOW"
    return "INTERNAL"


def _execute_worker(request: dict) -> dict:
    source = request["source"]
    timeout_seconds = float(request["timeout_seconds"])
    memory_limit_mb = int(request["memory_limit_mb"])
    max_output_chars = int(request["max_output_chars"])
    stdout_buffer = _BoundedTextBuffer(max_output_chars)
    stderr_buffer = _BoundedTextBuffer(max_output_chars)
    unix_limits = _apply_unix_limits(timeout_seconds, memory_limit_mb)
    safe_globals = {"__builtins__": _SAFE_BUILTINS, **_SAFE_MODULES}
    safe_locals: dict[str, Any] = {}
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(source, "<b2-sandbox>", "exec"), safe_globals, safe_locals)
        result = _json_safe(safe_locals.get("result")) if request["capture_result"] else None
        return {
            "status": "success",
            "result": result,
            "output": repr(result) if result is not None else "<no result>",
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "unix_resource_limits": unix_limits,
        }
    except BaseException as exc:
        return {
            "status": "error",
            "error": {
                "type": "SandboxMemoryError" if isinstance(exc, MemoryError) else type(exc).__name__,
                "message": (
                    f"execution exceeded {memory_limit_mb} MB memory limit"
                    if isinstance(exc, MemoryError)
                    else str(exc)
                ),
                "code": _runtime_error_code(exc),
            },
            "result": None,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "unix_resource_limits": unix_limits,
        }


def _worker_main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        payload = _execute_worker(request)
    except BaseException as exc:
        payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "code": "INTERNAL",
            },
            "result": None,
            "stdout": "",
            "stderr": "",
            "unix_resource_limits": False,
        }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil

        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            total += child.memory_info().rss
        return total / (1024 * 1024)
    except Exception:
        return None


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        import psutil

        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        root.terminate()
        _, alive = psutil.wait_procs([*children, root], timeout=1.0)
        for item in alive:
            item.kill()
        return
    except Exception:
        pass
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _minimal_environment() -> dict[str, str]:
    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _security_metadata(
    timeout_seconds: float,
    memory_limit_mb: int,
    max_output_chars: int,
    peak_memory_mb: float | None,
    unix_resource_limits: bool = False,
) -> dict:
    return {
        "isolation": "subprocess-json-stdio",
        "platform": platform.system().lower(),
        "ast_policy": "allowlist-v1",
        "filesystem_access": "blocked_by_policy",
        "network_access": "blocked_by_policy",
        "timeout_seconds": timeout_seconds,
        "memory_limit_mb": memory_limit_mb,
        "memory_monitor": "psutil-rss" if _rss_mb(os.getpid()) is not None else "unavailable",
        "unix_resource_limits": unix_resource_limits,
        "max_output_chars": max_output_chars,
        "peak_memory_mb": round(peak_memory_mb, 3) if peak_memory_mb is not None else None,
    }


def _policy_error(
    exc: BaseException,
    timeout_seconds: float,
    memory_limit_mb: int,
    max_output_chars: int,
) -> dict:
    return {
        "status": "error",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "code": "PERMISSION_DENIED" if isinstance(exc, SandboxPolicyError) else "INVALID_INPUT",
        },
        "result": None,
        "stdout": "",
        "stderr": "",
        "security": _security_metadata(
            timeout_seconds, memory_limit_mb, max_output_chars, None
        ),
    }


def safe_python_exec(
    source: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    allow_return: bool = True,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
) -> dict:
    """Execute a small Python snippet in a restricted subprocess.

    Put the desired value in a variable named ``result``. ``allow_return`` is
    retained for compatibility and controls whether that value is returned.
    Imports, file/network access, private attributes, dynamic calls, and
    user-defined functions are rejected before the worker starts.
    """
    try:
        timeout_value = float(timeout_seconds)
        memory_value = int(memory_limit_mb)
        output_value = int(max_output_chars)
    except (TypeError, ValueError) as exc:
        return _policy_error(
            exc,
            _DEFAULT_TIMEOUT_SECONDS,
            _DEFAULT_MEMORY_LIMIT_MB,
            _DEFAULT_MAX_OUTPUT_CHARS,
        )

    try:
        if not 0.05 <= timeout_value <= 30:
            raise ValueError("timeout_seconds must be between 0.05 and 30")
        if not 32 <= memory_value <= 1024:
            raise ValueError("memory_limit_mb must be between 32 and 1024")
        if not 100 <= output_value <= 100_000:
            raise ValueError("max_output_chars must be between 100 and 100000")
        _validate_source(source)
    except (TypeError, ValueError) as exc:
        return _policy_error(exc, timeout_value, memory_value, output_value)

    request = {
        "source": source,
        "capture_result": bool(allow_return),
        "timeout_seconds": timeout_value,
        "memory_limit_mb": memory_value,
        "max_output_chars": output_value,
    }
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    with tempfile.TemporaryDirectory(prefix="b2_sandbox_") as work_dir:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=work_dir,
            env=_minimal_environment(),
            creationflags=creationflags,
            **popen_kwargs,
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps(request, ensure_ascii=False))
        process.stdin.close()
        process.stdin = None

        deadline = time.monotonic() + timeout_value
        peak_memory_mb: float | None = None
        memory_exceeded = False
        while process.poll() is None and time.monotonic() < deadline:
            current_memory = _rss_mb(process.pid)
            if current_memory is not None:
                peak_memory_mb = max(peak_memory_mb or 0.0, current_memory)
                if current_memory > memory_value:
                    memory_exceeded = True
                    break
            time.sleep(0.02)

        if memory_exceeded:
            _kill_process_tree(process)
            process.communicate()
            return {
                "status": "error",
                "error": {
                    "type": "SandboxMemoryError",
                    "message": f"execution exceeded {memory_value} MB memory limit",
                    "code": "OVERFLOW",
                },
                "result": None,
                "stdout": "",
                "stderr": "",
                "security": _security_metadata(
                    timeout_value, memory_value, output_value, peak_memory_mb
                ),
            }

        if process.poll() is None:
            _kill_process_tree(process)
            process.communicate()
            return {
                "status": "timeout",
                "error": {
                    "type": "SandboxTimeout",
                    "message": f"execution exceeded {timeout_value:.2f}s timeout",
                    "code": "EXECUTION_TIMEOUT",
                },
                "result": None,
                "stdout": "",
                "stderr": "",
                "security": _security_metadata(
                    timeout_value, memory_value, output_value, peak_memory_mb
                ),
            }

        worker_stdout, worker_stderr = process.communicate()
        try:
            payload = json.loads(worker_stdout)
        except json.JSONDecodeError:
            payload = {
                "status": "error",
                "error": {
                    "type": "SandboxProcessError",
                    "message": worker_stderr.strip() or f"worker exited with code {process.returncode}",
                    "code": "INTERNAL",
                },
                "result": None,
                "stdout": "",
                "stderr": worker_stderr[:output_value],
                "unix_resource_limits": False,
            }
        payload["security"] = _security_metadata(
            timeout_value,
            memory_value,
            output_value,
            peak_memory_mb,
            bool(payload.pop("unix_resource_limits", False)),
        )
        return payload


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    raise SystemExit("safe_python_exec.py is an internal worker; use b2_advanced.py")
