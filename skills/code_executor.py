"""
code_executor.py — 在隔离子进程中执行 Python 代码片段。

这是 HAL1000 最强大的工具：Agent 生成代码，子进程运行，结果（stdout/stderr/返回值）
反馈回 Agent，Agent 可以根据结果决定修改代码再次执行。

安全机制：
  - 子进程完全隔离：崩溃不影响主进程
  - 硬超时：默认 15s，超时强制 kill
  - 禁止网络访问（不 import socket/urllib/requests）
  - stdout/stderr 截断（最多 4000 字符）
  - 代码写入临时文件，执行完删除
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from time import perf_counter

# 禁止的 import 模式（黑名单检查，防止最明显的危险用法）
_FORBIDDEN = [
    "import socket", "import urllib", "import requests", "import http",
    "import ftplib", "import smtplib", "import telnetlib",
    "__import__('os').system", "os.system(", "os.popen(",
    "subprocess.call(", "subprocess.run(", "subprocess.Popen(",
]

_MAX_OUTPUT = 4000  # 最多返回字符数


def _check_forbidden(code: str) -> str | None:
    """返回第一个命中的禁止模式，否则 None。"""
    for pat in _FORBIDDEN:
        if pat in code:
            return pat
    return None


def code_executor(
    code: str,
    language: str = "python",
    timeout: float = 15.0,
    data_root: str | None = None,
) -> dict:
    """
    在隔离子进程中执行 Python 代码片段。

    参数：
        code     : 要执行的 Python 代码（字符串）
        language : 目前只支持 "python"（预留扩展）
        timeout  : 执行超时秒数，默认 15s
        data_root: 数据根目录，会注入到代码的工作目录

    返回 dict：
        stdout      : 标准输出（字符串，最多 4000 字符）
        stderr      : 标准错误（字符串）
        returncode  : 退出码（0 = 正常）
        timed_out   : 是否超时
        error       : 如果有禁止操作，这里是错误描述
        elapsed_ms  : 执行耗时（毫秒）
    """
    if language != "python":
        return {
            "stdout": "",
            "stderr": f"不支持的语言: {language}，目前只支持 python",
            "returncode": -1,
            "timed_out": False,
            "error": "unsupported_language",
            "elapsed_ms": 0,
        }

    # 安全检查
    hit = _check_forbidden(code)
    if hit:
        return {
            "stdout": "",
            "stderr": f"代码包含禁止操作: {hit!r}",
            "returncode": -1,
            "timed_out": False,
            "error": f"forbidden: {hit}",
            "elapsed_ms": 0,
        }

    # 写入临时文件
    work_dir = str(data_root) if data_root else tempfile.gettempdir()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
        dir=work_dir, prefix="hal_exec_",
        encoding="utf-8",
    )
    try:
        tmp.write(textwrap.dedent(code))
        tmp.flush()
        tmp.close()

        t0 = perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "code")},
            )
            elapsed_ms = round((perf_counter() - t0) * 1000, 1)

            stdout = proc.stdout[:_MAX_OUTPUT]
            stderr = proc.stderr[:_MAX_OUTPUT]
            if len(proc.stdout) > _MAX_OUTPUT:
                stdout += f"\n... [输出过长，已截断，共 {len(proc.stdout)} 字符]"

            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
                "timed_out": False,
                "error": None,
                "elapsed_ms": elapsed_ms,
            }
        except subprocess.TimeoutExpired:
            elapsed_ms = round((perf_counter() - t0) * 1000, 1)
            return {
                "stdout": "",
                "stderr": f"执行超时（>{timeout}s），子进程已强制终止",
                "returncode": -1,
                "timed_out": True,
                "error": "timeout",
                "elapsed_ms": elapsed_ms,
            }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
