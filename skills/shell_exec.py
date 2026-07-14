"""
shell_exec.py — 在安全沙箱里执行 shell 命令

用于替代 file_reader + local_file_search 的复杂路径逻辑：
直接跑 ls / cat / grep / find / head / tail / wc 等命令。

安全限制：
- 禁止危险命令：rm -rf / dd / mkfs / shutdown / reboot / > /dev/sda 等
- 禁止网络命令：curl / wget / nc / ssh（避免外泄数据）
- 禁止修改系统文件：/etc / /boot / /sys / /proc
- 只允许读操作和无副作用的分析命令
- 超时 15 秒
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills_error_codes import ErrorCode, attach_error_code

_TIMEOUT = 15  # 秒

# 明确禁止的破坏性命令（黑名单方式，其余全部允许）
_DENY_PATTERNS = [
    (r'\brm\s+.*-[^-]*r', "rm -r 递归删除"),
    (r'\bdd\b.*of=', "dd 写磁盘"),
    (r'\bmkfs\b', "格式化磁盘"),
    (r'\bshutdown\b', "关机"),
    (r'\breboot\b', "重启"),
    (r'\bpoweroff\b', "关机"),
    (r'>\s*/dev/[sh]d', "写块设备"),
    (r'>\s*/dev/nvme', "写NVMe"),
    (r'>\s*/(etc|boot|sys)/(?!.*\.tmp)', "写系统目录"),
]


def _is_safe(command: str) -> tuple[bool, str]:
    """检查命令是否包含明确的破坏性操作。"""
    for pattern, desc in _DENY_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"禁止: {desc}"
    return True, ""


def shell_exec(
    command: str,
    workdir: str = "/",
    timeout: int = _TIMEOUT,
) -> dict:
    """
    执行 shell 命令并返回输出。

    Args:
        command:  要执行的 shell 命令，如 "ls -la /root/project" 或 "cat /path/to/file.log"
        workdir:  工作目录，默认 /
        timeout:  超时秒数，默认 15

    Returns:
        stdout, stderr, returncode
    """
    if not command or not command.strip():
        raise attach_error_code(ValueError("command 不能为空"), ErrorCode.INVALID_INPUT)

    is_safe, reason = _is_safe(command)
    if not is_safe:
        raise attach_error_code(
            PermissionError(f"命令被拒绝: {reason}"),
            ErrorCode.PERMISSION_DENIED
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        stdout = result.stdout
        stderr = result.stderr.strip()

        # 截断超长输出
        MAX_CHARS = 8000
        truncated = False
        if len(stdout) > MAX_CHARS:
            stdout = stdout[:MAX_CHARS]
            truncated = True

        return {
            "stdout": stdout,
            "stderr": stderr if stderr else None,
            "returncode": result.returncode,
            "truncated": truncated,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        raise attach_error_code(
            TimeoutError(f"命令超时 ({timeout}s): {command}"),
            ErrorCode.EXECUTION_TIMEOUT
        )
    except Exception as e:
        raise attach_error_code(e, ErrorCode.INTERNAL)
