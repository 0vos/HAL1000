"""
file_writer.py — 把文本内容写入 data 目录下的文件。

Agent 可用这个工具把代码、报告、结果保存到磁盘，
然后把路径告诉用户。
"""
from __future__ import annotations

import os
from pathlib import Path

from skills import resolve_data_path
from skills_error_codes import ErrorCode, attach_error_code

_ALLOWED_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".csv", ".yaml", ".yml",
    ".sh", ".html", ".js", ".ts", ".java", ".c", ".cpp",
}
_MAX_CONTENT_BYTES = 200_000  # 200 KB


def file_writer(
    path: str,
    content: str,
    overwrite: bool = True,
    *,
    data_root: str | None = None,
) -> dict:
    """
    把 content 写入 data 目录下的 path 文件。

    参数：
        path      : 相对于 data/ 目录的路径，例如 "algorithms/bubble_sort.py"
        content   : 要写入的文本内容
        overwrite : 文件已存在时是否覆盖，默认 True

    返回 dict：
        written_path : 实际写入的绝对路径
        num_bytes    : 写入字节数
        num_lines    : 写入行数
        overwritten  : 是否覆盖了已有文件
    """
    if not isinstance(content, str):
        raise attach_error_code(
            TypeError("content must be a string"), ErrorCode.INVALID_INPUT
        )
    if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
        raise attach_error_code(
            ValueError(f"content too large (>{_MAX_CONTENT_BYTES} bytes)"), ErrorCode.INVALID_INPUT
        )

    candidate, root = resolve_data_path(path, data_root)
    suffix = candidate.suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise attach_error_code(
            ValueError(
                f"file extension {suffix!r} not allowed. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
            ErrorCode.INVALID_INPUT,
        )

    already_exists = candidate.exists()
    if already_exists and not overwrite:
        raise attach_error_code(
            FileExistsError(f"file already exists: {path} (set overwrite=true to replace)"),
            ErrorCode.INVALID_INPUT,
        )

    # 自动创建父目录
    candidate.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入：先写临时文件，再 os.replace() 重命名
    # 即使进程在写入途中崩溃，目标文件要么是旧的完整版，要么是新的完整版，绝不会是半截
    tmp_path = candidate.with_name("." + candidate.name + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, candidate)
    except BaseException:
        # 清理临时文件，不遮蔽原异常
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    num_bytes = len(content.encode("utf-8"))
    num_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    return {
        "written_path": str(candidate),
        "relative_path": path,
        "num_bytes": num_bytes,
        "num_lines": num_lines,
        "overwritten": already_exists and overwrite,
    }
