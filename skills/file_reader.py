from __future__ import annotations

import signal
from contextlib import contextmanager

from skills import resolve_data_path
from skills_error_codes import ErrorCode, attach_error_code


@contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise TimeoutError(f"file_reader exceeded {seconds:.1f}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


_TIMEOUT_SECONDS = 3.0

# 二进制格式黑名单：这些文件无法用 utf-8 读取有意义的文本
_BINARY_EXTS = {
    # 图片
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff", ".tif",
    # 音视频
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg",
    # 文档/压缩包
    ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".tar", ".gz", ".rar", ".7z",
    # 可执行文件
    ".exe", ".dll", ".so", ".dylib", ".bin", ".elf",
    # 数据库
    ".db", ".sqlite", ".sqlite3",
    # 其他二进制
    ".pyc", ".pyo", ".class", ".o", ".a",
}


def file_reader(path: str, max_chars: int = 2000, *, data_root: str | None = None) -> dict:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise attach_error_code(ValueError("max_chars must be a positive integer"), ErrorCode.INVALID_INPUT)
    if not path or not str(path).strip():
        raise attach_error_code(
            ValueError("file_reader 需要指定文件路径，不能为空。请提供具体的文件名（如 sort.py、docs/report.md）"),
            ErrorCode.INVALID_INPUT,
        )
    # 绝对路径直接用，不过 resolve_data_path
    from pathlib import Path as _Path
    _path_obj = _Path(path)
    if _path_obj.is_absolute():
        source = _path_obj
        root = _path_obj.parent
    else:
        source, root = resolve_data_path(path, data_root)
    ext = source.suffix.lower()
    if ext in _BINARY_EXTS:
        raise attach_error_code(
            ValueError(f"file_reader 不支持二进制格式 {ext or '(无扩展名)'}，请使用对应的专用工具（如 image_qa、pdf_reader）"),
            ErrorCode.UNSUPPORTED_TYPE,
        )
    if not source.is_file():
        raise attach_error_code(FileNotFoundError(f"file not found: {path}"), ErrorCode.FILE_NOT_FOUND)
    try:
        with _time_limit(_TIMEOUT_SECONDS):
            try:
                original = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # 不在黑名单中但实际是二进制内容，也要主动报错
                raise attach_error_code(
                    ValueError(f"文件 {source.name} 内容不是 UTF-8 文本，无法读取"),
                    ErrorCode.UNSUPPORTED_TYPE,
                )
    except TimeoutError as exc:
        raise attach_error_code(exc, ErrorCode.EXECUTION_TIMEOUT) from exc
    content = original[:max_chars]
    return {
        "content": content,
        "num_chars": len(content),
        "source": source.relative_to(root).as_posix(),
        "truncated": len(original) > len(content),
    }
