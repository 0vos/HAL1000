"""
local_file_search.py — 本地文件搜索和目录浏览

支持两种模式：
1. 目录浏览（query 为空或 "list"）：列出目录下所有文件和子目录
2. 关键词搜索（query 非空）：在目录下的文本文件里搜索关键词

支持任意绝对路径（不限于 data/ 目录）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills import DEFAULT_DATA_ROOT, resolve_data_path
from skills_error_codes import ErrorCode, attach_error_code

# 支持内容搜索的文件类型
_TEXT_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".sh", ".csv", ".html", ".css", ".toml", ".ini", ".cfg", ".log",
    ".rst", ".xml", ".sql",
}

_TIMEOUT_SECONDS = 10


def _snippet(text: str, terms: list[str], context: int = 80) -> str:
    lowered = text.casefold()
    for term in terms:
        idx = lowered.find(term.casefold())
        if idx >= 0:
            start = max(0, idx - context)
            end = min(len(text), idx + len(term) + context)
            return text[start:end].replace("\n", " ")
    return text[:160].replace("\n", " ")


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def local_file_search(
    query: str,
    root_dir: str = ".",
    file_types: list[str] | None = None,
    top_k: int = 50,
    *,
    data_root: str | None = None,
) -> dict:
    """
    搜索或列出目录内容。

    Args:
        query:      搜索关键词；为空/"list"/"ls"/"dir" 时列出目录结构
        root_dir:   搜索根目录，支持绝对路径（如 /root/siton-tmp/HAL1000/.logs）
        file_types: 限定文件类型（如 ["py", "md"]），默认搜索所有文本文件
        top_k:      最多返回结果数
        data_root:  数据根目录（相对路径的基准），默认为 data/
    """
    # 解析目录路径
    try:
        search_root, data_root_path = resolve_data_path(root_dir, data_root)
    except Exception:
        # resolve_data_path 失败时直接用绝对路径
        search_root = Path(root_dir).expanduser().resolve()
        data_root_path = search_root

    if not search_root.exists():
        raise attach_error_code(
            FileNotFoundError(f"目录不存在: {root_dir}"), ErrorCode.FILE_NOT_FOUND
        )
    if not search_root.is_dir():
        raise attach_error_code(
            ValueError(f"不是目录: {root_dir}"), ErrorCode.INVALID_INPUT
        )

    # 判断模式：目录浏览 vs 关键词搜索
    list_mode = not query or query.strip().lower() in ("list", "ls", "dir", "列出", "浏览", "")

    # 构建文件扩展名过滤器
    if file_types:
        exts = {f".{e.lower().lstrip('.')}" for e in file_types}
    else:
        exts = _TEXT_EXTS  # 默认所有文本类型

    if list_mode:
        # ── 目录浏览模式：列出所有文件和子目录 ──────────────────
        entries = []
        dirs_seen = set()
        try:
            for path in sorted(search_root.rglob("*")):
                if len(entries) >= top_k:
                    break
                rel = path.relative_to(search_root)
                if path.is_dir():
                    dirs_seen.add(str(rel))
                    entries.append({
                        "type": "dir",
                        "path": str(rel),
                        "full_path": str(path),
                    })
                elif path.is_file():
                    try:
                        size = path.stat().st_size
                    except Exception:
                        size = 0
                    entries.append({
                        "type": "file",
                        "path": str(rel),
                        "full_path": str(path),
                        "ext": path.suffix.lower(),
                        "size": _format_size(size),
                    })
        except PermissionError as e:
            raise attach_error_code(PermissionError(f"权限不足: {e}"), ErrorCode.PERMISSION_ERROR)

        # 生成可读的目录树摘要
        summary_lines = [f"目录: {search_root}", f"共 {len([e for e in entries if e['type']=='file'])} 个文件，{len([e for e in entries if e['type']=='dir'])} 个子目录", ""]
        for e in entries:
            indent = "  " * (str(e["path"]).count("/") + str(e["path"]).count("\\"))
            if e["type"] == "dir":
                summary_lines.append(f"{indent}📁 {Path(e['path']).name}/")
            else:
                size_str = f" ({e['size']})" if e.get("size") else ""
                summary_lines.append(f"{indent}📄 {Path(e['path']).name}{size_str}")

        return {
            "mode": "list",
            "root": str(search_root),
            "entries": entries,
            "summary": "\n".join(summary_lines),
        }

    else:
        # ── 关键词搜索模式 ──────────────────────────────────────
        terms = [t for t in re.split(r"\s+", query.strip()) if t]
        results = []
        try:
            for path in sorted(search_root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in exts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                lowered = text.casefold()
                score = sum(lowered.count(t.casefold()) for t in terms)
                if score:
                    try:
                        rel = path.relative_to(search_root)
                    except ValueError:
                        rel = path
                    results.append({
                        "path": str(rel),
                        "full_path": str(path),
                        "score": score,
                        "snippet": _snippet(text, terms),
                    })
        except PermissionError as e:
            raise attach_error_code(PermissionError(f"权限不足: {e}"), ErrorCode.PERMISSION_ERROR)

        results.sort(key=lambda x: (-x["score"], x["path"]))
        return {
            "mode": "search",
            "query": query,
            "root": str(search_root),
            "results": results[:top_k],
            "total_found": len(results),
        }
