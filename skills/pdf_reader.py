"""pdf_reader.py — 读取 PDF 文件内容"""
from __future__ import annotations

import shutil
import subprocess

from skills import resolve_data_path


def pdf_reader(path: str, data_root: str | None = None, max_chars: int = 5000) -> dict:
    """
    读取 PDF 文件，返回文本内容。

    优先用 pdfplumber，fallback 用 pypdf2（PyPDF2），再 fallback 用 markitdown CLI。
    path 相对于 data_root（即 data/ 目录）。

    返回 {"content": str, "num_pages": int, "num_chars": int}
    """
    source, root = resolve_data_path(path, data_root)

    if source.suffix.lower() != ".pdf":
        raise ValueError(f"pdf_reader only supports .pdf files, got: {source.suffix}")
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    content = ""
    num_pages = 0

    # ── 优先：pdfplumber ──────────────────────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(str(source)) as pdf:
            num_pages = len(pdf.pages)
            parts = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                parts.append(text)
            content = "\n".join(parts)
        return {
            "content": content[:max_chars],
            "num_pages": num_pages,
            "num_chars": len(content[:max_chars]),
        }
    except ImportError:
        pass
    except Exception as exc:
        # pdfplumber 解析失败，继续 fallback
        _pdfplumber_err = str(exc)

    # ── Fallback 1：PyPDF2 ────────────────────────────────────────────
    try:
        import PyPDF2  # noqa: N813
        with open(str(source), "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            parts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                parts.append(text)
            content = "\n".join(parts)
        return {
            "content": content[:max_chars],
            "num_pages": num_pages,
            "num_chars": len(content[:max_chars]),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # ── Fallback 2：markitdown CLI ────────────────────────────────────
    try:
        import subprocess
        import shutil
        if shutil.which("markitdown"):
            result = subprocess.run(
                ["markitdown", str(source)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                content = result.stdout
                # markitdown 不提供页数，估算
                num_pages = max(1, content.count("\n\n") // 5)
                return {
                    "content": content[:max_chars],
                    "num_pages": num_pages,
                    "num_chars": len(content[:max_chars]),
                }
    except Exception:
        pass

    # ── Fallback 3：pdftotext 命令（poppler-utils）────────────────────
    if shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", str(source), "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                text = result.stdout[:max_chars]
                return {
                    "content": text,
                    "num_chars": len(text),
                    "source": str(source),
                    "backend": "pdftotext",
                }
        except Exception:
            pass

    raise RuntimeError(
        f"无法读取 PDF 文件 {path}：pdfplumber、PyPDF2、markitdown 和 pdftotext 均不可用或解析失败。"
        "请安装：pip install pdfplumber"
    )
