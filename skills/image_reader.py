"""image_reader.py — 读取本地图片文件，返回元信息 + 绝对路径（供 B4 视觉推理使用）。

设计：
  - 本 skill 只做「文件校验 + 元信息提取」，不调用模型。
  - 返回的 abs_path 会被 hal_chat.py / task_executor.py 用于下一轮生成时
    把图片作为多模态输入喂给 Qwen3.5（原生视觉-语言模型）。
  - 不在 output 里塞入完整 base64（避免把巨大字符串写进 tool_message JSON、
    撑爆上下文/日志），只在需要时提供小尺寸缩略图预览的 base64（可选）。
"""
from __future__ import annotations

from skills import resolve_data_path
from skills_error_codes import ErrorCode, attach_error_code

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MAX_BYTES = 20 * 1024 * 1024  # 20MB 上限，避免超大图片拖垮推理


def image_reader(
    path: str,
    *,
    data_root: str | None = None,
    include_thumbnail_base64: bool = False,
    thumbnail_max_side: int = 256,
) -> dict:
    """
    读取本地图片，校验格式/大小，返回元信息和绝对路径。

    Args:
        path: 图片路径，相对于 data_root（data/ 目录）
        include_thumbnail_base64: 是否附带一个小尺寸缩略图的 base64（默认 False，
            仅在需要在纯文本渠道预览时使用；视觉推理直接用 abs_path 加载原图）
        thumbnail_max_side: 缩略图最长边像素数

    Returns:
        {
          "abs_path": str,          # 绝对路径，供 B4 AutoProcessor 直接加载原图
          "relative_path": str,     # 相对 data_root 的路径
          "mime_type": str,
          "num_bytes": int,
          "width": int | None,      # 需要 PIL 才能拿到，没装则为 None
          "height": int | None,
          "thumbnail_base64": str | None,  # include_thumbnail_base64=True 时才有
        }
    """
    source, root = resolve_data_path(path, data_root)
    ext = source.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise attach_error_code(
            ValueError(f"image_reader only supports {sorted(_SUPPORTED_EXTS)}, got: {ext}"),
            ErrorCode.UNSUPPORTED_TYPE,
        )
    if not source.is_file():
        raise attach_error_code(FileNotFoundError(f"image not found: {path}"), ErrorCode.FILE_NOT_FOUND)

    num_bytes = source.stat().st_size
    if num_bytes > _MAX_BYTES:
        raise attach_error_code(
            ValueError(f"image too large: {num_bytes} bytes (limit {_MAX_BYTES})"),
            ErrorCode.INVALID_INPUT,
        )

    width = height = None
    thumbnail_base64 = None
    try:
        from PIL import Image
        with Image.open(source) as img:
            width, height = img.size
            if include_thumbnail_base64:
                import base64
                import io
                thumb = img.copy()
                thumb.thumbnail((thumbnail_max_side, thumbnail_max_side))
                buf = io.BytesIO()
                thumb.convert("RGB").save(buf, format="JPEG", quality=70)
                thumbnail_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        pass  # 没装 Pillow：跳过尺寸/缩略图，abs_path 仍然可用

    return {
        "abs_path": str(source),
        "relative_path": source.relative_to(root).as_posix(),
        "mime_type": _MIME_MAP[ext],
        "num_bytes": num_bytes,
        "width": width,
        "height": height,
        "thumbnail_base64": thumbnail_base64,
    }
