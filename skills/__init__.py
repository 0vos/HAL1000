from __future__ import annotations

from pathlib import Path


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def resolve_data_path(path: str, data_root: str | None = None) -> tuple[Path, Path]:
    root = Path(data_root).resolve() if data_root else DEFAULT_DATA_ROOT.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # 相对路径：相对 data_root 解析
        candidate = (root / candidate).resolve()
    else:
        # 绝对路径：直接使用，不检查是否在 data_root 内
        candidate = candidate.resolve()
    return candidate, root
