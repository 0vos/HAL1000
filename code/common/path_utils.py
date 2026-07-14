from __future__ import annotations

import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def bootstrap_project_root() -> Path:
    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return PROJECT_ROOT


def expand_placeholders(value: str) -> str:
    if not isinstance(value, str) or "${" not in value:
        return value

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None and env_value != "":
            return env_value
        return default if default is not None else ""

    return _ENV_PLACEHOLDER.sub(replace, value)


def expand_env(value):
    if isinstance(value, str):
        return expand_placeholders(value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate.resolve()


def resolve_cli_path(path: str | Path) -> Path:
    return resolve_path(path, Path.cwd())


def resolve_from_file(path: str | Path, containing_file: str | Path) -> Path:
    return resolve_path(path, Path(containing_file).resolve().parent)


def resolve_model_path(raw: str | Path, base_dir: str | Path) -> Path:
    """Resolve a model path that may contain ${VAR} / ${VAR:-default} placeholders.

    Order of resolution:
    1. If the (expanded) path is absolute and exists, use it.
    2. Otherwise, fall back to a few well-known locations:
       - $HAL_MODEL_PATH environment variable (already expanded)
       - base_dir / raw (relative resolution)
       - PROJECT_ROOT.parent / <basename>
       - PROJECT_ROOT / models / <basename>
    """
    if raw is None:
        raise ValueError("model path must not be empty")
    text = expand_placeholders(str(raw)).strip()
    if not text:
        raise ValueError("model path expanded to an empty string")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
    else:
        relative = (Path(base_dir) / candidate).resolve()
        if relative.exists():
            return relative
    env_path = os.environ.get("HAL_MODEL_PATH", "").strip()
    if env_path:
        env_candidate = Path(env_path).expanduser().resolve()
        if env_candidate.exists():
            return env_candidate
    basename = candidate.name or text
    sibling = (PROJECT_ROOT.parent / basename).resolve()
    if sibling.exists():
        return sibling
    models_dir = (PROJECT_ROOT / "models" / basename).resolve()
    if models_dir.exists():
        return models_dir
    searched = [
        text,
        str((Path(base_dir) / candidate).resolve()) if not candidate.is_absolute() else text,
        env_path or "<unset>",
        str(sibling),
        str(models_dir),
    ]
    raise FileNotFoundError(
        "model path not found. Searched locations: " + ", ".join(searched)
    )


def require_within(path: str | Path, root: str | Path) -> Path:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes data root: {path}") from exc
    return resolved_path