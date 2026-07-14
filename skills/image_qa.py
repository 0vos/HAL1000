"""image_qa.py — 视觉问答 skill：读取本地图片 + 用户问题，调用 Qwen3.5 视觉能力生成回答。

设计：
  - 复用 image_reader 做路径校验（格式/大小/存在性检查）
  - 调用 b4_local_agent_llm.generate_vision_answer 做真实推理
  - mode 由 tools.yaml 传入的 model_config 决定（跟其他工具一致的模式：
    这个 skill 本身不知道 mock/prompt_json，由 task_executor / hal_chat 在
    调用时决定是否真的跑模型；这里默认 prompt_json，因为视觉问答没有
    意义明确的 mock 行为）
"""
from __future__ import annotations

import sys
from pathlib import Path

from skills.image_reader import image_reader
from skills_error_codes import ErrorCode, attach_error_code

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def image_qa(
    path: str,
    question: str,
    *,
    data_root: str | None = None,
    model_config: str | None = None,
    mode: str | None = None,
    max_new_tokens: int = 512,
) -> dict:
    """
    对本地图片提问，返回模型的文本回答。

    Args:
        path: 图片路径，相对于 data_root
        question: 用户问题，例如"这张图里有什么？"
        model_config: model.yaml 路径，默认使用 configs/model.yaml
        mode: mock | prompt_json。为 None 时自动读取 model.yaml 里的
              runtime.default_mode（与 b1_agent_runtime.py 的 _default_llm_mode
              逻辑一致），避免写死 prompt_json 导致 mock 会话中调用真实模型
        max_new_tokens: 生成的最大 token 数

    Returns:
        {
          "answer": str,
          "image_meta": {...},   # image_reader 的输出（不含缩略图）
        }
    """
    if not isinstance(question, str) or not question.strip():
        raise attach_error_code(ValueError("question must be a non-empty string"), ErrorCode.INVALID_INPUT)

    # 先校验图片存在/格式/大小，拿到绝对路径
    meta = image_reader(path, data_root=data_root, include_thumbnail_base64=False)

    project_root = Path(__file__).resolve().parents[1]
    if model_config is None:
        model_config = str(project_root / "configs" / "model.yaml")

    if mode is None:
        try:
            from common.io_utils import read_yaml
            cfg = read_yaml(model_config)
            mode = cfg.get("runtime", {}).get("default_mode", "mock")
        except Exception:
            mode = "mock"

    from b4_local_agent_llm import generate_vision_answer

    answer = generate_vision_answer(
        model_config=model_config,
        image_path=meta["abs_path"],
        question=question,
        mode=mode,
        max_new_tokens=max_new_tokens,
    )

    return {
        "answer": answer.strip(),
        "image_meta": {
            "relative_path": meta["relative_path"],
            "mime_type": meta["mime_type"],
            "num_bytes": meta["num_bytes"],
            "width": meta["width"],
            "height": meta["height"],
        },
    }
