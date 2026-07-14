"""
b4_model_switch.py — 支持在一次 Agent 运行中切换不同本地模型配置的模块。

提供函数：
  - load_model_roster: 读取 model_roster.yaml 返回 roster dict
  - select_model_for_task: 根据任务类型选择合适的 model_config 路径
  - generate_with_model_selection: 选择模型后调用 b4_local_agent_llm.generate_ai_message

CLI:
  python b4_model_switch.py \\
    --roster ../configs/model_roster.yaml \\
    --task_type plan \\
    --messages ../data/messages/messages_no_tool.json \\
    --tools_schema ../data/messages/tools_schema_basic.json \\
    --mode mock \\
    --outdir ../outputs/B4_model_switch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# project root bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.io_utils import ensure_dir, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file

import b4_local_agent_llm


# ---------------------------------------------------------------------------
# Task-type → roster key mapping
# ---------------------------------------------------------------------------

_TASK_TO_ROSTER_KEY: dict[str, str] = {
    "plan": "default",
    "execute": "fast",
    "summarize": "strict",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_model_roster(roster_path: str) -> dict:
    """
    读取 model_roster.yaml，返回解析后的 dict。

    期望格式：
      roster:
        default:
          model_config: model.yaml
        fast:
          model_config: model.yaml
          generation_overrides:
            max_new_tokens: 256
        strict:
          model_config: model.yaml
          generation_overrides:
            temperature: 0
            max_new_tokens: 512
    """
    raw = read_yaml(roster_path)
    if not isinstance(raw, dict):
        raise ValueError(f"model_roster.yaml must contain an object, got: {type(raw).__name__}")
    roster = raw.get("roster")
    if not isinstance(roster, dict):
        raise ValueError("model_roster.yaml must have a top-level 'roster' key containing an object")
    return raw  # 返回完整 dict（含 roster key），方便调用方直接用 roster["roster"]


def select_model_for_task(task_type: str, roster: dict) -> str:
    """
    根据任务类型从 roster 中选择合适的 model_config 路径。

    task_type 映射规则：
      - "plan"      → "default"
      - "execute"   → "fast"（不存在则 fallback 到 "default"）
      - "summarize" → "strict"（不存在则 fallback 到 "default"）
      - 其他未知类型 → fallback 到 "default"

    roster 参数既可以是 load_model_roster 返回的完整 dict（含 "roster" key），
    也可以是直接传入的 roster 子 dict（不含外层 key）。

    返回 model_config 路径字符串（来自 roster 条目的 model_config 字段）。
    """
    # 处理两种传入格式
    if "roster" in roster:
        roster_entries: dict = roster["roster"]
    else:
        roster_entries = roster

    preferred_key = _TASK_TO_ROSTER_KEY.get(task_type, "default")

    # 尝试首选 key，fallback 到 default
    entry = roster_entries.get(preferred_key) or roster_entries.get("default")
    if entry is None:
        # roster 完全为空时，使用第一个条目
        entry = next(iter(roster_entries.values()), None)
    if entry is None:
        raise ValueError("model_roster 中没有可用的模型条目")

    model_config_path = entry.get("model_config")
    if not isinstance(model_config_path, str) or not model_config_path:
        raise ValueError(f"roster 条目缺少有效的 model_config 字段: {entry}")
    return model_config_path


def generate_with_model_selection(
    task_type: str,
    roster: dict,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str,
    artifact_dir: str | None,
    artifact_stem: str | None,
) -> dict:
    """
    选择合适的模型并调用 b4_local_agent_llm.generate_ai_message。
    返回和 generate_ai_message 相同的格式：
      {"ai_message": {...}, "status": "success"|"error", "error": ...}
    """
    model_config_path = select_model_for_task(task_type, roster)
    return b4_local_agent_llm.generate_ai_message(
        model_config=model_config_path,
        messages=messages,
        tools_schema=tools_schema,
        mode=mode,
        artifact_dir=artifact_dir,
        artifact_stem=artifact_stem,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-switching demo: select a model based on task type and run generate_ai_message."
    )
    parser.add_argument("--roster", required=True, help="Path to model_roster.yaml")
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["plan", "execute", "summarize"],
        help="Task type to select the model for",
    )
    parser.add_argument("--messages", required=True, help="Path to messages JSON file")
    parser.add_argument("--tools_schema", required=True, help="Path to tools_schema JSON file")
    parser.add_argument("--mode", choices=["mock", "prompt_json"], default="mock", help="LLM mode")
    parser.add_argument("--outdir", required=True, help="Directory to save outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        roster_path = resolve_cli_path(args.roster)
        messages_path = resolve_cli_path(args.messages)
        tools_schema_path = resolve_cli_path(args.tools_schema)
        outdir = resolve_cli_path(args.outdir)
        ensure_dir(outdir)

        # 加载 roster
        roster = load_model_roster(str(roster_path))

        # 选择模型
        selected_model_config = select_model_for_task(args.task_type, roster)
        print(f"[b4_model_switch] task_type={args.task_type!r} → model_config={selected_model_config!r}")

        # 解析 model_config 路径（相对于 roster 文件所在目录）
        model_config_abs = resolve_from_file(selected_model_config, roster_path)
        print(f"[b4_model_switch] resolved model_config -> {model_config_abs}")

        messages = read_json(messages_path)
        tools_schema = read_json(tools_schema_path)

        # 调用 generate_ai_message
        result = b4_local_agent_llm.generate_ai_message(
            model_config=str(model_config_abs),
            messages=messages,
            tools_schema=tools_schema,
            mode=args.mode,
            artifact_dir=str(outdir),
            artifact_stem=None,
        )

        # 保存 ai_message.json（generate_ai_message 已通过 artifact_dir 保存，
        # 此处额外追加 model_config 元信息并重写）
        ai_message_path = outdir / "ai_message.json"
        ai_message_record = {
            "task_type": args.task_type,
            "selected_model_config": selected_model_config,
            "resolved_model_config": str(model_config_abs),
            "mode": args.mode,
            "generated_at": now_iso(),
            "ai_message": result["ai_message"],
            "status": result["status"],
            "error": result.get("error"),
        }
        write_json(ai_message_record, ai_message_path)

        print(f"[b4_model_switch] ai_message.json -> {ai_message_path}")
        print(f"[b4_model_switch] status={result['status']}")
        print(f"[b4_model_switch] model_config (in record)={selected_model_config!r}")

        if result.get("error"):
            print(f"[b4_model_switch] error={result['error']}", file=sys.stderr)

        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
