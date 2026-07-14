"""
b4_multi_tool.py — 支持单轮多个 tool_calls 及单轮接收多个 ToolMessage 的演示模块。

新功能：
  - generate_multi_tool_mock: Mock 生成包含 2 个 tool_calls 的 AIMessage，或根据 ToolMessage 生成最终回答
  - demo_multi_tool_round: 完整演示多工具调用轮次（独立，不依赖 B1）

CLI:
  python b4_multi_tool.py \\
    --model_config ../configs/model.yaml \\
    --tools_schema ../data/messages/tools_schema_basic.json \\
    --outdir ../outputs/B4_multi_tool
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

from common.io_utils import ensure_dir, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path
from common.schemas import make_ai_message, make_tool_message


# ---------------------------------------------------------------------------
# Mock SkillResult payloads
# ---------------------------------------------------------------------------

_FILE_READER_MOCK_RESULT = json.dumps(
    {
        "skill_name": "file_reader",
        "status": "success",
        "input": {"path": "docs/agent_intro.txt", "max_chars": 2000},
        "output": {
            "content": (
                "Agent 系统通常由模型、工具、记忆和执行循环组成。\n"
                "工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答。\n"
                "Memory 为 Agent 提供全局知识和历史对话上下文。"
            ),
            "num_chars": 85,
            "source": "docs/agent_intro.txt",
            "truncated": False,
        },
        "error": None,
        "latency_ms": 1.0,
    },
    ensure_ascii=False,
)

_CALCULATOR_MOCK_RESULT = json.dumps(
    {
        "skill_name": "calculator",
        "status": "success",
        "input": {"expression": "2+2"},
        "output": {"result": 4, "expression": "2+2"},
        "error": None,
        "latency_ms": 0.5,
    },
    ensure_ascii=False,
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def generate_multi_tool_mock(messages: list[dict], tools_schema: list[dict]) -> dict:
    """
    Mock 模式：
    - 若消息列表中尚无 ToolMessage（第一轮），返回包含 2 个 tool_calls 的 AIMessage：
        1. file_reader: 读取 docs/agent_intro.txt
        2. calculator: 计算 "2+2"
    - 若已有 ToolMessage（第二轮），汇总所有工具结果，生成最终回答。

    返回标准 AIMessage dict（role/content/tool_calls）。
    """
    tool_messages = [m for m in messages if m.get("role") == "tool"]

    if not tool_messages:
        # 第一轮：生成 2 个 tool_calls
        return make_ai_message(
            "",
            [
                {
                    "id": "call_001",
                    "name": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 2000},
                },
                {
                    "id": "call_002",
                    "name": "calculator",
                    "args": {"expression": "2+2"},
                },
            ],
        )

    # 第二轮：汇总所有工具结果，生成最终回答
    summaries: list[str] = []
    for tm in tool_messages:
        try:
            result = json.loads(tm["content"])
        except (KeyError, json.JSONDecodeError, TypeError):
            summaries.append(f"工具 {tm.get('name', '?')} 返回了无法解析的结果。")
            continue

        skill_name = result.get("skill_name", tm.get("name", "?"))
        status = result.get("status", "unknown")

        if status != "success":
            err = result.get("error") or {}
            detail = err.get("message", "未知错误") if isinstance(err, dict) else str(err)
            summaries.append(f"工具 {skill_name} 执行失败：{detail}")
            continue

        output = result.get("output") or {}
        if skill_name == "file_reader":
            content_text = output.get("content", "（无内容）")
            summaries.append(f"文件读取结果：{content_text.strip()}")
        elif skill_name == "calculator":
            expr = output.get("expression", "?")
            res = output.get("result", "?")
            summaries.append(f"计算结果：{expr} = {res}")
        else:
            summaries.append(f"工具 {skill_name} 输出：{json.dumps(output, ensure_ascii=False)}")

    answer_lines = ["工具调用已完成，以下是汇总结果：", ""]
    for idx, s in enumerate(summaries, 1):
        answer_lines.append(f"{idx}. {s}")

    return make_ai_message("\n".join(answer_lines), [])


def demo_multi_tool_round(
    model_config: str,
    tools_schema: list[dict],
    outdir: str,
) -> dict:
    """
    演示完整的多工具调用轮次（独立演示，不依赖 B1）：
      1. 构造初始 messages（system + user）
      2. 调用 generate_multi_tool_mock 获得 AIMessage（含 2 个 tool_calls）
      3. 模拟执行每个 tool_call，生成对应 ToolMessage（mock 内容）
      4. 把 AIMessage + ToolMessages 追加到 messages
      5. 再次调用 generate_multi_tool_mock 获得最终回答
      6. 保存 messages.json 和 demo_report.json 到 outdir
      7. 返回 {"messages": [...], "final_answer": "...", "status": "success"}
    """
    out_path = Path(outdir)
    ensure_dir(out_path)

    # Step 1: 构造初始 messages
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a local tool-using agent. You may call multiple tools in one turn. "
                "Wait for all ToolMessages before generating a final answer."
            ),
        },
        {
            "role": "user",
            "content": "请读取 docs/agent_intro.txt 的内容，并计算 2+2。",
        },
    ]

    # Step 2: 第一轮调用 — 获得含 2 个 tool_calls 的 AIMessage
    first_ai_message = generate_multi_tool_mock(messages, tools_schema)
    messages.append(first_ai_message)

    # Step 3 & 4: 模拟执行每个 tool_call，生成对应 ToolMessage
    _mock_tool_results: dict[str, str] = {
        "file_reader": _FILE_READER_MOCK_RESULT,
        "calculator": _CALCULATOR_MOCK_RESULT,
    }

    for tool_call in first_ai_message.get("tool_calls", []):
        call_id = tool_call["id"]
        tool_name = tool_call["name"]
        mock_content = _mock_tool_results.get(
            tool_name,
            json.dumps(
                {
                    "skill_name": tool_name,
                    "status": "success",
                    "input": tool_call.get("args", {}),
                    "output": {"result": "mock"},
                    "error": None,
                    "latency_ms": 1.0,
                },
                ensure_ascii=False,
            ),
        )
        tool_msg = make_tool_message(
            tool_call_id=call_id,
            name=tool_name,
            content=mock_content,
            status="success",
        )
        messages.append(tool_msg)

    # Step 5: 第二轮调用 — 汇总生成最终回答
    final_ai_message = generate_multi_tool_mock(messages, tools_schema)
    messages.append(final_ai_message)
    final_answer = final_ai_message.get("content", "")

    # Step 6: 保存输出
    write_json(messages, out_path / "messages.json")

    demo_report = {
        "demo": "b4_multi_tool",
        "generated_at": now_iso(),
        "model_config": model_config,
        "num_messages": len(messages),
        "num_tool_calls": len(first_ai_message.get("tool_calls", [])),
        "final_answer": final_answer,
        "status": "success",
    }
    write_json(demo_report, out_path / "demo_report.json")

    print(f"[b4_multi_tool] messages.json  -> {out_path / 'messages.json'}")
    print(f"[b4_multi_tool] demo_report.json -> {out_path / 'demo_report.json'}")
    print(f"[b4_multi_tool] final_answer: {final_answer[:120]!r}")

    return {
        "messages": messages,
        "final_answer": final_answer,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate multi-tool-call support (mock mode, no real LLM required)."
    )
    parser.add_argument("--model_config", required=True, help="Path to model.yaml (used for metadata only in mock mode)")
    parser.add_argument("--tools_schema", required=True, help="Path to tools_schema JSON file")
    parser.add_argument("--outdir", required=True, help="Directory to save outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tools_schema = read_json(resolve_cli_path(args.tools_schema))
        outdir = str(resolve_cli_path(args.outdir))
        model_config = str(resolve_cli_path(args.model_config))
        result = demo_multi_tool_round(model_config, tools_schema, outdir)
        print(f"[b4_multi_tool] status={result['status']}")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
