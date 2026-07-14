"""
b4_plan_execute.py — Plan-and-Execute Agent 演示模块。

概念：先让 LLM（此处为 Mock）生成一个执行计划（plan），再逐步执行计划中的每个步骤。

提供函数：
  - mock_planner: 生成执行计划 dict
  - execute_plan: 逐步执行计划，返回结构化结果
  - run_plan_execute_demo: 完整演示入口

CLI:
  python b4_plan_execute.py \\
    --model_config ../configs/model.yaml \\
    --tools_schema ../data/messages/tools_schema_basic.json \\
    --outdir ../outputs/B4_plan_execute \\
    [--user_input "请读取 agent_intro.txt 并计算 1+1"]
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

from common.io_utils import ensure_dir, read_json, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path
from common.schemas import make_ai_message, make_tool_message


# ---------------------------------------------------------------------------
# Mock SkillResult payloads
# ---------------------------------------------------------------------------

def _make_file_reader_result(path: str, max_chars: int = 2000) -> str:
    return json.dumps(
        {
            "skill_name": "file_reader",
            "status": "success",
            "input": {"path": path, "max_chars": max_chars},
            "output": {
                "content": (
                    "Agent 系统通常由模型、工具、记忆和执行循环组成。\n"
                    "工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答。\n"
                    "Memory 为 Agent 提供全局知识和历史对话上下文。"
                ),
                "num_chars": 85,
                "source": path,
                "truncated": False,
            },
            "error": None,
            "latency_ms": 1.0,
        },
        ensure_ascii=False,
    )


def _make_calculator_result(expression: str) -> str:
    # 简单 mock 计算：仅支持加法，否则返回 mock 结果
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307 — mock only, controlled input
        if not isinstance(result, (int, float)):
            raise ValueError("unsupported")
    except Exception:
        result = 0
    return json.dumps(
        {
            "skill_name": "calculator",
            "status": "success",
            "input": {"expression": expression},
            "output": {"result": result, "expression": expression},
            "error": None,
            "latency_ms": 0.5,
        },
        ensure_ascii=False,
    )


def _make_generic_result(tool_name: str, args: dict) -> str:
    return json.dumps(
        {
            "skill_name": tool_name,
            "status": "success",
            "input": args,
            "output": {"result": "mock"},
            "error": None,
            "latency_ms": 1.0,
        },
        ensure_ascii=False,
    )


def _simulate_tool(tool_name: str, args: dict) -> str:
    """根据工具名称和参数生成 mock SkillResult 字符串。"""
    if tool_name == "file_reader":
        return _make_file_reader_result(
            path=args.get("path", "docs/agent_intro.txt"),
            max_chars=args.get("max_chars", 2000),
        )
    elif tool_name == "calculator":
        return _make_calculator_result(args.get("expression", "0"))
    else:
        return _make_generic_result(tool_name, args)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def mock_planner(user_input: str, tools_schema: list[dict]) -> dict:
    """
    Mock 规划器：根据 user_input 生成一个执行计划。

    返回：
    {
      "plan": [
        {"step": 1, "tool": "file_reader", "args": {...}, "reason": "..."},
        {"step": 2, "tool": "calculator", "args": {...}, "reason": "..."}
      ],
      "goal": "完成用户请求：{user_input}"
    }
    """
    # 从 tools_schema 中获取可用工具名称
    available_tools = {
        entry.get("function", {}).get("name", "")
        for entry in tools_schema
        if isinstance(entry, dict) and "function" in entry
    }

    # Mock 固定计划：file_reader + calculator（符合任务要求）
    plan_steps = [
        {
            "step": 1,
            "tool": "file_reader",
            "args": {"path": "docs/agent_intro.txt", "max_chars": 2000},
            "reason": "读取文件内容",
        },
        {
            "step": 2,
            "tool": "calculator",
            "args": {"expression": "1+1"},
            "reason": "验证计算能力",
        },
    ]

    # 仅保留 tools_schema 中实际存在的工具步骤（健壮性）
    valid_steps = [s for s in plan_steps if s["tool"] in available_tools] if available_tools else plan_steps

    return {
        "plan": valid_steps,
        "goal": f"完成用户请求：{user_input}",
    }


def execute_plan(
    plan: dict,
    tools_schema: list[dict],
    model_config: str,
    outdir: str,
) -> dict:
    """
    逐步执行 plan 中的每个 step：
      1. 对每个 step，构造一个只包含该 step 的 AIMessage（带 tool_calls）
      2. 生成对应的 mock ToolMessage
      3. 记录每步的输入/输出
      4. 最后生成汇总回答

    返回：
    {
      "plan": {...},
      "steps": [
        {"step": 1, "tool_call": {...}, "tool_result": {...}, "status": "success"},
        ...
      ],
      "final_answer": "...",
      "status": "success"
    }
    """
    steps_results: list[dict] = []
    step_summaries: list[str] = []

    for step_def in plan.get("plan", []):
        step_num = step_def["step"]
        tool_name = step_def["tool"]
        tool_args = step_def.get("args", {})
        reason = step_def.get("reason", "")

        # 构造只含该 step 的 AIMessage
        call_id = f"call_{step_num:03d}"
        tool_call_dict = {"id": call_id, "name": tool_name, "args": tool_args}
        ai_message = make_ai_message("", [tool_call_dict])

        # 模拟执行工具，生成 ToolMessage
        mock_content = _simulate_tool(tool_name, tool_args)
        tool_msg = make_tool_message(
            tool_call_id=call_id,
            name=tool_name,
            content=mock_content,
            status="success",
        )

        # 解析 ToolMessage 结果
        try:
            tool_result = json.loads(mock_content)
        except json.JSONDecodeError:
            tool_result = {"raw": mock_content}

        steps_results.append(
            {
                "step": step_num,
                "tool": tool_name,
                "reason": reason,
                "tool_call": tool_call_dict,
                "ai_message": ai_message,
                "tool_message": tool_msg,
                "tool_result": tool_result,
                "status": "success",
            }
        )

        # 生成步骤摘要
        output = tool_result.get("output") or {}
        if tool_name == "file_reader":
            content_text = output.get("content", "（无内容）")
            step_summaries.append(f"步骤 {step_num}（{reason}）：已读取文件，内容：{content_text.strip()[:60]}…")
        elif tool_name == "calculator":
            expr = output.get("expression", "?")
            res = output.get("result", "?")
            step_summaries.append(f"步骤 {step_num}（{reason}）：计算 {expr} = {res}")
        else:
            step_summaries.append(f"步骤 {step_num}（{reason}）：{tool_name} 执行完成，输出：{json.dumps(output, ensure_ascii=False)[:60]}")

    # 生成汇总回答
    final_lines = [
        f"执行计划完成，目标：{plan.get('goal', '')}",
        "",
        "执行步骤汇总：",
    ]
    for summary in step_summaries:
        final_lines.append(f"- {summary}")

    final_answer = "\n".join(final_lines)

    return {
        "plan": plan,
        "steps": steps_results,
        "final_answer": final_answer,
        "status": "success",
    }


def run_plan_execute_demo(
    model_config: str,
    tools_schema_path: str,
    outdir: str,
    user_input: str = "请读取 agent_intro.txt 并计算 1+1",
) -> dict:
    """
    完整演示：
      1. 调用 mock_planner 生成 plan
      2. 调用 execute_plan 执行
      3. 保存 plan.json、steps.json、final_answer.md 到 outdir
      4. 返回完整结果
    """
    out_path = Path(outdir)
    ensure_dir(out_path)

    tools_schema = read_json(tools_schema_path)

    # Step 1: 生成计划
    plan = mock_planner(user_input, tools_schema)

    # Step 2: 执行计划
    result = execute_plan(plan, tools_schema, model_config, outdir)

    # Step 3: 保存输出
    write_json(plan, out_path / "plan.json")
    print(f"[b4_plan_execute] plan.json -> {out_path / 'plan.json'}")

    # steps.json — 仅序列化可 JSON 化的字段（去掉内部 schema 对象）
    steps_serializable = []
    for s in result["steps"]:
        steps_serializable.append(
            {
                "step": s["step"],
                "tool": s["tool"],
                "reason": s["reason"],
                "tool_call": s["tool_call"],
                "tool_result": s["tool_result"],
                "status": s["status"],
            }
        )
    write_json(steps_serializable, out_path / "steps.json")
    print(f"[b4_plan_execute] steps.json -> {out_path / 'steps.json'}")

    write_text(result["final_answer"], out_path / "final_answer.md")
    print(f"[b4_plan_execute] final_answer.md -> {out_path / 'final_answer.md'}")
    print(f"[b4_plan_execute] final_answer:\n{result['final_answer']}")

    return {
        "plan": plan,
        "steps": result["steps"],
        "final_answer": result["final_answer"],
        "status": "success",
        "generated_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan-and-Execute Agent demo (mock mode, no real LLM required)."
    )
    parser.add_argument("--model_config", required=True, help="Path to model.yaml")
    parser.add_argument("--tools_schema", required=True, help="Path to tools_schema JSON file")
    parser.add_argument("--outdir", required=True, help="Directory to save outputs")
    parser.add_argument(
        "--user_input",
        default="请读取 agent_intro.txt 并计算 1+1",
        help="User request for the planner",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model_config = str(resolve_cli_path(args.model_config))
        tools_schema_path = str(resolve_cli_path(args.tools_schema))
        outdir = str(resolve_cli_path(args.outdir))
        run_plan_execute_demo(model_config, tools_schema_path, outdir, args.user_input)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
