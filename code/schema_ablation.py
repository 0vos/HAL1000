"""30-sample ablation suite: detailed vs minimal schema description.

Each sample has:
    - id
    - user_query: 中文用户问题
    - expected_tools: list of {tool_name, required_args_subset, optional_args_subset}

We feed the user query + schema into the LLM (mock or prompt_json) and
check whether the returned tool_calls match the expected tools.

Run with:
    python schema_ablation.py --mode mock
    python schema_ablation.py --mode prompt_json --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common.path_utils
common.path_utils.bootstrap_project_root()


SAMPLES: list[dict] = [
    {
        "id": "s01",
        "user_query": "帮我阅读 docs/agent_intro.txt，总结三条中文要点。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "docs/agent_intro.txt"}}],
    },
    {
        "id": "s02",
        "user_query": "计算 (123 + 456) * 7 - 89 的结果。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "(123 + 456) * 7 - 89"}}],
    },
    {
        "id": "s03",
        "user_query": "搜索 docs 目录下提到 Agent 的文件。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "Agent"}}],
    },
    {
        "id": "s04",
        "user_query": "读取 data/tables/results.csv 并给我前 5 行预览和数值统计。",
        "expected": [{"tool_name": "table_analyzer", "required": {"path": "tables/results.csv"}}],
    },
    {
        "id": "s05",
        "user_query": "把以下文本转成 markdown 项目符号列表：\\nAgent 系统\\n模型与工具\\n记忆模块",
        "expected": [{"tool_name": "format_converter", "required": {"text": "Agent 系统\\n模型与工具\\n记忆模块", "target_format": "markdown"}}],
    },
    {
        "id": "s06",
        "user_query": "请先阅读 README，然后帮我算 17 * 23 + 9 的结果。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "17 * 23 + 9"}}],
    },
    {
        "id": "s07",
        "user_query": "查找 docs 中所有含 memory 关键字的文件。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "memory"}}],
    },
    {
        "id": "s08",
        "user_query": "把这段 JSON 转成格式化的 JSON: {\"k\": 1, \"v\": 2}",
        "expected": [{"tool_name": "format_converter", "required": {"text": "{\"k\": 1, \"v\": 2}", "target_format": "json"}}],
    },
    {
        "id": "s09",
        "user_query": "计算 2 的 16 次方。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "2 ** 16"}}],
    },
    {
        "id": "s10",
        "user_query": "阅读 docs/agent_intro.txt 给我看里面有什么。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "docs/agent_intro.txt"}}],
    },
    {
        "id": "s11",
        "user_query": "搜索 Agent 工具调用 相关内容，列出最多 5 个文件。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "Agent 工具调用"}}],
    },
    {
        "id": "s12",
        "user_query": "把以下键值对转成 JSON：name: HAL1000\\ntype: agent",
        "expected": [{"tool_name": "format_converter", "required": {"text": "name: HAL1000\\ntype: agent", "target_format": "json"}}],
    },
    {
        "id": "s13",
        "user_query": "分析 results.csv 表，告诉我有多少列。",
        "expected": [{"tool_name": "table_analyzer", "required": {"path": "tables/results.csv"}}],
    },
    {
        "id": "s14",
        "user_query": "算一下 (3.14 * 2 + 1) / 2 的结果。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "(3.14 * 2 + 1) / 2"}}],
    },
    {
        "id": "s15",
        "user_query": "看一下 README.md 文件。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "README.md"}}],
    },
    {
        "id": "s16",
        "user_query": "把多行文本转成 markdown bullet：\\nfirst\\nsecond\\nthird",
        "expected": [{"tool_name": "format_converter", "required": {"text": "first\\nsecond\\nthird", "target_format": "markdown"}}],
    },
    {
        "id": "s17",
        "user_query": "在 docs 里搜素包含 execution 的文档。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "execution"}}],
    },
    {
        "id": "s18",
        "user_query": "给我计算 100 / 4 + 3。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "100 / 4 + 3"}}],
    },
    {
        "id": "s19",
        "user_query": "读取 docs/agent_intro.txt 给我内容。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "docs/agent_intro.txt"}}],
    },
    {
        "id": "s20",
        "user_query": "分析 tables/results.csv，给我每列的统计信息。",
        "expected": [{"tool_name": "table_analyzer", "required": {"path": "tables/results.csv"}}],
    },
    {
        "id": "s21",
        "user_query": "搜索提到 工具 的文档。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "工具"}}],
    },
    {
        "id": "s22",
        "user_query": "把以下 Python dict 转成格式化 JSON 字符串：{\"a\": [1,2,3], \"b\": {\"x\": 1}}",
        "expected": [{"tool_name": "format_converter", "required": {"text": "{\"a\": [1,2,3], \"b\": {\"x\": 1}}", "target_format": "json"}}],
    },
    {
        "id": "s23",
        "user_query": "算一下 999 + 1 的值。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "999 + 1"}}],
    },
    {
        "id": "s24",
        "user_query": "请阅读 data/docs/agent_intro.txt 文件。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "docs/agent_intro.txt"}}],
    },
    {
        "id": "s25",
        "user_query": "查看 results.csv 的前 10 行。",
        "expected": [{"tool_name": "table_analyzer", "required": {"path": "tables/results.csv"}}],
    },
    {
        "id": "s26",
        "user_query": "搜索 query 为 \"model\" 的文档。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "model"}}],
    },
    {
        "id": "s27",
        "user_query": "把以下内容转成 markdown 项目符号：\\nL1\\nL2\\nL3",
        "expected": [{"tool_name": "format_converter", "required": {"text": "L1\\nL2\\nL3", "target_format": "markdown"}}],
    },
    {
        "id": "s28",
        "user_query": "用 calculator 算 12 * 12。",
        "expected": [{"tool_name": "calculator", "required": {"expression": "12 * 12"}}],
    },
    {
        "id": "s29",
        "user_query": "读取 README 文件以了解项目。",
        "expected": [{"tool_name": "file_reader", "required": {"path": "README"}}],
    },
    {
        "id": "s30",
        "user_query": "搜索 docs 中所有提到 loop 的文件路径。",
        "expected": [{"tool_name": "local_file_search", "required": {"query": "loop"}}],
    },
]


def minimal_schema(detailed_schema: list[dict]) -> list[dict]:
    """Replace each description with a minimal one-sentence tag."""
    out = []
    for tool in detailed_schema:
        cloned = deepcopy(tool)
        function = cloned.get("function", {})
        function["description"] = function.get("name", "tool")
        for prop in function.get("parameters", {}).get("properties", {}).values():
            prop["description"] = ""
        out.append(cloned)
    return out


def _args_match(actual: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _score(predicted_calls: list[dict], expected: list[dict]) -> dict:
    if not expected:
        return {"tool_match": predicted_calls == [], "exact_match": predicted_calls == []}
    matched = [e for e in expected if any(p["name"] == e["tool_name"] and _args_match(p["args"], e["required"]) for p in predicted_calls)]
    tool_match = len(matched) == len(expected)
    expected_calls = [{"name": e["tool_name"], "args": e["required"]} for e in expected]
    exact = False
    if tool_match and len(predicted_calls) == len(expected):
        for pred, exp in zip(predicted_calls, expected_calls):
            if pred["name"] != exp["name"] or not _args_match(pred["args"], exp["args"]):
                break
        else:
            exact = True
    return {"tool_match": tool_match, "exact_match": exact, "matched": len(matched), "expected": len(expected)}


def run_ablation(detailed_schema: list[dict], mode: str, model_config: str, limit: int | None) -> dict:
    from b4_local_agent_llm import generate_ai_message
    minimal = minimal_schema(detailed_schema)
    samples = SAMPLES if not limit else SAMPLES[:limit]
    detailed_results = []
    minimal_results = []
    for sample in samples:
        messages = [
            {"role": "system", "content": "You are a tool-using agent. Output JSON with content or tool_calls."},
            {"role": "user", "content": sample["user_query"]},
        ]
        for variant, schema in (("detailed", detailed_schema), ("minimal", minimal)):
            artifact_dir = f"/tmp/ablation_{sample['id']}_{variant}"
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            try:
                result = generate_ai_message(
                    model_config,
                    messages,
                    schema,
                    mode=mode,
                    artifact_dir=artifact_dir,
                    artifact_stem=f"{variant}_{sample['id']}",
                )
                ai_message = result.get("ai_message") or {}
                tool_calls = ai_message.get("tool_calls", [])
                content = ai_message.get("content", "")
                score = _score(tool_calls, sample["expected"])
                entry = {
                    "sample_id": sample["id"],
                    "variant": variant,
                    "status": result.get("status"),
                    "predicted_tools": [c["name"] for c in tool_calls],
                    "predicted_args": {c["name"]: c["args"] for c in tool_calls},
                    "tool_match": score["tool_match"],
                    "exact_match": score["exact_match"],
                    "expected_tools": [e["tool_name"] for e in sample["expected"]],
                    "content_len": len(content),
                }
            except Exception as exc:
                entry = {
                    "sample_id": sample["id"],
                    "variant": variant,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "tool_match": False,
                    "exact_match": False,
                }
            if variant == "detailed":
                detailed_results.append(entry)
            else:
                minimal_results.append(entry)
    return {"detailed": detailed_results, "minimal": minimal_results, "samples": samples}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare schema description variants on tool calling accuracy.")
    parser.add_argument("--mode", choices=["mock", "prompt_json"], required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--toolset", default="basic_tools")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from b3_tool_layer import get_tools_schema
        detailed = get_tools_schema(args.tools_config, args.toolset)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        minimal = minimal_schema(detailed)
        write_json_safe(detailed, outdir / "schema_detailed.json")
        write_json_safe(minimal, outdir / "schema_minimal.json")
        result = run_ablation(detailed, args.mode, args.model_config, args.limit)
        write_json_safe(result["detailed"], outdir / "results_detailed.jsonl")
        write_json_safe(result["minimal"], outdir / "results_minimal.jsonl")
        # Aggregate
        def aggregate(entries):
            total = len(entries)
            tool_match = sum(1 for e in entries if e.get("tool_match"))
            exact_match = sum(1 for e in entries if e.get("exact_match"))
            errors = sum(1 for e in entries if e.get("status") != "success")
            avg_content_len = sum(e.get("content_len", 0) for e in entries) / max(total, 1)
            return {
                "total": total,
                "tool_match_rate": round(tool_match / total, 4) if total else 0.0,
                "exact_match_rate": round(exact_match / total, 4) if total else 0.0,
                "errors": errors,
                "avg_response_content_len": round(avg_content_len, 1),
            }
        comparison = {
            "mode": args.mode,
            "samples_count": len(result["samples"]),
            "detailed": aggregate(result["detailed"]),
            "minimal": aggregate(result["minimal"]),
            "delta_tool_match": round(aggregate(result["detailed"])["tool_match_rate"] - aggregate(result["minimal"])["tool_match_rate"], 4),
            "delta_exact_match": round(aggregate(result["detailed"])["exact_match_rate"] - aggregate(result["minimal"])["exact_match_rate"], 4),
        }
        write_json_safe(comparison, outdir / "comparison.json")
        write_markdown_summary(comparison, outdir / "comparison.md")
        print(outdir / "comparison.md")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def write_json_safe(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list):
        with path.open("w", encoding="utf-8") as f:
            for entry in obj:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_summary(comparison: dict, path: Path) -> None:
    md = []
    md.append(f"# Schema Ablation Summary (mode = {comparison['mode']})")
    md.append("")
    md.append(f"Samples: **{comparison['samples_count']}**")
    md.append("")
    md.append("| Variant | Tool Match | Exact Match | Errors | Avg Content Len |")
    md.append("|---|---|---|---|---|")
    for name in ("detailed", "minimal"):
        agg = comparison[name]
        md.append(f"| {name} | {agg['tool_match_rate']:.2%} ({agg['tool_match_rate']*agg['total']:.0f}/{agg['total']}) | {agg['exact_match_rate']:.2%} | {agg['errors']} | {agg['avg_response_content_len']} |")
    md.append("")
    md.append(f"Δ tool_match = detailed - minimal = **{comparison['delta_tool_match']:+.2%}**")
    md.append(f"Δ exact_match = detailed - minimal = **{comparison['delta_exact_match']:+.2%}**")
    md.append("")
    md.append("## Observations")
    md.append("")
    if comparison["delta_tool_match"] > 0:
        md.append("- Detailed descriptions **improve** tool selection rate over the minimal variant.")
    elif comparison["delta_tool_match"] < 0:
        md.append("- Minimal descriptions **outperform** the detailed variant — extra text may distract the model.")
    else:
        md.append("- Both variants produce identical tool-selection rates.")
    if comparison["delta_exact_match"] > 0:
        md.append("- Detailed descriptions yield more **fully-correct** tool calls (matching args).")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())