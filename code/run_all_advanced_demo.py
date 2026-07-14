"""
run_all_advanced_demo.py — 一键跑完所有 B1~B4 基础 + 进阶功能演示。

用法（在 code/ 目录下执行）：
    python run_all_advanced_demo.py \\
        --model_path /root/siton-tmp/HAL1000/Qwen3.5-4B \\
        --outdir     ../outputs/all_demo \\
        [--llm_mode  mock]          # 没有 GPU 时加这个，默认 prompt_json

覆盖的功能：
    B1 基础     : fixture 模式 Agent 完整运行（不调用模型，验证流程）
    B1 integrated: integrated 模式完整运行（真实调用 LLM，llm_mode 真正生效）
    B1 进阶1 : 批量任务（batch runner）
    B1 进阶2 : 历史消息压缩（compress）
    B1 进阶3 : 断点续跑（checkpoint → resume）
    B1 进阶4 : system prompt 模板切换（prompt_patches）
    B4 基础  : 真实 llm_mode 调用 generate_ai_message（mock 或 prompt_json）
    B4 进阶1 : 单轮多 tool_calls（multi_tool）
    B4 进阶2 : Plan-and-Execute
    B4 进阶3 : 模型切换（model_switch）
    B2 基础  : 5 个 skill 各跑一次
    B2 进阶  : 复合 skill + 沙箱执行
    B3 基础  : 生成 tools_schema + 执行 tool_calls
    B3 进阶  : auto_schema + retry + cache + stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# bootstrap: 把 code/ 加入 sys.path
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.io_utils import ensure_dir, read_json, write_json, write_text, append_jsonl
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path

# ============================================================
# 工具函数
# ============================================================

_RESULTS: list[dict] = []

def _section(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)

def _ok(label: str, detail: str = "") -> None:
    msg = f"  [OK]  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)

def _fail(label: str, err: str) -> None:
    print(f"  [FAIL] {label}")
    print(f"         {err}")

def _run(label: str, fn, *args, **kwargs):
    """执行一个演示步骤，记录结果。返回 (success, result_or_None)。"""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        _ok(label, f"{elapsed} ms")
        _RESULTS.append({"label": label, "status": "ok", "elapsed_ms": elapsed})
        return True, result
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        err_text = f"{type(exc).__name__}: {exc}"
        _fail(label, err_text)
        _RESULTS.append({"label": label, "status": "fail", "elapsed_ms": elapsed, "error": err_text})
        if os.environ.get("DEMO_TRACEBACK"):
            traceback.print_exc()
        return False, None


# ============================================================
# 路径常量（相对 code/ 解析）
# ============================================================

ROOT        = HERE.parent
CONFIGS     = ROOT / "configs"
DATA        = ROOT / "data"
PROMPTS     = ROOT / "prompts"
FIXTURES    = DATA / "b1_fixtures"
MESSAGES    = DATA / "messages"
TOOL_INPUTS = DATA / "tool_inputs"


def _cfg(name: str) -> str:
    return str(CONFIGS / name)

def _data(name: str) -> str:
    return str(DATA / name)

def _fix(name: str) -> str:
    return str(FIXTURES / name)

def _msg(name: str) -> str:
    return str(MESSAGES / name)


# ============================================================
# B1 基础：fixture 模式完整 Agent 运行
# ============================================================

def demo_b1_fixture(outdir: Path, model_config: str, llm_mode: str) -> None:
    from b1_agent_runtime import run_agent
    result = run_agent(
        str(FIXTURES / "b1_fixture_input.json"),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        str(outdir),
        llm_mode,
    )
    assert result["status"] == "success", f"status={result['status']}"
    assert result["final_answer"].strip(), "final_answer is empty"


# ============================================================
# B1 进阶1：批量任务
# ============================================================

def demo_b1_batch(outdir: Path, model_config: str, llm_mode: str) -> None:
    from b1_batch_runner import run_batch
    summary = run_batch(
        str(DATA / "batch_input.json"),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        str(outdir),
        llm_mode,
    )
    assert summary["total"] >= 2, "expected at least 2 batch tasks"
    assert summary["success"] >= 1, "expected at least 1 successful task"


# ============================================================
# B1 进阶2：历史消息压缩
# ============================================================

def demo_b1_compress(outdir: Path) -> None:
    from b1_compress import maybe_compress_messages
    from common.io_utils import write_json

    # 构造一个够长的 messages 列表（10 条 non-system）
    messages = [{"role": "system", "content": "You are an agent."}]
    for i in range(10):
        if i % 2 == 0:
            messages.append({"role": "user", "content": f"User message {i}"})
        else:
            messages.append({
                "role": "assistant",
                "content": f"Assistant reply {i}",
                "tool_calls": [],
            })

    compressed, was_compressed = maybe_compress_messages(
        messages, compress_after=6, keep_recent=4
    )
    assert was_compressed, "expected compression to trigger"
    assert compressed[0]["role"] == "system", "first message must be system"

    ensure_dir(outdir)
    write_json(compressed, outdir / "compressed_messages.json")
    write_json({
        "was_compressed": was_compressed,
        "original_count": len(messages),
        "compressed_count": len(compressed),
    }, outdir / "compression_report.json")


# ============================================================
# B1 进阶3：断点续跑
# ============================================================

def demo_b1_checkpoint(outdir: Path, model_config: str, llm_mode: str) -> None:
    from b1_agent_runtime import run_agent
    from b1_checkpoint import load_checkpoint, save_checkpoint

    input_path = DATA / "runtime_input_checkpoint.json"
    # 正常跑完（fixture 模式会自动写 checkpoint 然后清理，此处验证正常完成即可）
    result = run_agent(
        str(input_path),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        str(outdir),
        llm_mode,
    )
    assert result["status"] == "success"

    # 手动写一个 mid-run checkpoint，然后用 resume 续跑验证接口
    mid_ckpt = {
        "conversation_id": "conv_checkpoint_test",
        "execution_mode": "fixture",
        "resume_from_turn": 0,
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "帮我阅读 docs/agent_intro.txt，总结三条中文要点。"},
        ],
        "tool_rounds": 0,
        "llm_calls": 0,
        "turns": [],
        "status": "running",
        "runtime_input": read_json(input_path),
    }
    # resolve system_prompt_path in runtime_input to absolute
    mid_ckpt["runtime_input"]["system_prompt_path"] = str(PROMPTS / "local_tool_agent.txt")
    for key in ("selected_memory_path", "tools_schema_path", "ai_messages_path", "tool_messages_path"):
        rel = mid_ckpt["runtime_input"]["fixtures"][key]
        # compute absolute path relative to input file parent
        abs_path = (input_path.parent / rel).resolve()
        mid_ckpt["runtime_input"]["fixtures"][key] = str(abs_path)

    resume_dir = outdir.parent / (outdir.name + "_resume")
    ensure_dir(resume_dir)
    save_checkpoint(str(resume_dir), mid_ckpt)
    assert load_checkpoint(str(resume_dir)) is not None, "checkpoint not saved"

    from b1_resume import resume_agent
    res2 = resume_agent(
        str(resume_dir),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        llm_mode,
    )
    assert res2["status"] == "success", f"resume status={res2['status']}"


# ============================================================
# B1 进阶4：system prompt 模板切换
# ============================================================

def demo_b1_prompt_patches(outdir: Path, model_config: str, llm_mode: str) -> None:
    from b1_agent_runtime import run_agent
    from common.io_utils import write_json

    # 基于 fixture input，注入 prompt_patches
    base = read_json(FIXTURES / "b1_fixture_input.json")
    patched_input = deepcopy(base)
    patched_input["conversation_id"] = "conv_prompt_patch_demo"
    patched_input["prompt_patches"] = [
        {"after_turn": 0, "append": "\n\n## 额外规则\n请用中文回答，回答要简洁。"},
    ]
    # resolve relative paths in fixtures to absolute
    for key in ("selected_memory_path", "tools_schema_path", "ai_messages_path", "tool_messages_path"):
        rel = patched_input["fixtures"][key]
        patched_input["fixtures"][key] = str((FIXTURES / rel).resolve())
    patched_input["system_prompt_path"] = str(PROMPTS / "local_tool_agent.txt")

    input_file = outdir / "runtime_input_patch.json"
    ensure_dir(outdir)
    write_json(patched_input, input_file)

    result = run_agent(
        str(input_file),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        str(outdir),
        llm_mode,
    )
    assert result["status"] == "success"


# ============================================================
# B1 integrated：真实 LLM 调用（integrated 模式，llm_mode 真正生效）
# ============================================================

def demo_b1_integrated(outdir: Path, model_config: str, llm_mode: str) -> None:
    """integrated 模式跑完整 Agent，真实调用 LLM（不是 fixture 预设）。"""
    from b1_agent_runtime import run_agent
    from common.io_utils import write_json

    runtime_input = {
        "conversation_id": "conv_integrated_demo",
        "execution_mode": "integrated",
        "user_input": "帮我阅读 docs/agent_intro.txt，总结三条中文要点。",
        "system_prompt_path": str(PROMPTS / "local_tool_agent.txt"),
        "selected_memory_ids": [],
        "use_global_memory": False,
        "toolset": "basic_tools",
        "max_turns": 3,
        "save_memory": "none",
    }
    ensure_dir(outdir)
    input_file = outdir / "runtime_input_integrated.json"
    write_json(runtime_input, input_file)

    result = run_agent(
        str(input_file),
        _cfg("tools.yaml"),
        _cfg("memory.yaml"),
        model_config,
        str(outdir),
        llm_mode,
    )
    # 验证 LLM 确实被调用：llm_calls/ 目录和 raw_model_output 文件必须存在
    llm_calls_dir = outdir / "llm_calls"
    assert llm_calls_dir.exists(), "llm_calls/ 目录不存在，模型可能未被调用"
    raw_outputs = list(llm_calls_dir.glob("*raw_model_output.json"))
    assert raw_outputs, "没有 raw_model_output.json，模型未被真实调用"
    raw = read_json(raw_outputs[0])
    print(f"  [LLM] mode={raw['mode']}, backend={raw['backend']}, status={raw['status']}")
    print(f"  [LLM] raw_text[:80]: {str(raw.get('raw_text', ''))[:80]}")
    assert result["status"] in ("success", "partial"), f"status={result['status']}"
    assert result["final_answer"].strip(), "final_answer is empty"


# ============================================================
# B4 基础：真实 llm_mode 调用 generate_ai_message
# ============================================================

def demo_b4_single_tool(outdir: Path, model_config: str, llm_mode: str) -> None:
    """直接调用 generate_ai_message，llm_mode 真正传入（mock 或 prompt_json）。"""
    from b4_local_agent_llm import generate_ai_message

    messages = [
        {"role": "system", "content": "You are a local tool-using agent."},
        {"role": "user", "content": "帮我阅读 docs/agent_intro.txt，总结三条中文要点。"},
    ]
    tools_schema = read_json(MESSAGES / "tools_schema_basic.json")
    ensure_dir(outdir)
    result = generate_ai_message(
        model_config, messages, tools_schema,
        mode=llm_mode,
        artifact_dir=str(outdir),
        artifact_stem="b4_single",
    )
    raw_path = outdir / "b4_single_raw_model_output.json"
    if raw_path.exists():
        raw = read_json(raw_path)
        print(f"  [LLM] mode={raw['mode']}, backend={raw['backend']}, status={raw['status']}")
        print(f"  [LLM] raw_text[:80]: {str(raw.get('raw_text', ''))[:80]}")
    assert result["status"] == "success", f"B4 status={result['status']}, error={result.get('error')}"
    ai_msg = result["ai_message"]
    assert ai_msg.get("tool_calls") or ai_msg.get("content", "").strip(), \
        "AIMessage 既无 tool_calls 也无 content"


# ============================================================
# B4 进阶1：单轮多 tool_calls
# ============================================================

def demo_b4_multi_tool(outdir: Path, model_config: str) -> None:
    from b4_multi_tool import demo_multi_tool_round

    tools_schema = read_json(MESSAGES / "tools_schema_basic.json")
    result = demo_multi_tool_round(model_config, tools_schema, str(outdir))
    assert result["status"] == "success"
    assert result["final_answer"].strip()
    msgs = result["messages"]
    # 应该有一个 AIMessage 带 2 个 tool_calls
    multi = [m for m in msgs if m.get("role") == "assistant" and len(m.get("tool_calls", [])) >= 2]
    assert multi, "expected an AIMessage with ≥2 tool_calls"


# ============================================================
# B4 进阶2：Plan-and-Execute
# ============================================================

def demo_b4_plan_execute(outdir: Path, model_config: str) -> None:
    from b4_plan_execute import run_plan_execute_demo

    result = run_plan_execute_demo(
        model_config,
        str(MESSAGES / "tools_schema_basic.json"),
        str(outdir),
        user_input="请读取 agent_intro.txt 并计算 1+1",
    )
    assert result["status"] == "success"
    assert result["final_answer"].strip()
    assert len(result["steps"]) >= 1


# ============================================================
# B4 进阶3：模型切换
# ============================================================

def demo_b4_model_switch(outdir: Path, model_config: str) -> None:
    from b4_model_switch import load_model_roster, generate_with_model_selection, select_model_for_task
    from b4_local_agent_llm import generate_ai_message
    import copy

    roster_path = _cfg("model_roster.yaml")
    roster = load_model_roster(roster_path)
    messages = read_json(MESSAGES / "messages_no_tool.json")
    tools_schema = read_json(MESSAGES / "tools_schema_basic.json")
    ensure_dir(outdir)
    # model_roster.yaml 中的 model_config 是相对 configs/ 的路径，需要解析为绝对路径
    roster_dir = Path(roster_path).resolve().parent
    for task_type in ("plan", "execute", "summarize"):
        rel_mc = select_model_for_task(task_type, roster)
        abs_mc = str((roster_dir / rel_mc).resolve())
        result = generate_ai_message(
            abs_mc, messages, tools_schema,
            mode="mock",
            artifact_dir=str(outdir),
            artifact_stem=f"switch_{task_type}",
        )
        assert result["status"] == "success", f"model_switch {task_type} failed"


# ============================================================
# B2 基础：5 个 skill
# ============================================================

def demo_b2_baseline(outdir: Path) -> None:
    import importlib, sys as _sys
    # skills/ 在 HAL1000/skills/ 下，需要把 HAL1000/ 加入 path 让 `import skills.xxx` 可用
    # 同时也把 HAL1000/skills/ 本身加入，让 `import calculator` 可用
    for extra in (str(ROOT), str(ROOT / "skills")):
        if extra not in _sys.path:
            _sys.path.insert(0, extra)

    data_root = str(DATA)

    cases = [
        ("calculator",      {"expression": "56*29+81"},                                        {}),
        ("file_reader",     {"path": "docs/agent_intro.txt", "max_chars": 500},               {"data_root": data_root}),
        ("local_file_search", {"query": "工具调用", "root_dir": str(DATA / "docs"), "file_types": [".txt", ".md"], "top_k": 3}, {}),
        ("table_analyzer",  {"path": str(DATA / "tables/results.csv"), "max_rows_preview": 5, "describe": True}, {}),
        ("format_converter", {"text": "Hello Agent", "target_format": "markdown"},            {}),
    ]
    results = {}
    for skill_name, args, extra in cases:
        mod = importlib.import_module(skill_name)
        fn = getattr(mod, skill_name)
        output = fn(**args, **extra)
        results[skill_name] = output
        assert output is not None

    ensure_dir(outdir)
    write_json(results, outdir / "b2_baseline_results.json")


# ============================================================
# B2 进阶：复合 skill + 沙箱
# ============================================================

def demo_b2_advanced(outdir: Path) -> None:
    from b2_advanced import run_advanced_skill as run_adv

    # 复合 skill — 传 data_root 让相对路径 docs/agent_intro.txt 能找到
    composite_input = read_json(DATA / "tool_inputs/advanced/composite_ok.json")
    res_composite = run_adv(
        "read_and_convert", composite_input,
        data_root=str(DATA),
        output_dir=str(outdir / "composite"),
    )
    assert res_composite["status"] == "success", f"composite: {res_composite}"

    # 沙箱
    sandbox_input = read_json(DATA / "tool_inputs/advanced/sandbox_ok.json")
    res_sandbox = run_adv(
        "safe_python_exec", sandbox_input,
        output_dir=str(outdir / "sandbox"),
    )
    assert res_sandbox["status"] == "success", f"sandbox: {res_sandbox}"


# ============================================================
# B3 基础：schema 生成 + tool_calls 执行
# ============================================================

def demo_b3_baseline(outdir: Path) -> None:
    from b3_tool_layer import get_tools_schema, execute_tool_calls

    tools_schema = get_tools_schema(_cfg("tools.yaml"), "basic_tools", str(outdir))
    assert len(tools_schema) >= 5, "expected at least 5 tools in schema"

    tool_calls_payload = read_json(MESSAGES / "ai_message_with_tool_calls.json")
    tool_calls = (
        tool_calls_payload.get("tool_calls")
        if isinstance(tool_calls_payload, dict)
        else tool_calls_payload
    )
    tool_messages = execute_tool_calls(
        tool_calls, _cfg("tools.yaml"), "basic_tools", str(outdir)
    )
    assert tool_messages, "expected at least one ToolMessage"


# ============================================================
# B3 进阶：auto_schema + retry + cache + stats
# ============================================================

def demo_b3_advanced(outdir: Path) -> None:
    from b3_advanced import auto_schema_from_module, execute_with_features
    from tool_cache import ToolCache
    from tool_stats import ToolStats

    # auto_schema
    schemas = auto_schema_from_module("skills.calculator", outdir=str(outdir / "auto_schema"))
    assert schemas, "auto_schema returned empty list"

    # retry + cache + stats
    tool_calls_payload = read_json(MESSAGES / "ai_message_with_tool_calls.json")
    tool_calls = (
        tool_calls_payload.get("tool_calls")
        if isinstance(tool_calls_payload, dict)
        else tool_calls_payload
    )
    cache = ToolCache()
    stats = ToolStats()

    # 第一次（cache miss）
    execute_with_features(
        tool_calls,
        tools_config=_cfg("tools.yaml"),
        toolset="basic_tools",
        cache=cache,
        stats=stats,
        retry_attempts=2,
        outdir=str(outdir / "first_run"),
    )
    # 第二次（cache hit）
    execute_with_features(
        tool_calls,
        tools_config=_cfg("tools.yaml"),
        toolset="basic_tools",
        cache=cache,
        stats=stats,
        retry_attempts=2,
        outdir=str(outdir / "second_run"),
    )
    snap = stats.snapshot()
    ensure_dir(outdir)
    write_json(snap, outdir / "tool_stats.json")
    assert snap.get("cache_hits", 0) >= 1, "expected at least 1 cache hit on second run"


# ============================================================
# 汇总报告
# ============================================================

def _write_report(outdir: Path, elapsed_total: float) -> None:
    ok_count  = sum(1 for r in _RESULTS if r["status"] == "ok")
    fail_count = sum(1 for r in _RESULTS if r["status"] == "fail")
    lines = [
        "# Advanced Demo — Full Report",
        "",
        f"- **时间**: {now_iso()}",
        f"- **总耗时**: {elapsed_total:.1f} ms",
        f"- **通过**: {ok_count} / {len(_RESULTS)}",
        f"- **失败**: {fail_count}",
        "",
        "## 逐项结果",
        "",
        "| # | 功能 | 状态 | 耗时(ms) | 备注 |",
        "|---|------|------|---------|------|",
    ]
    for i, r in enumerate(_RESULTS, 1):
        status = "✅ OK" if r["status"] == "ok" else "❌ FAIL"
        note = r.get("error", "")[:80]
        lines.append(f"| {i} | {r['label']} | {status} | {r['elapsed_ms']} | {note} |")
    lines += ["", "## 输出目录", "", f"`{outdir}`", ""]
    write_text("\n".join(lines), outdir / "full_advanced_report.md")
    write_json(_RESULTS, outdir / "full_advanced_results.json")


# ============================================================
# 主函数
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="一键运行 B1~B4 所有基础 + 进阶功能演示。"
    )
    p.add_argument(
        "--model_path",
        default="/root/siton-tmp/HAL1000/Qwen3.5-4B",
        help="本地模型路径，写入 HAL_MODEL_PATH 环境变量后传给 model.yaml",
    )
    p.add_argument(
        "--model_config",
        default=None,
        help="直接指定 model.yaml 路径（优先于 --model_path）",
    )
    p.add_argument(
        "--tools_config",  default=None,
        help="tools.yaml 路径，默认 configs/tools.yaml",
    )
    p.add_argument(
        "--memory_config", default=None,
        help="memory.yaml 路径，默认 configs/memory.yaml",
    )
    p.add_argument(
        "--outdir",
        default="../outputs/all_advanced_demo",
        help="输出根目录",
    )
    p.add_argument(
        "--llm_mode",
        choices=["mock", "prompt_json"],
        default="prompt_json",
        help="LLM 模式：mock=不需要GPU，prompt_json=调用本地模型（默认）",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="跳过指定模块，如 --skip b2_advanced b3_advanced",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 把 model_path 注入环境变量，让 model.yaml 中的 ${HAL_MODEL_PATH} 生效
    if args.model_path:
        os.environ["HAL_MODEL_PATH"] = args.model_path

    model_config = args.model_config or _cfg("model.yaml")
    outdir = resolve_cli_path(args.outdir)
    llm_mode = args.llm_mode
    skip = set(args.skip or [])

    print(f"\n{'='*60}")
    print(f"  HAL1000 All-Feature Advanced Demo")
    print(f"  模型路径   : {args.model_path}")
    print(f"  LLM 模式   : {llm_mode}")
    print(f"  输出目录   : {outdir}")
    print(f"  跳过模块   : {skip or '无'}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # B1
    # ------------------------------------------------------------------
    _section("B1 基础：fixture 模式完整 Agent 运行")
    _run("B1-基础 fixture 运行", demo_b1_fixture,
         outdir / "B1_fixture", model_config, llm_mode)

    _section("B1 进阶1：批量任务运行")
    _run("B1-进阶 批量运行", demo_b1_batch,
         outdir / "B1_batch", model_config, llm_mode)

    _section("B1 进阶2：历史消息压缩")
    _run("B1-进阶 消息压缩", demo_b1_compress,
         outdir / "B1_compress")

    _section("B1 进阶3：断点续跑（checkpoint + resume）")
    _run("B1-进阶 断点续跑", demo_b1_checkpoint,
         outdir / "B1_checkpoint", model_config, llm_mode)

    _section("B1 进阶4：System Prompt 模板切换（prompt_patches）")
    _run("B1-进阶 prompt_patches", demo_b1_prompt_patches,
         outdir / "B1_prompt_patches", model_config, llm_mode)

    # ------------------------------------------------------------------
    # B4
    # ------------------------------------------------------------------
    _section("B1 integrated：真实 LLM 调用（llm_mode 真正生效）")
    _run("B1-integrated 真实LLM调用", demo_b1_integrated,
         outdir / "B1_integrated", model_config, llm_mode)

    _section("B4 基础：真实 llm_mode 调用 generate_ai_message")
    _run("B4-基础 generate_ai_message", demo_b4_single_tool,
         outdir / "B4_single_tool", model_config, llm_mode)

    _section("B4 进阶1：单轮多 tool_calls")
    _run("B4-进阶 多工具调用", demo_b4_multi_tool,
         outdir / "B4_multi_tool", model_config)

    _section("B4 进阶2：Plan-and-Execute")
    _run("B4-进阶 Plan-and-Execute", demo_b4_plan_execute,
         outdir / "B4_plan_execute", model_config)

    _section("B4 进阶3：模型切换（model_switch）")
    _run("B4-进阶 模型切换", demo_b4_model_switch,
         outdir / "B4_model_switch", model_config)

    # ------------------------------------------------------------------
    # B2
    # ------------------------------------------------------------------
    _section("B2 基础：5 个 Skill 各运行一次")
    if "b2_baseline" not in skip:
        _run("B2-基础 5个skill", demo_b2_baseline,
             outdir / "B2_baseline")
    else:
        print("  [SKIP] b2_baseline")

    _section("B2 进阶：复合 Skill + 沙箱执行")
    if "b2_advanced" not in skip:
        _run("B2-进阶 复合skill+沙箱", demo_b2_advanced,
             outdir / "B2_advanced")
    else:
        print("  [SKIP] b2_advanced")

    # ------------------------------------------------------------------
    # B3
    # ------------------------------------------------------------------
    _section("B3 基础：生成 tools_schema + 执行 tool_calls")
    if "b3_baseline" not in skip:
        _run("B3-基础 schema+执行", demo_b3_baseline,
             outdir / "B3_baseline")
    else:
        print("  [SKIP] b3_baseline")

    _section("B3 进阶：auto_schema + retry + cache + stats")
    if "b3_advanced" not in skip:
        _run("B3-进阶 auto_schema+cache+stats", demo_b3_advanced,
             outdir / "B3_advanced")
    else:
        print("  [SKIP] b3_advanced")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    elapsed_total = round((time.perf_counter() - t_start) * 1000, 1)
    ok_count   = sum(1 for r in _RESULTS if r["status"] == "ok")
    fail_count = sum(1 for r in _RESULTS if r["status"] == "fail")

    _section(f"汇总  通过 {ok_count}/{len(_RESULTS)}  失败 {fail_count}")
    for r in _RESULTS:
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"  {icon}  {r['label']}  ({r['elapsed_ms']} ms)")

    print(f"\n  总耗时：{elapsed_total} ms")

    _write_report(outdir, elapsed_total)
    print(f"\n  完整报告：{outdir / 'full_advanced_report.md'}")
    print(f"  结果 JSON：{outdir / 'full_advanced_results.json'}\n")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
