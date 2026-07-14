"""
b1_resume.py — CLI for resuming an agent run from a checkpoint.

Usage:
    python b1_resume.py --outdir ../outputs/B1_checkpoint_test \\
        --tools_config ../configs/tools.yaml \\
        --memory_config ../configs/memory.yaml \\
        --model_config ../configs/model.yaml \\
        [--llm_mode mock|prompt_json]
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from b1_checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from common.io_utils import append_jsonl, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path


def resume_agent(
    outdir: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    llm_mode: str | None = None,
) -> dict:
    """Resume an agent run from a saved checkpoint in *outdir*.

    The function mirrors the main loop of b1_agent_runtime.run_agent() but
    initialises all loop counters from the checkpoint state rather than from
    scratch.
    """
    output_dir = Path(outdir).resolve()

    # ------------------------------------------------------------------ #
    # 1. Load checkpoint
    # ------------------------------------------------------------------ #
    ckpt = load_checkpoint(output_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {output_dir}")

    print(f"[resume] Resuming from turn {ckpt['resume_from_turn']} "
          f"(conversation_id={ckpt['conversation_id']})")

    execution_mode = ckpt["execution_mode"]
    runtime = ckpt["runtime_input"]
    messages = ckpt["messages"]
    tool_rounds: int = ckpt["tool_rounds"]
    llm_calls: int = ckpt["llm_calls"]
    turns: list = ckpt["turns"]

    # ------------------------------------------------------------------ #
    # 2. Reconstruct helpers (mirrors b1_agent_runtime.run_agent)
    # ------------------------------------------------------------------ #
    # We need the input_file only when fixture mode needs to resolve paths.
    # Runtime_input already contains resolved fixture data embedded in ckpt.
    fixture_data: dict | None = None
    tools_file = memory_file = model_file = None
    mode = "fixture"

    if execution_mode == "fixture":
        # Re-load fixture inputs from the paths stored in runtime_input.
        # We derive a pseudo input_file from the outdir itself since the
        # real input file may no longer be the cwd.
        from b1_agent_runtime import _load_fixture_inputs

        # Build a temporary input file reference so resolve_from_file works.
        # The simplest approach: write runtime_input to a temp file.
        import json, tempfile, os
        tmp_input = output_dir / "_resume_runtime_input_temp.json"
        with open(tmp_input, "w", encoding="utf-8") as fh:
            json.dump(runtime, fh, ensure_ascii=False, indent=2)
        try:
            fixture_data = _load_fixture_inputs(tmp_input, runtime)
        finally:
            if tmp_input.exists():
                tmp_input.unlink()
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError(
                "integrated mode requires --tools_config, --memory_config, and --model_config"
            )
        from b3_tool_layer import execute_tool_calls
        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        from b1_agent_runtime import _default_llm_mode
        mode = llm_mode or _default_llm_mode(model_file)

    from b1_agent_runtime import generate_ai_message, _fixture_tool_messages

    # ------------------------------------------------------------------ #
    # 3. Continue the main loop from resume_from_turn
    # ------------------------------------------------------------------ #
    started = perf_counter()
    all_tool_messages: list[dict] = []
    final_answer = ""
    status = "success"
    terminal_error = None
    warnings: list = []

    while True:
        llm_calls += 1
        turn_start = perf_counter()

        if execution_mode == "fixture":
            if llm_calls > len(fixture_data["ai_messages"]):
                raise ValueError(
                    "fixture AIMessage sequence ended before a final answer"
                )
            ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
            llm_status = "success"
            llm_error = None
        else:
            llm_result = generate_ai_message(
                str(model_file),
                messages,
                fixture_data["tools_schema"] if fixture_data else [],
                mode,
                str(output_dir / "llm_calls"),
                f"llm_call_{llm_calls:03d}",
            )
            if not isinstance(llm_result, dict) or not isinstance(
                llm_result.get("ai_message"), dict
            ):
                raise ValueError("B4 result must contain an ai_message object")
            ai_message = llm_result["ai_message"]
            llm_status = llm_result.get("status")
            llm_error = llm_result.get("error")

        messages.append(ai_message)
        turn = {
            "turn_index": llm_calls,
            "ai_message": ai_message,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "latency_ms": None,
        }

        if llm_status != "success":
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_calls,
                "cause": llm_error,
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        tool_calls = ai_message.get("tool_calls", [])
        if not tool_calls:
            final_answer = ai_message["content"]
            print(f"content: {final_answer}")
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        if tool_rounds >= runtime["max_turns"]:
            requested = ", ".join(call.get("name", "unknown") for call in tool_calls)
            final_answer = (
                "任务因超过最大工具调用轮次而终止，"
                f"最后一次模型仍请求调用工具：{requested}。"
            )
            status = "max_turns_exceeded"
            terminal_error = {
                "type": "MaxTurnsExceeded",
                "message": final_answer,
                "unexecuted_tool_calls": tool_calls,
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        if execution_mode == "fixture":
            tool_messages = _fixture_tool_messages(
                tool_calls, fixture_data["tool_messages"]
            )
        else:
            tool_messages = execute_tool_calls(
                tool_calls,
                str(tools_file),
                runtime["toolset"],
                str(output_dir),
            )

        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        turn["tool_messages"] = tool_messages
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)

        # Save checkpoint after each turn
        save_checkpoint(
            output_dir,
            {
                "conversation_id": runtime["conversation_id"],
                "execution_mode": execution_mode,
                "resume_from_turn": tool_rounds,
                "messages": messages,
                "tool_rounds": tool_rounds,
                "llm_calls": llm_calls,
                "turns": turns,
                "status": "running",
                "runtime_input": runtime,
            },
        )

    # ------------------------------------------------------------------ #
    # 4. Write outputs and clean up checkpoint
    # ------------------------------------------------------------------ #
    write_json(messages, output_dir / "messages.json")
    if execution_mode == "integrated":
        write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")

    trace = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": status,
        "toolset": runtime["toolset"],
        "max_turns": runtime["max_turns"],
        "tool_rounds_used": tool_rounds,
        "llm_call_count": llm_calls,
        "turns": turns,
        "final_answer_path": "final_answer.md",
        "memory_save": {"requested": runtime["save_memory"], "status": "not_requested"},
        "warnings": warnings,
        "error": terminal_error,
        "resumed": True,
    }
    write_json(trace, output_dir / "trace.json")

    clear_checkpoint(output_dir)

    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    print(f"[resume] Done. status={status}, elapsed_ms={elapsed_ms}")
    return {
        "conversation_id": runtime["conversation_id"],
        "status": status,
        "final_answer": final_answer,
        "elapsed_ms": elapsed_ms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume an agent run from a saved checkpoint."
    )
    parser.add_argument("--outdir", required=True, help="Directory containing checkpoint.json")
    parser.add_argument("--tools_config", default=None)
    parser.add_argument("--memory_config", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = resume_agent(
            str(resolve_cli_path(args.outdir)),
            str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
            str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
            str(resolve_cli_path(args.model_config)) if args.model_config else None,
            args.llm_mode,
        )
        print(result)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
