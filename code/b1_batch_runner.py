"""
b1_batch_runner.py — Batch task runner for b1_agent_runtime.

Reads a JSON array of runtime_input objects and runs each one sequentially,
collecting results into a batch_summary.json.

Usage:
    python b1_batch_runner.py \\
        --batch_input ../data/batch_input.json \\
        --tools_config ../configs/tools.yaml \\
        --memory_config ../configs/memory.yaml \\
        --model_config ../configs/model.yaml \\
        --outdir ../outputs/B1_batch_test \\
        [--llm_mode mock|prompt_json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from b1_agent_runtime import run_agent
from common.io_utils import write_json
from common.path_utils import resolve_cli_path


def _resolve_path_field(value: str, base_dir: Path) -> str:
    """Resolve a relative path string against *base_dir* to an absolute path."""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    resolved = (base_dir / p).resolve()
    return str(resolved)


def _resolve_task_paths(task: dict, base_dir: Path) -> dict:
    """Return a copy of *task* with all path fields resolved to absolute paths.

    This ensures that the temporary runtime_input file can be placed anywhere
    on disk without breaking relative path references.
    """
    import copy
    resolved = copy.deepcopy(task)

    # Top-level path field
    for key in ("system_prompt_path",):
        if key in resolved and isinstance(resolved[key], str):
            resolved[key] = _resolve_path_field(resolved[key], base_dir)

    # Fixture sub-paths
    fixtures = resolved.get("fixtures")
    if isinstance(fixtures, dict):
        for key in (
            "selected_memory_path",
            "tools_schema_path",
            "ai_messages_path",
            "tool_messages_path",
        ):
            if key in fixtures and isinstance(fixtures[key], str):
                fixtures[key] = _resolve_path_field(fixtures[key], base_dir)

    # prompt_patches switch_to paths
    for patch in resolved.get("prompt_patches", []):
        if "switch_to" in patch and isinstance(patch["switch_to"], str):
            patch["switch_to"] = _resolve_path_field(patch["switch_to"], base_dir)

    return resolved


def run_batch(
    batch_input_path: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    base_outdir: str,
    llm_mode: str | None = None,
) -> dict:
    """Run all tasks in *batch_input_path* and write a batch_summary.json.

    Parameters
    ----------
    batch_input_path:
        Path to a JSON file containing an array of runtime_input objects.
    tools_config / memory_config / model_config:
        Config paths forwarded to run_agent (required for integrated mode;
        may be None when tasks use fixture mode).
    base_outdir:
        Parent output directory.  Each task writes to
        ``{base_outdir}/{conversation_id}/``.
    llm_mode:
        Optional LLM mode override ("mock" | "prompt_json").

    Returns
    -------
    dict
        The batch summary dict (also written to disk as batch_summary.json).
    """
    batch_path = Path(batch_input_path).resolve()
    # The batch file's directory is the base for resolving relative paths
    batch_dir = batch_path.parent
    base_out = Path(base_outdir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    with open(batch_path, "r", encoding="utf-8") as fh:
        tasks: list[dict] = json.load(fh)

    if not isinstance(tasks, list):
        raise ValueError("batch_input must be a JSON array of runtime_input objects")

    task_results = []
    success_count = 0
    failed_count = 0

    for task in tasks:
        conv_id = task.get("conversation_id", f"task_{len(task_results)}")
        task_outdir = base_out / conv_id
        task_outdir.mkdir(parents=True, exist_ok=True)

        # Resolve relative paths in the task dict so that run_agent can find
        # them regardless of where the temp input file is written.
        resolved_task = _resolve_task_paths(task, batch_dir)

        # Write the resolved runtime_input to a temporary file in the task
        # output directory so that run_agent can read it.
        temp_input_path = task_outdir / "runtime_input_temp.json"
        with open(temp_input_path, "w", encoding="utf-8") as fh:
            json.dump(resolved_task, fh, ensure_ascii=False, indent=2)

        t0 = perf_counter()
        try:
            result = run_agent(

                str(temp_input_path),
                tools_config,
                memory_config,
                model_config,
                str(task_outdir),
                llm_mode,
            )
            elapsed_ms = round((perf_counter() - t0) * 1000, 3)
            task_results.append(
                {
                    "conversation_id": conv_id,
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "outdir": str(task_outdir),
                    "agent_status": result.get("status"),
                }
            )
            success_count += 1
            print(f"[batch] {conv_id}: success ({elapsed_ms} ms)")
        except Exception as exc:
            elapsed_ms = round((perf_counter() - t0) * 1000, 3)
            task_results.append(
                {
                    "conversation_id": conv_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": elapsed_ms,
                    "outdir": str(task_outdir),
                }
            )
            failed_count += 1
            print(f"[batch] {conv_id}: ERROR — {type(exc).__name__}: {exc}")

    summary = {
        "total": len(tasks),
        "success": success_count,
        "failed": failed_count,
        "tasks": task_results,
    }
    write_json(summary, base_out / "batch_summary.json")
    print(
        f"[batch] Finished. total={summary['total']}, "
        f"success={summary['success']}, failed={summary['failed']}"
    )
    print(f"[batch] Summary written to {base_out / 'batch_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch of agent tasks.")
    parser.add_argument("--batch_input", required=True, help="Path to batch JSON file")
    parser.add_argument("--tools_config", default=None)
    parser.add_argument("--memory_config", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--outdir", required=True, help="Base output directory")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_batch(
            str(resolve_cli_path(args.batch_input)),
            str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
            str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
            str(resolve_cli_path(args.model_config)) if args.model_config else None,
            str(resolve_cli_path(args.outdir)),
            args.llm_mode,
        )
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
