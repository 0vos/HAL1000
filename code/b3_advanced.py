"""B3 advanced features.

Wraps the base b3_tool_layer with:
    - auto_schema_from_func(module, name): generate schema from a Python function
    - retry_transient_errors(): wrap execute with retry on recoverable failures
    - cache_results(): in-memory + disk cache for tool call results
    - stats_tracker(): per-tool call/failure/latency statistics
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path, resolve_from_file
from common.schemas import make_skill_result, make_tool_message, normalize_tool_call

from auto_schema import schema_from_function, schemas_from_module
from retry import call_with_retry, should_retry, should_retry_result
from tool_cache import ToolCache, get_global_cache, reset_global_cache
from tool_stats import ToolStats


bootstrap_project_root()


# ---------------------------------------------------------------------------
# Auto-schema
# ---------------------------------------------------------------------------

def auto_schema_from_module(module_name: str, names: list[str] | None = None, outdir: str | None = None) -> list[dict]:
    module = importlib.import_module(module_name)
    schemas = schemas_from_module(module, names)
    if outdir:
        out = Path(outdir)
        write_json(schemas, out / "auto_tools_schema.json")
        write_json(
            {
                "status": "success",
                "module": module_name,
                "names": names or [s["function"]["name"] for s in schemas],
                "tool_count": len(schemas),
            },
            out / "auto_schema_report.json",
        )
    return schemas


def auto_schema_from_function(module_name: str, function_name: str, outdir: str | None = None) -> dict:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    schema = schema_from_function(function, name=function_name)
    if outdir:
        out = Path(outdir)
        write_json([schema], out / "auto_tools_schema.json")
        write_json(
            {"status": "success", "module": module_name, "name": function_name},
            out / "auto_schema_report.json",
        )
    return schema


# ---------------------------------------------------------------------------
# Retry / cache / stats decorators for execute_tool_calls
# ---------------------------------------------------------------------------

def execute_with_features(
    tool_calls: list[dict],
    *,
    tools_config: str | Path,
    toolset: str,
    cache: ToolCache | None = None,
    stats: ToolStats | None = None,
    retry_attempts: int = 1,
    outdir: str | Path | None = None,
) -> list[dict]:
    """Execute tool_calls with optional retry / cache / stats tracking.

    cache and stats are passed in by the caller (so they can be shared
    across multiple invocations).
    """
    from b3_tool_layer import _load_tools_config, _resolve_toolset, _error_result, execute_tool_calls

    config_path, config = _load_tools_config(tools_config)
    selected, allowed_tools = _resolve_toolset(config, toolset)
    data_root_setting = config.get("settings", {}).get("data_root", "../data")
    resolved_data_root = resolve_from_file(data_root_setting, config_path)
    output_dir = Path(outdir) if outdir else None
    messages = []
    log_records = []
    for index, raw_call in enumerate(tool_calls):
        try:
            call = normalize_tool_call(raw_call, index)
        except Exception as exc:
            call = {"id": f"call_{index + 1:03d}", "name": "unknown", "args": {}}
            result = _error_result(call["name"], call["args"], exc)
            messages.append(make_tool_message(call["id"], call["name"], json.dumps(result, ensure_ascii=False), result["status"]))
            log_records.append({"tool_call_id": call["id"], "name": call["name"], "status": "error", "args": call["args"], "skill_result": result})
            if stats:
                stats.record(call["name"], result["status"], result["latency_ms"], (result.get("error") or {}).get("code"))
            continue
        name = call["name"]
        args = call["args"]
        if name not in allowed_tools or name not in config["tools"]:
            result = _error_result(name, args, ValueError(f"tool is not available in {selected}: {name}"))
            content = json.dumps(result, ensure_ascii=False)
            messages.append(make_tool_message(call["id"], name, content, result["status"]))
            log_records.append({"tool_call_id": call["id"], "name": name, "status": "error", "args": args, "skill_result": result})
            if stats:
                stats.record(name, result["status"], result["latency_ms"], (result.get("error") or {}).get("code"))
            continue

        # cache lookup
        cached = None
        if cache is not None:
            cached = cache.get(name, args)
            if cached is not None:
                if stats:
                    stats.record_cache_hit()
                result = cached["result"]
                content = json.dumps(result, ensure_ascii=False)
                messages.append(make_tool_message(call["id"], name, content, result["status"]))
                log_records.append({
                    "tool_call_id": call["id"],
                    "name": name,
                    "status": result["status"],
                    "args": args,
                    "skill_result": result,
                    "cache": "hit",
                })
                if stats:
                    stats.record(name, result["status"], result.get("latency_ms", 0.0), (result.get("error") or {}).get("code"))
                continue
            if stats:
                stats.record_cache_miss()

        # invoke with optional retry
        definition = config["tools"][name]
        from b3_tool_layer import _validate_args
        try:
            _validate_args(args, definition)
        except Exception as exc:
            result = _error_result(name, args, exc)
            content = json.dumps(result, ensure_ascii=False)
            messages.append(make_tool_message(call["id"], name, content, result["status"]))
            log_records.append({"tool_call_id": call["id"], "name": name, "status": "error", "args": args, "skill_result": result})
            if stats:
                stats.record(name, result["status"], result["latency_ms"], (result.get("error") or {}).get("code"))
            continue
        module = importlib.import_module(definition["module"])
        function = getattr(module, definition["function"])
        kwargs = dict(args)
        signature = inspect.signature(function)
        if "data_root" in signature.parameters:
            kwargs["data_root"] = str(resolved_data_root)
        if "output_dir" in signature.parameters:
            kwargs["output_dir"] = str(output_dir) if output_dir else None
        start = time.perf_counter()
        try:
            if retry_attempts > 1:
                if stats:
                    for _ in range(retry_attempts - 1):
                        stats.record_retry()
                result, attempt_log = call_with_retry(lambda: _invoke_to_skill_result(name, function, kwargs), max_attempts=retry_attempts)
                log_records.append({
                    "tool_call_id": call["id"],
                    "name": name,
                    "status": result.get("status"),
                    "args": args,
                    "skill_result": result,
                    "retry_attempts": attempt_log,
                })
            else:
                result = _invoke_to_skill_result(name, function, kwargs)
                log_records.append({
                    "tool_call_id": call["id"],
                    "name": name,
                    "status": result["status"],
                    "args": args,
                    "skill_result": result,
                })
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            result = _error_result(name, args, exc, elapsed_ms)
            log_records.append({"tool_call_id": call["id"], "name": name, "status": "error", "args": args, "skill_result": result})

        if cache is not None:
            cache.put(name, args, result)
        if stats:
            stats.record(name, result["status"], result["latency_ms"], (result.get("error") or {}).get("code"))

        content = json.dumps(result, ensure_ascii=False)
        messages.append(make_tool_message(call["id"], name, content, result["status"]))

    if output_dir:
        write_json(messages, output_dir / "tool_messages.json")
        for record in log_records:
            append_jsonl(record, output_dir / "tool_call_log.jsonl")
        if stats is not None:
            write_json(stats.snapshot(), output_dir / "tool_stats.json")
    return messages


def _invoke_to_skill_result(name: str, function, kwargs: dict) -> dict:
    from time import perf_counter
    start = perf_counter()
    try:
        output = function(**kwargs)
        latency_ms = round((perf_counter() - start) * 1000, 3)
        return make_skill_result(name, "success", kwargs, output, None, latency_ms)
    except Exception as exc:
        from skills_error_codes import enrich_error_payload
        latency_ms = round((perf_counter() - start) * 1000, 3)
        return make_skill_result(name, "error", kwargs, None, enrich_error_payload(exc), latency_ms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B3 advanced features (auto_schema, retry, cache, stats).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    schema_p = sub.add_parser("auto_schema", help="Generate tools schema from a Python module or function.")
    schema_p.add_argument("--module", required=True)
    schema_p.add_argument("--function", default=None, help="If omitted, generates for every public callable.")
    schema_p.add_argument("--outdir", required=True)

    run_p = sub.add_parser("execute", help="Execute tool calls with retry/cache/stats.")
    run_p.add_argument("--tools_config", required=True)
    run_p.add_argument("--toolset", required=True)
    run_p.add_argument("--tool_calls", required=True)
    run_p.add_argument("--outdir", required=True)
    run_p.add_argument("--retry", type=int, default=1)
    run_p.add_argument("--cache", action="store_true")
    run_p.add_argument("--cache_path", default=None)
    run_p.add_argument("--stats", action="store_true")
    run_p.add_argument("--stats_path", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "auto_schema":
            outdir = resolve_cli_path(args.outdir)
            if args.function:
                auto_schema_from_function(args.module, args.function, str(outdir))
            else:
                auto_schema_from_module(args.module, None, str(outdir))
            print(outdir / "auto_tools_schema.json")
            return 0
        if args.cmd == "execute":
            outdir = resolve_cli_path(args.outdir)
            config_path = resolve_cli_path(args.tools_config)
            calls_path = resolve_cli_path(args.tool_calls)
            payload = read_json(calls_path)
            tool_calls = payload.get("tool_calls") if isinstance(payload, dict) else payload
            cache = ToolCache(persist_path=args.cache_path) if args.cache else None
            stats = ToolStats(persist_path=args.stats_path) if args.stats else None
            messages = execute_with_features(
                tool_calls,
                tools_config=str(config_path),
                toolset=args.toolset,
                cache=cache,
                stats=stats,
                retry_attempts=max(1, int(args.retry)),
                outdir=str(outdir),
            )
            print(outdir / "tool_messages.json")
            return 0
        return 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())