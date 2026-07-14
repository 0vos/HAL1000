"""Advanced Skill execution entry point.

Extends the base b2_run_skill.py with:
    - error-code classification (ErrorCode enum)
    - composite Skill `read_and_convert`
    - sandboxed Skill `safe_python_exec`
    - timeout enforcement for high-risk Skills
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path
from common.schemas import make_skill_result
from skills_error_codes import ErrorCode, attach_error_code, classify_exception, enrich_error_payload


bootstrap_project_root()


ADVANCED_SKILLS = {
    "read_and_convert": {
        "module": "composite_skill",
        "function": "read_and_convert",
        "description": "Read a local file and convert it to markdown bullets or JSON in one call.",
        "risk_level": "low",
    },
    "safe_python_exec": {
        "module": "safe_python_exec",
        "function": "safe_python_exec",
        "description": "Evaluate a small Python snippet in a sandboxed environment (no imports, timeout enforced).",
        "risk_level": "high",
        "timeout_seconds": 5.0,
    },
}

HIGH_RISK_SKILLS = {name for name, meta in ADVANCED_SKILLS.items() if meta.get("risk_level") == "high"}


def _run_sandbox(function, kwargs: dict, timeout_seconds: float) -> dict:
    """Normalize the cross-platform restricted executor result."""
    start = perf_counter()
    try:
        outcome = function(**kwargs)
    except Exception as exc:
        error = enrich_error_payload(exc)
        return {"status": "error", "error": error, "latency_ms": round((perf_counter() - start) * 1000, 3)}
    latency_ms = round((perf_counter() - start) * 1000, 3)
    if not isinstance(outcome, dict):
        return {"status": "success", "output": outcome, "latency_ms": latency_ms}
    internal_status = outcome.get("status")
    if internal_status in {"timeout", "error"}:
        error_payload = outcome.get("error")
        if isinstance(error_payload, dict):
            err = dict(error_payload)
        else:
            err = {
                "type": "SandboxError",
                "message": str(error_payload),
                "code": "INTERNAL",
            }
        if internal_status == "timeout":
            err.setdefault("type", "SandboxTimeout")
            err.setdefault("code", ErrorCode.EXECUTION_TIMEOUT.value)
        if isinstance(outcome.get("security"), dict):
            err["security"] = outcome["security"]
        return {"status": "error", "error": err, "latency_ms": latency_ms}
    return {"status": "success", "output": outcome, "latency_ms": latency_ms}


def _import_skill_module(name: str):
    if name not in ADVANCED_SKILLS:
        raise ValueError(f"unknown advanced skill: {name}")
    return importlib.import_module(ADVANCED_SKILLS[name]["module"])


def _run_with_timeout(function, kwargs: dict, timeout_seconds: float) -> dict:
    """Run a function in a separate thread so we can enforce a timeout.

    Note: threads cannot be killed in Python, so this works best for
    CPU-bound loops where the worker checks time. For I/O bound work
    this only marks the call as "exceeded" but does not stop it.
    """
    result: dict = {}

    def target():
        try:
            result["output"] = function(**kwargs)
            result["status"] = "success"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return {
            "status": "error",
            "error": attach_error_code(
                TimeoutError(f"execution exceeded {timeout_seconds:.1f}s"),
                ErrorCode.EXECUTION_TIMEOUT,
            ),
        }
    if result.get("status") == "error":
        return {"status": "error", "error": result["error"]}
    return {"status": "success", "output": result["output"]}


def run_advanced_skill(skill_name: str, input_data: dict, data_root: str | None = None, output_dir: str | None = None) -> dict:
    if skill_name not in ADVANCED_SKILLS:
        raise ValueError(f"unknown advanced skill: {skill_name}")
    if not isinstance(input_data, dict):
        raise ValueError("skill input must be a JSON object")
    meta = ADVANCED_SKILLS[skill_name]
    module = importlib.import_module(meta["module"])
    function = getattr(module, meta["function"])
    kwargs = dict(input_data)
    signature = inspect.signature(function)
    if "data_root" in signature.parameters:
        kwargs["data_root"] = data_root or str(Path(__file__).resolve().parent.parent / "data")
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = output_dir
    start = perf_counter()
    try:
        if skill_name == "safe_python_exec":
            timeout = float(meta.get("timeout_seconds", 5.0))
            outcome = _run_sandbox(function, kwargs, timeout)
            if outcome.get("status") == "error":
                error = outcome["error"]
                output = None
            else:
                output = outcome["output"]
                error = None
        else:
            output = function(**kwargs)
            error = None
    except Exception as exc:
        output = None
        error = enrich_error_payload(attach_error_code(exc, classify_exception(exc)))
    latency_ms = round((perf_counter() - start) * 1000, 3)
    status = "error" if error else "success"
    return make_skill_result(skill_name, status, input_data, output, error, latency_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one advanced local Agent skill (composite or sandboxed).")
    parser.add_argument("--skill", required=True, choices=sorted(ADVANCED_SKILLS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--data_root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = resolve_cli_path(args.input)
        outdir = resolve_cli_path(args.outdir)
        input_data = read_json(input_path)
        data_root = str(resolve_cli_path(args.data_root)) if args.data_root else None
        outdir.mkdir(parents=True, exist_ok=True)
        result = run_advanced_skill(args.skill, input_data, data_root, str(outdir))
        result_path = outdir / f"{args.skill}_result.json"
        write_json(result, result_path)
        append_jsonl(
            {
                "timestamp": now_iso(),
                "skill_name": args.skill,
                "status": result["status"],
                "error_code": (result.get("error") or {}).get("code"),
                "result_path": str(result_path),
                "latency_ms": result["latency_ms"],
            },
            outdir / "skill_run_log.jsonl",
        )
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
