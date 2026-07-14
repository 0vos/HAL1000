"""
b1_compress.py — Deterministic history-message compression for long conversations.

When the non-system message count exceeds a threshold, older messages are
summarised into a concise text block and replaced by a single summary
HumanMessage, keeping the most recent messages intact.

This module is intentionally **LLM-free**: compression is purely rule-based
so that it is fast, cheap, and reliable.

CLI usage:
    python b1_compress.py \\
        --messages path/to/messages.json \\
        --outdir   path/to/output_dir  \\
        [--compress_after 8] \\
        [--keep_recent 4]

Outputs:
    {outdir}/compressed_messages.json
    {outdir}/compression_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io_utils import write_json
from common.path_utils import resolve_cli_path


# ---------------------------------------------------------------------------
# Core compression logic
# ---------------------------------------------------------------------------

def _format_message_summary_line(msg: dict[str, Any]) -> str | None:
    """Return a single human-readable summary line for *msg*, or None to skip."""
    role = msg.get("role", "")
    content = msg.get("content") or ""

    if role == "system":
        return None  # system messages are always preserved separately

    if role == "user":
        snippet = str(content)[:200]
        return f"[用户] {snippet}"

    if role == "assistant":
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            first = tool_calls[0]
            name = first.get("name", "unknown")
            args = first.get("args") or first.get("arguments") or {}
            if isinstance(args, str):
                # Some serialisations keep args as a JSON string
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if isinstance(args, dict):
                args_summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3])
            else:
                args_summary = str(args)[:80]
            return f"[助手→工具] {name}({args_summary})"
        snippet = str(content)[:200]
        return f"[助手] {snippet}"

    if role == "tool":
        name = msg.get("name", "unknown")
        status = msg.get("status", "")
        output = msg.get("output") or {}
        # 无论 status 是否已知，都尝试从 content 解析来补充 output
        if not output or not status:
            try:
                parsed = json.loads(msg.get("content", "{}"))
                if not status:
                    status = parsed.get("status", "unknown")
                if not output:
                    output = parsed.get("output") or {}
            except Exception:
                if not status:
                    status = "unknown"
        # 保留关键信息：file_writer 保留路径，file_reader 保留读了哪个文件
        detail = ""
        if name == "file_writer":
            path = output.get("written_path") or output.get("path", "")
            if path:
                import os as _os
                detail = f" → {_os.path.basename(path)}"
        elif name == "file_reader":
            source = output.get("source") or output.get("path", "")
            if source:
                detail = f" ← {source}"
        return f"[工具结果:{name}] {status}{detail}"

    # Unknown role — skip
    return None


def _build_summary_text(messages_to_compress: list[dict]) -> str:
    """Build a human-readable summary string from the messages to be compressed."""
    lines = []
    for msg in messages_to_compress:
        line = _format_message_summary_line(msg)
        if line is not None:
            lines.append(line)
    return "\n".join(lines)


def maybe_compress_messages(
    messages: list[dict],
    compress_after: int = 8,
    keep_recent: int = 4,
) -> tuple[list[dict], bool]:
    """Compress old messages into a summary when the history grows too long.

    Parameters
    ----------
    messages:
        Full message list (the first element is expected to be the system message).
    compress_after:
        Maximum number of non-system messages before compression kicks in.
    keep_recent:
        Number of recent non-system messages to preserve verbatim.

    Returns
    -------
    (new_messages, was_compressed)
    """
    if not messages:
        return messages, False

    # Split system vs non-system
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= compress_after:
        return messages, False

    # Determine which messages to compress and which to keep
    to_compress = non_system[:-keep_recent] if keep_recent > 0 else non_system
    recent_messages = non_system[-keep_recent:] if keep_recent > 0 else []

    summary_text = _build_summary_text(to_compress)
    summary_msg = {
        "role": "user",
        "content": f"[历史摘要]\n{summary_text}",
    }

    # Reconstruct: system (first one), summary, recent messages
    original_system_msg = system_msgs[0] if system_msgs else {"role": "system", "content": ""}
    new_messages = [original_system_msg, summary_msg] + recent_messages
    return new_messages, True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compress a messages.json file by summarising old messages."
    )
    parser.add_argument("--messages", required=True, help="Path to messages.json")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--compress_after",
        type=int,
        default=8,
        help="Compress when non-system message count exceeds this (default: 8)",
    )
    parser.add_argument(
        "--keep_recent",
        type=int,
        default=4,
        help="Number of recent non-system messages to keep verbatim (default: 4)",
    )
    args = parser.parse_args(argv)

    messages_path = Path(resolve_cli_path(args.messages)).resolve()
    outdir = Path(resolve_cli_path(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(messages_path, "r", encoding="utf-8") as fh:
        messages: list[dict] = json.load(fh)

    if not isinstance(messages, list):
        print("fatal: messages file must contain a JSON array", file=sys.stderr)
        return 1

    original_count = len(messages)
    non_system_before = sum(1 for m in messages if m.get("role") != "system")

    new_messages, was_compressed = maybe_compress_messages(
        messages,
        compress_after=args.compress_after,
        keep_recent=args.keep_recent,
    )

    non_system_after = sum(1 for m in new_messages if m.get("role") != "system")

    # Build summary text for the report (re-derive it)
    non_system = [m for m in messages if m.get("role") != "system"]
    if was_compressed:
        keep = args.keep_recent if args.keep_recent > 0 else 0
        to_compress_msgs = non_system[:-keep] if keep > 0 else non_system
        summary_text = _build_summary_text(to_compress_msgs)
    else:
        summary_text = ""

    report = {
        "was_compressed": was_compressed,
        "original_message_count": original_count,
        "compressed_message_count": len(new_messages),
        "non_system_before": non_system_before,
        "non_system_after": non_system_after,
        "compress_after": args.compress_after,
        "keep_recent": args.keep_recent,
        "summary_text": summary_text,
    }

    write_json(new_messages, outdir / "compressed_messages.json")
    write_json(report, outdir / "compression_report.json")

    print(f"was_compressed: {was_compressed}")
    print(f"original messages: {original_count}, compressed: {len(new_messages)}")
    print(f"Output written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
