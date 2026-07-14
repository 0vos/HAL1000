"""Composite Skill: read a file and convert its content in one call."""
from __future__ import annotations

from pathlib import Path

from skills import resolve_data_path


def read_and_convert(
    path: str,
    target_format: str = "markdown",
    max_chars: int = 2000,
    output_filename: str | None = None,
    output_dir: str | None = None,
    *,
    data_root: str | None = None,
) -> dict:
    """Read a local txt/md file and convert to markdown bullets or JSON.

    Returns a dict containing both the read result and the conversion result.
    Raises ValueError/FileNotFoundError on invalid inputs.
    """
    # Inline import to avoid circular import (skills package init is light)
    from skills.file_reader import file_reader
    from skills.format_converter import format_converter

    reader_result = file_reader(path=path, max_chars=max_chars, data_root=data_root)
    raw_text = reader_result["content"]
    converter_result = format_converter(
        text=raw_text,
        target_format=target_format,
        output_filename=output_filename,
        output_dir=output_dir,
    )
    return {
        "read": {
            "source": reader_result["source"],
            "num_chars": reader_result["num_chars"],
            "truncated": reader_result["truncated"],
        },
        "convert": {
            "target_format": target_format,
            "formatted_text": converter_result["formatted_text"],
            "generated_file_path": converter_result["generated_file_path"],
        },
        "pipeline": ["file_reader", "format_converter"],
    }