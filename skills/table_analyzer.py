from __future__ import annotations

import csv
import signal
import statistics
from contextlib import contextmanager

from skills import resolve_data_path
from skills_error_codes import ErrorCode, attach_error_code


@contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise TimeoutError(f"table_analyzer exceeded {seconds:.1f}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


_TIMEOUT_SECONDS = 5.0


def table_analyzer(
    path: str,
    max_rows_preview: int = 5,
    describe: bool = True,
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(max_rows_preview, int) or isinstance(max_rows_preview, bool) or max_rows_preview < 0:
        raise attach_error_code(ValueError("max_rows_preview must be a non-negative integer"), ErrorCode.INVALID_INPUT)
    source, root = resolve_data_path(path, data_root)
    if source.suffix.lower() not in {".csv", ".tsv"}:
        raise attach_error_code(ValueError("table_analyzer only supports .csv and .tsv files"), ErrorCode.UNSUPPORTED_TYPE)
    if not source.is_file():
        raise attach_error_code(FileNotFoundError(f"table file not found: {path}"), ErrorCode.FILE_NOT_FOUND)
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    try:
        with _time_limit(_TIMEOUT_SECONDS):
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if not reader.fieldnames:
                    raise attach_error_code(ValueError("table must contain a header row"), ErrorCode.INVALID_INPUT)
                rows = list(reader)
                columns = list(reader.fieldnames)
    except TimeoutError as exc:
        raise attach_error_code(exc, ErrorCode.EXECUTION_TIMEOUT) from exc
    stats: dict[str, dict] = {}
    if describe:
        for column in columns:
            raw_values = [row.get(column, "").strip() for row in rows]
            if not raw_values or any(value == "" for value in raw_values):
                continue
            try:
                values = [float(value) for value in raw_values]
            except ValueError:
                continue
            stats[column] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }
    return {
        "path": source.relative_to(root).as_posix(),
        "num_rows": len(rows),
        "num_columns": len(columns),
        "columns": columns,
        "preview": rows[:max_rows_preview],
        "describe": stats,
    }
