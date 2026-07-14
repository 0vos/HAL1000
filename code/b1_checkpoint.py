"""
b1_checkpoint.py — Checkpoint save/load/clear utilities for b1_agent_runtime.

Provides atomic checkpoint persistence so that an interrupted agent run
can be resumed from the last completed turn via b1_resume.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CHECKPOINT_FILENAME = "checkpoint.json"
CHECKPOINT_TMP_FILENAME = ".checkpoint.json.tmp"


def save_checkpoint(outdir: str | Path, state: dict[str, Any]) -> None:
    """Atomically write *state* to {outdir}/checkpoint.json.

    Uses a write-then-rename pattern so that a crash mid-write never
    leaves a partially-written checkpoint file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_path = outdir / CHECKPOINT_TMP_FILENAME
    final_path = outdir / CHECKPOINT_FILENAME
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def load_checkpoint(outdir: str | Path) -> dict[str, Any] | None:
    """Return the checkpoint dict stored in *outdir*, or *None* if absent."""
    path = Path(outdir) / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def clear_checkpoint(outdir: str | Path) -> None:
    """Remove the checkpoint file from *outdir* if it exists."""
    path = Path(outdir) / CHECKPOINT_FILENAME
    if path.exists():
        path.unlink()
    tmp_path = Path(outdir) / CHECKPOINT_TMP_FILENAME
    if tmp_path.exists():
        tmp_path.unlink()
