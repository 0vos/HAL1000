#!/bin/bash
set -euo pipefail

ROOT="/root/siton-tmp/HAL1000/agent"
PY="/opt/conda/envs/hal/bin/python"
OUT="$ROOT/outputs/B2_innovations"

mkdir -p "$OUT"

"$PY" -m unittest discover \
    -s "$ROOT/tests_b2" \
    -p "test_*.py" \
    -v 2>&1 | tee "$OUT/test_report.txt"

"$PY" "$ROOT/code/b2_run_skill.py" \
    --skill document_inspector \
    --input "$ROOT/data/tool_inputs/advanced/document_inspector_ok.json" \
    --outdir "$OUT/document_inspector_ok"

"$PY" "$ROOT/code/b2_run_skill.py" \
    --skill document_inspector \
    --input "$ROOT/data/tool_inputs/advanced/document_inspector_error.json" \
    --outdir "$OUT/document_inspector_error"

"$PY" "$ROOT/code/b2_run_skill.py" \
    --skill document_inspector \
    --input "$ROOT/data/tool_inputs/advanced/document_inspector_poster.json" \
    --outdir "$OUT/document_inspector_poster"

echo "B2 innovation outputs: $OUT"
