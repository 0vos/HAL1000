#!/bin/bash
# Full system demo via B1 Agent runtime (integrated mode, mock LLM)
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
DATA="$ROOT/data"

PY=/opt/conda/envs/hal/bin/python

OUT="$ROOT/outputs/full_demo"
rm -rf "$OUT"

echo "=== run_full_demo (mock mode) ==="
"$PY" "$CODE/run_full_demo.py" \
    --input "$DATA/runtime_input.json" \
    --tools_config "$ROOT/configs/tools.yaml" \
    --memory_config "$ROOT/configs/memory.yaml" \
    --model_config "$ROOT/configs/model.yaml" \
    --llm_mode mock \
    --outdir "$OUT" 2>&1 | tail -30

echo ""
echo "=== outputs ==="
ls "$OUT"