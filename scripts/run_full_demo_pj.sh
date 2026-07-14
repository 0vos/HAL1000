#!/bin/bash
# Full system demo with REAL prompt_json model (Qwen3.5-4B)
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
DATA="$ROOT/data"

PY=/opt/conda/envs/hal/bin/python

OUT="$ROOT/outputs/full_demo_prompt_json"
rm -rf "$OUT"

echo "=== run_full_demo (prompt_json mode, real model) ==="
"$PY" "$CODE/run_full_demo.py" \
    --input "$DATA/runtime_input.json" \
    --tools_config "$ROOT/configs/tools.yaml" \
    --memory_config "$ROOT/configs/memory.yaml" \
    --model_config "$ROOT/configs/model.yaml" \
    --llm_mode prompt_json \
    --outdir "$OUT" 2>&1 | tail -30