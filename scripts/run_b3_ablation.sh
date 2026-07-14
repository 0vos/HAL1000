#!/bin/bash
# B3 ablation: schema detailed vs minimal on real Qwen3.5-4B (prompt_json)
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
CFG="$ROOT/configs/tools.yaml"
OUT="$ROOT/outputs/B3_ablation/prompt_json"

rm -rf "$OUT"

PY=/opt/conda/envs/hal/bin/python

echo "=== prompt_json mode: 5 samples ==="
"$PY" "$CODE/schema_ablation.py" \
    --mode prompt_json \
    --model_config "$ROOT/configs/model.yaml" \
    --tools_config "$CFG" \
    --limit 5 \
    --outdir "$OUT"

cat "$OUT/comparison.md"