#!/bin/bash
# Run B1 with 4 additional runtime_input samples (one per Skill)
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
DATA="$ROOT/data"

PY=/opt/conda/envs/hal/bin/python

declare -A CASES=(
    ["calc"]="runtime_input_calc.json"
    ["search"]="runtime_input_search.json"
    ["table"]="runtime_input_table.json"
    ["format"]="runtime_input_format.json"
)

for label in calc search table format; do
    OUT="$ROOT/outputs/full_demo_${label}"
    INPUT="$DATA/${CASES[$label]}"
    rm -rf "$OUT"
    echo "=== full_demo_${label} :: $INPUT ==="
    "$PY" "$CODE/run_full_demo.py" \
        --input "$INPUT" \
        --tools_config "$ROOT/configs/tools.yaml" \
        --memory_config "$ROOT/configs/memory.yaml" \
        --model_config "$ROOT/configs/model.yaml" \
        --llm_mode mock \
        --outdir "$OUT" 2>&1 | tail -10
    echo ""
done