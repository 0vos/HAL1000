#!/bin/bash
# One-command launcher for all B2 + B3 personal demos and full-system demo.
# Usage:
#     bash run_all_demos.sh           # everything, mock LLM
#     bash run_all_demos.sh pj        # everything, real Qwen3.5-4B in prompt_json mode
set -e
cd "$(dirname "$0")"
ROOT="/root/siton-tmp/HAL1000/agent"

MODE="${1:-mock}"

echo "=================================================="
echo "B2 + B3 + full-system demo (mode=$MODE)"
echo "=================================================="

bash "$ROOT/scripts/run_b2_baseline.sh"     > /dev/null && echo "[ok] B2 baseline"
bash "$ROOT/scripts/run_b2_advanced.sh"     > /dev/null && echo "[ok] B2 advanced"
bash "$ROOT/scripts/run_b3_baseline.sh"     > /dev/null && echo "[ok] B3 baseline"
bash "$ROOT/scripts/run_b3_advanced.sh"     > /dev/null && echo "[ok] B3 advanced"

if [[ "$MODE" == "pj" ]]; then
    bash "$ROOT/scripts/run_b3_ablation.sh"  > /dev/null && echo "[ok] B3 ablation (prompt_json)"
    bash "$ROOT/scripts/run_full_demo_pj.sh" > /dev/null && echo "[ok] Full demo (prompt_json)"
else
    # mock ablation is informative but shows identical rates — skip
    bash "$ROOT/scripts/run_full_system_demo.sh" > /dev/null && echo "[ok] Full demo (mock)"
fi

bash "$ROOT/scripts/run_full_demo_skills.sh" > /dev/null && echo "[ok] Full demo x4 skills (mock)"

echo "=================================================="
echo "All artifacts under $ROOT/outputs/"
echo "Reports under $ROOT/reports/"
echo "=================================================="
ls "$ROOT/outputs"