#!/bin/bash
# B2 advanced demo - test error codes, composite skill, sandbox, and timeouts
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
OUT="$ROOT/outputs/B2_advanced"
INPUTS="$ROOT/data/tool_inputs"

rm -rf "$OUT"
mkdir -p "$OUT"

PY=/opt/conda/envs/hal/bin/python

echo "=== baseline: error code propagation (run base skills with error inputs) ==="
mkdir -p "$OUT/baseline_error_codes"
"$PY" "$CODE/b2_run_skill.py" --skill calculator \
    --input "$INPUTS/tool_input_calculator_error.json" \
    --outdir "$OUT/baseline_error_codes/calculator"
"$PY" "$CODE/b2_run_skill.py" --skill file_reader \
    --input "$INPUTS/tool_input_file_reader_error.json" \
    --outdir "$OUT/baseline_error_codes/file_reader"
"$PY" "$CODE/b2_run_skill.py" --skill table_analyzer \
    --input "$INPUTS/tool_input_table_analyzer_error.json" \
    --outdir "$OUT/baseline_error_codes/table_analyzer"
"$PY" "$CODE/b2_run_skill.py" --skill local_file_search \
    --input "$INPUTS/tool_input_file_search_error.json" \
    --outdir "$OUT/baseline_error_codes/local_file_search"
"$PY" "$CODE/b2_run_skill.py" --skill format_converter \
    --input "$INPUTS/tool_input_format_converter_error.json" \
    --outdir "$OUT/baseline_error_codes/format_converter"

echo ""
echo "=== new: error code overflow / unsupported on calculator ==="
mkdir -p "$OUT/calculator_edge"
"$PY" "$CODE/b2_run_skill.py" --skill calculator \
    --input "$INPUTS/advanced/calc_overflow.json" \
    --outdir "$OUT/calculator_edge/overflow"
"$PY" "$CODE/b2_run_skill.py" --skill calculator \
    --input "$INPUTS/advanced/calc_unsupported.json" \
    --outdir "$OUT/calculator_edge/unsupported"

echo ""
echo "=== composite skill (read_and_convert) ==="
mkdir -p "$OUT/composite"
"$PY" "$CODE/b2_advanced.py" --skill read_and_convert \
    --input "$INPUTS/advanced/composite_ok.json" \
    --outdir "$OUT/composite/ok"
"$PY" "$CODE/b2_advanced.py" --skill read_and_convert \
    --input "$INPUTS/advanced/composite_err.json" \
    --outdir "$OUT/composite/err"

echo ""
echo "=== sandbox skill (safe_python_exec) ==="
mkdir -p "$OUT/sandbox"
"$PY" "$CODE/b2_advanced.py" --skill safe_python_exec \
    --input "$INPUTS/advanced/sandbox_ok.json" \
    --outdir "$OUT/sandbox/ok"
"$PY" "$CODE/b2_advanced.py" --skill safe_python_exec \
    --input "$INPUTS/advanced/sandbox_blocked.json" \
    --outdir "$OUT/sandbox/blocked"
"$PY" "$CODE/b2_advanced.py" --skill safe_python_exec \
    --input "$INPUTS/advanced/sandbox_timeout.json" \
    --outdir "$OUT/sandbox/timeout"

echo ""
echo "=== summary ==="
"$PY" - <<'PYEOF'
import json, pathlib
out = pathlib.Path('/root/siton-tmp/HAL1000/agent/outputs/B2_advanced')
print(f'{"label":<35} {"skill":<20} {"status":<10} {"code":<22} {"latency_ms":>10}')
for path in sorted(out.rglob('*_result.json')):
    d = json.loads(path.read_text(encoding='utf-8'))
    label = '_'.join(path.relative_to(out).parts[:-1])
    code = (d.get('error') or {}).get('code', '-')
    print(f'{label:<35} {d["skill_name"]:<20} {d["status"]:<10} {code:<22} {d["latency_ms"]:>10.3f}')
PYEOF