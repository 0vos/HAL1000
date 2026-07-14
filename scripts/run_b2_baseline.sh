#!/bin/bash
# B2 individual demo - run all 5 Skills with both happy and error paths
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
OUT="$ROOT/outputs/B2_skills"
INPUTS="$ROOT/data/tool_inputs"
LOG="$OUT/skill_run_log.jsonl"

rm -rf "$OUT"
mkdir -p "$OUT"

PY=/opt/conda/envs/hal/bin/python

run_skill() {
    local skill="$1"
    local input="$2"
    local label="$3"
    if [[ -z "$label" ]]; then
        label="ok"
    fi
    local outdir="$OUT/${skill}_${label}"
    mkdir -p "$outdir"
    echo "=== $skill :: $label :: $input ==="
    "$PY" "$CODE/b2_run_skill.py" --skill "$skill" --input "$input" --outdir "$outdir"
}

run_skill calculator         "$INPUTS/tool_input_calculator.json"         ok
run_skill calculator         "$INPUTS/tool_input_calculator_error.json"   err
run_skill file_reader        "$INPUTS/tool_input_file_reader.json"        ok
run_skill file_reader        "$INPUTS/tool_input_file_reader_error.json"  err
run_skill local_file_search  "$INPUTS/tool_input_file_search.json"        ok
run_skill local_file_search  "$INPUTS/tool_input_file_search_error.json"  err
run_skill table_analyzer     "$INPUTS/tool_input_table_analyzer.json"     ok
run_skill table_analyzer     "$INPUTS/tool_input_table_analyzer_error.json" err
run_skill format_converter   "$INPUTS/tool_input_format_converter.json"   ok
run_skill format_converter   "$INPUTS/tool_input_format_converter_error.json" err

echo ""
echo "=== merged run log ==="
"$PY" -c "
import json, pathlib
log = pathlib.Path('$OUT').rglob('skill_run_log.jsonl')
merged = []
for path in sorted(log):
    merged.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
with open('$LOG', 'w', encoding='utf-8') as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'merged {len(merged)} entries into $LOG')
"

echo ""
echo "=== summary ==="
"$PY" -c "
import json, pathlib
out = pathlib.Path('$OUT')
print(f'{\"skill\":<22} {\"status\":<8} {\"latency_ms\":>10}  {\"path\":<40}')
for path in sorted(out.rglob('*_result.json')):
    data = json.loads(path.read_text(encoding='utf-8'))
    rel = path.relative_to(out)
    print(f'{data[\"skill_name\"]:<22} {data[\"status\"]:<8} {data[\"latency_ms\"]:>10.3f}  {rel}')
"