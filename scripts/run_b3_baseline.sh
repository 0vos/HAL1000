#!/bin/bash
# B3 individual demo - generate schema + execute 4 kinds of tool_call
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
DATA="$ROOT/data"
OUT="$ROOT/outputs/B3_tools"
CFG="$ROOT/configs/tools.yaml"

rm -rf "$OUT"
mkdir -p "$OUT"

PY=/opt/conda/envs/hal/bin/python

# 1. export_schema
echo "=== export_schema ==="
"$PY" "$CODE/b3_tool_layer.py" \
    --tools_config "$CFG" \
    --toolset basic_tools \
    --export_schema \
    --outdir "$OUT/schema"

# 2-5. execute tool_calls
for label in with_tool_calls format_converter_valid unknown_tool missing_required; do
    case "$label" in
        with_tool_calls)        input="$DATA/messages/ai_message_with_tool_calls.json" ;;
        format_converter_valid) input="$DATA/messages/b3_tool_call_format_converter_valid.json" ;;
        unknown_tool)           input="$DATA/messages/b3_tool_call_unknown_tool.json" ;;
        missing_required)       input="$DATA/messages/b3_tool_call_missing_required.json" ;;
    esac
    outdir="$OUT/$label"
    mkdir -p "$outdir"
    echo "=== execute $label :: $input ==="
    "$PY" "$CODE/b3_tool_layer.py" \
        --tools_config "$CFG" \
        --toolset basic_tools \
        --tool_calls "$input" \
        --execute \
        --outdir "$outdir"
done

# 6. merge tool_call_log
"$PY" - <<'PYEOF'
import json, pathlib
out = pathlib.Path('/root/siton-tmp/HAL1000/agent/outputs/B3_tools')
log = out / 'tool_call_log.jsonl'
records = []
for path in sorted(out.rglob('tool_call_log.jsonl')):
    records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
with log.open('w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'merged {len(records)} records into {log}')
PYEOF

echo ""
echo "=== summary ==="
"$PY" - <<'PYEOF'
import json, pathlib
out = pathlib.Path('/root/siton-tmp/HAL1000/agent/outputs/B3_tools')
print(f'{"label":<28} {"tool":<22} {"status":<8} {"latency_ms":>10}')
for path in sorted(out.rglob('tool_messages.json')):
    msgs = json.loads(path.read_text(encoding='utf-8'))
    label = path.relative_to(out).parts[0]
    for m in msgs:
        try:
            res = json.loads(m['content'])
            print(f'{label:<28} {m["name"]:<22} {res["status"]:<8} {res.get("latency_ms", 0):>10.3f}')
        except Exception as e:
            print(f'{label:<28} {m["name"]:<22} (parse error) {e}')
PYEOF