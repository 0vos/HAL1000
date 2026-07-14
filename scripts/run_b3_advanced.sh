#!/bin/bash
# B3 advanced demo: auto_schema, retry, cache, stats
set -e
ROOT="/root/siton-tmp/HAL1000/agent"
CODE="$ROOT/code"
DATA="$ROOT/data"
CFG="$ROOT/configs/tools.yaml"
OUT="$ROOT/outputs/B3_advanced"
# preserve pre-uploaded fixtures
mv "$OUT/batch_calls.json" /tmp/batch_calls.json 2>/dev/null || true
rm -rf "$OUT"
mkdir -p "$OUT"
mv /tmp/batch_calls.json "$OUT/batch_calls.json" 2>/dev/null || true

PY=/opt/conda/envs/hal/bin/python

echo "=== 1. auto_schema: per-function ==="
"$PY" "$CODE/b3_advanced.py" auto_schema --module skills.calculator --function calculator --outdir "$OUT/auto_schema/calculator"
"$PY" "$CODE/b3_advanced.py" auto_schema --module skills.file_reader --function file_reader --outdir "$OUT/auto_schema/file_reader"
"$PY" "$CODE/b3_advanced.py" auto_schema --module skills.table_analyzer --function table_analyzer --outdir "$OUT/auto_schema/table_analyzer"
"$PY" "$CODE/b3_advanced.py" auto_schema --module skills.local_file_search --function local_file_search --outdir "$OUT/auto_schema/local_file_search"
"$PY" "$CODE/b3_advanced.py" auto_schema --module skills.format_converter --function format_converter --outdir "$OUT/auto_schema/format_converter"

echo ""
echo "=== 2. auto_schema: full module ==="
"$PY" "$CODE/b3_advanced.py" auto_schema --module composite_skill --outdir "$OUT/auto_schema/composite_module"

echo ""
echo "=== 3. retry: run with retry_attempts=3 on a mixed batch ==="
"$PY" "$CODE/b3_advanced.py" execute \
    --tools_config "$CFG" --toolset basic_tools \
    --tool_calls "$DATA/messages/ai_message_with_tool_calls.json" \
    --retry 3 --stats --stats_path "$OUT/tool_stats.json" \
    --outdir "$OUT/retry_then_stats"

echo ""
echo "=== 4. cache: first call miss, second call hit ==="
"$PY" "$CODE/b3_advanced.py" execute \
    --tools_config "$CFG" --toolset basic_tools \
    --tool_calls "$DATA/messages/ai_message_with_tool_calls.json" \
    --cache --cache_path "$OUT/tool_cache.json" \
    --stats --stats_path "$OUT/tool_stats.json" \
    --outdir "$OUT/cache_first"
"$PY" "$CODE/b3_advanced.py" execute \
    --tools_config "$CFG" --toolset basic_tools \
    --tool_calls "$DATA/messages/ai_message_with_tool_calls.json" \
    --cache --cache_path "$OUT/tool_cache.json" \
    --stats --stats_path "$OUT/tool_stats.json" \
    --outdir "$OUT/cache_second"

echo ""
echo "=== 5. batch with multiple distinct tools + cache ==="
"$PY" "$CODE/b3_advanced.py" execute \
    --tools_config "$CFG" --toolset basic_tools \
    --tool_calls "$OUT/batch_calls.json" \
    --cache --cache_path "$OUT/tool_cache.json" \
    --stats --stats_path "$OUT/tool_stats.json" \
    --outdir "$OUT/batch"

echo ""
echo "=== summary ==="
"$PY" - <<'PYEOF'
import json, pathlib, statistics
out = pathlib.Path('/root/siton-tmp/HAL1000/agent/outputs/B3_advanced')
stats_path = out / 'tool_stats.json'
if not stats_path.exists():
    print('no stats file found')
else:
    raw = json.loads(stats_path.read_text(encoding='utf-8'))
    counts = raw.get('counts', {})
    successes = raw.get('success', {})
    failures = raw.get('failures', {})
    latencies = raw.get('latencies', {})
    total_calls = sum(counts.values())
    cache_hits = raw.get('cache_hits', 0)
    cache_misses = raw.get('cache_misses', 0)
    retry_attempts = raw.get('retry_attempts', 0)
    print('total calls:', total_calls)
    print('cache hits:', cache_hits)
    print('cache misses:', cache_misses)
    print('retry attempts:', retry_attempts)
    print()
    def pct(values, p):
        if not values: return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c: return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)
    print(f'{"tool":<22} {"calls":>6} {"ok":>6} {"err":>6} {"fail_rate":>10} {"avg_ms":>8} {"p95_ms":>8}')
    for name in sorted(counts.keys()):
        total = counts[name]
        ok = successes.get(name, 0)
        err = total - ok
        fr = err / total if total else 0.0
        lat = latencies.get(name, [])
        avg = statistics.fmean(lat) if lat else 0.0
        p95 = pct(lat, 95)
        print(f'{name:<22} {total:>6} {ok:>6} {err:>6} {fr:>10.2%} {avg:>8.2f} {p95:>8.2f}')
    # also dump snapshot
    snap_path = out / 'tool_stats_snapshot.json'
    if not snap_path.exists():
        snap = {
            'total_calls': total_calls,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'retry_attempts': retry_attempts,
            'tools': {
                name: {
                    'calls': counts[name],
                    'successes': successes.get(name, 0),
                    'failures': counts[name] - successes.get(name, 0),
                    'failure_rate': round((counts[name] - successes.get(name, 0)) / counts[name], 4) if counts[name] else 0.0,
                    'avg_latency_ms': round(statistics.fmean(latencies.get(name, [0])), 3) if latencies.get(name) else 0.0,
                    'p50_latency_ms': round(pct(latencies.get(name, []), 50), 3),
                    'p95_latency_ms': round(pct(latencies.get(name, []), 95), 3),
                    'error_codes': failures.get(name, {}),
                }
                for name in counts
            },
        }
        snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'wrote snapshot: {snap_path}')
PYEOF