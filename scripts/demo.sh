#!/usr/bin/env bash
# The offline vertical slice: label the toy tickets with the rule-based fake teacher, train a
# tiny random-init ModernBERT on them, measure its agreement with the teacher, then run a
# three-round active-learning loop that retrains from scratch at 50, 100 and 200 labels.
# Before the loop, the first student is calibrated, exported to ONNX (parity-checked against
# PyTorch) and served: an input under the threshold escalates to the fake teacher.
# Then the same three steps for extraction: the fake teacher marks slots in restaurant
# queries with a fifth of its offsets off by one, alignment repairs them, and a tiny tagger
# is scored by entity F1 and token kappa. Last, generation: a fake teacher writes SQL for
# questions about one table, a LoRA adapter on a tiny random-init Qwen2 decoder learns from it,
# and the adapter is merged into the base weights, checked against the adapter's answers.
set -euo pipefail

home="${1:-.anvil/demo}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toy="$here/examples/toy"
slots="$here/examples/toy-extract"
sql="$here/examples/toy-generate"

case "$home" in
  */.anvil/*|*/.anvil) ;;
  *) echo "demo home must be under a .anvil directory, got: $home" >&2; exit 2 ;;
esac
rm -rf "$home"
export ANVIL_HOME="$home"
export ANVIL_LOG_LEVEL="${ANVIL_LOG_LEVEL:-WARNING}"
start=$(date +%s)

echo "== anvil label (fake teacher, offline)"
uv run anvil label --spec "$toy/anvil.yaml" --inputs "$toy/inputs.jsonl" --n 200 --teacher fake
echo "== anvil train (tiny random-init ModernBERT)"
uv run anvil train --spec "$toy/anvil.yaml" --teacher fake
echo "== anvil eval (held-out agreement with the teacher)"
uv run anvil eval --spec "$toy/anvil.yaml"
echo "== anvil calibrate (temperature and the threshold that keeps kappa 0.9 with the teacher)"
uv run anvil calibrate --spec "$toy/anvil.yaml"
echo "== anvil export (ONNX fp32, checked against PyTorch before it can be served)"
uv run anvil export --spec "$toy/anvil.yaml" --out "$home/bundle"
echo "== anvil serve (ONNX Runtime; inputs under the threshold escalate to the fake teacher)"
port="${DEMO_PORT:-8765}"
uv run anvil serve "$home/bundle" --teacher fake --port "$port" 2> "$home/serve.log" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$port/healthz" > /dev/null && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$port/predict" -H 'content-type: application/json' \
  -d '{"texts": ["I was charged twice for my invoice", "Please add dark mode",
       "The export button does nothing and I want my money back"]}' \
  | uv run python "$here/scripts/show_predictions.py"
kill $server && wait $server 2>/dev/null || true
echo "== anvil loop (random selection, three rounds, fresh student each round)"
uv run anvil loop --spec "$toy/anvil.yaml" --teacher fake --budgets 50,100,200 --id demo
echo "== anvil label (extraction, fake teacher with off-by-one offsets)"
uv run anvil label --spec "$slots/anvil.yaml" --inputs "$slots/inputs.jsonl" --n 240 --teacher fake
echo "== anvil train (tiny random-init ModernBERT tagger; aligns every answer first)"
uv run anvil train --spec "$slots/anvil.yaml" --teacher fake
echo "== anvil eval (entity F1 and token kappa against the teacher)"
uv run anvil eval --spec "$slots/anvil.yaml"
echo "== anvil label (generation, fake teacher writes one SQL query per question)"
uv run anvil label --spec "$sql/anvil.yaml" --inputs "$sql/inputs.jsonl" --n 120 --teacher fake
echo "== anvil train (LoRA on a tiny random-init Qwen2; chat template, prompt tokens masked)"
uv run anvil train --spec "$sql/anvil.yaml" --teacher fake
echo "== anvil eval (greedy answers vs the teacher's; a random-init 2-layer decoder learns the"
echo "   shape of the queries, not yet to copy values: real parity comes from Qwen2.5-0.5B)"
uv run anvil eval --spec "$sql/anvil.yaml"
echo "== anvil export (adapter merged into the base weights, same answers or nothing is written)"
uv run anvil export --spec "$sql/anvil.yaml"
echo "== demo finished in $(( $(date +%s) - start ))s"
