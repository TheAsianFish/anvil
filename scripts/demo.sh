#!/usr/bin/env bash
# The offline vertical slice: label the toy tickets with the rule-based fake teacher, train a
# tiny random-init ModernBERT on them, measure its agreement with the teacher, then run a
# three-round active-learning loop that retrains from scratch at 50, 100 and 200 labels.
# Then the same three steps for extraction: the fake teacher marks slots in restaurant
# queries with a fifth of its offsets off by one, alignment repairs them, and a tiny tagger
# is scored by entity F1 and token kappa.
set -euo pipefail

home="${1:-.anvil/demo}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toy="$here/examples/toy"
slots="$here/examples/toy-extract"

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
echo "== anvil loop (random selection, three rounds, fresh student each round)"
uv run anvil loop --spec "$toy/anvil.yaml" --teacher fake --budgets 50,100,200 --id demo
echo "== anvil label (extraction, fake teacher with off-by-one offsets)"
uv run anvil label --spec "$slots/anvil.yaml" --inputs "$slots/inputs.jsonl" --n 240 --teacher fake
echo "== anvil train (tiny random-init ModernBERT tagger; aligns every answer first)"
uv run anvil train --spec "$slots/anvil.yaml" --teacher fake
echo "== anvil eval (entity F1 and token kappa against the teacher)"
uv run anvil eval --spec "$slots/anvil.yaml"
echo "== demo finished in $(( $(date +%s) - start ))s"
