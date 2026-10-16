#!/usr/bin/env bash
# The offline vertical slice: label the toy tickets with the rule-based fake teacher, train a
# tiny random-init ModernBERT on them, and measure its agreement with the teacher.
set -euo pipefail

home="${1:-.anvil/demo}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toy="$here/examples/toy"

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
echo "== demo finished in $(( $(date +%s) - start ))s"
