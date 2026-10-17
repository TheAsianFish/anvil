# Results

Every number here was produced by a command in CONTRIBUTING.md and its raw output is committed
under `eval/results/`. Anything not yet run is listed under "To measure" with its command and a
cost estimate. No evaluation so far has called a real teacher: there is no API key in the build
environment yet.

## Measured

### ONNX export spike: ModernBERT-base on CPU (M1)

- Command: `make spike-onnx` (`uv run python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export`)
- Raw output: `eval/results/spike-onnx-export/results.json`, `meta.json`, `run.log`. `meta.json` names
  the SHA the tree was checked out at (`94389b5`); the script itself landed in the next commit
  (`11d3def`), and the line numbers in `run.log` match that committed version.
- Setup: `answerdotai/ModernBERT-base` with a random-init 4-class head (seed 0), torch 2.14.1+cpu,
  transformers 5.20.1, onnxruntime 1.31.0, onnxscript 0.7.2 (locked), 4 vCPU x86_64 GitHub-hosted runner. Parity over 32 sentences
  (3 to 108 tokens), as one padded batch and one by one (64 comparisons). Latency: one
  23-token sentence, batch 1, median of 20 runs after 5 warm-ups. Wall time 309.8 s, $0.
- Write-up: [docs/notes/onnx-export-spike.md](docs/notes/onnx-export-spike.md)

| Export | Size MB | Max abs logit diff vs PyTorch | Argmax agreement | ORT ms (median) |
|---|---|---|---|---|
| fp32, `torch.onnx` dynamo, sdpa | 601.3 | 8e-06 | 1.000 | 36.26 |
| fp32, `torch.onnx` TorchScript, sdpa | 599.1 | 8e-06 | 1.000 | 36.99 |
| int8 dynamic, per-tensor (dynamo) | 152.4 | 1.386 | 0.906 | 22.09 |
| int8 dynamic, per-tensor, `mlp.Wo` kept fp32 (dynamo) | 210.8 | 1.072 | 1.000 | 24.29 |
| int8 dynamic, per-channel (dynamo) | 153.1 | 3.579 | 0.094 | 22.76 |

PyTorch on the same input: 66.11 ms median (dynamo-sdpa run). The head is untrained, so its
logits are small (largest 1.84) and int8 agreement is a pessimistic proxy until M5 re-measures
on a trained student.

## To measure

| What | Command | Needs | Estimated cost |
|---|---|---|---|
| Live teacher test and the replay recordings (8 toy inputs) | `make record` | `ANTHROPIC_API_KEY` | under $0.01 with `claude-haiku-5-5` |
| Student vs Claude on the toy task (91 held-out + 200 pool labels) | set `teacher.model` in `examples/toy/anvil.yaml`, then `anvil label -i inputs.jsonl -n 200 && anvil train && anvil eval` from `examples/toy/` | `ANTHROPIC_API_KEY` | about $0.04 with Haiku 5.5, about $1.60 with Opus 5.5 (300 labels at roughly 300 input and 200 output tokens each) |
| Parity on AG News and Banking77 | `make eval-classify` (M2) | key | see PLAN.md |

The toy task is synthetic and only exercises the pipeline; real parity numbers come from the
M2 datasets.
