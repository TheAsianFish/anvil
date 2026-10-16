# Spike: ONNX export of ModernBERT-base on CPU

## Question

Can `answerdotai/ModernBERT-base`, loaded as `AutoModelForSequenceClassification` with
`num_labels=4`, be exported to ONNX on CPU with transformers 5.20, and does ONNX Runtime
reproduce the PyTorch logits? Secondary: does dynamic int8 quantization keep parity, and is
the `reference_compile` hazard still real?

The plan assumed optimum-onnx would do the export. That is not possible next to transformers
5.x: optimum-onnx 0.1.0 depends on `transformers>=4.36,<4.58.0`, and uv's resolver reports
`optimum-onnx==0.1.0` plus `transformers==5.20.1` as unsatisfiable. So I tested
`torch.onnx.export` directly.

## Setup

- torch 2.14.1+cpu, transformers 5.20.1, onnx 1.23.2, onnxruntime 1.31.0, onnxscript 0.7.2,
  Python 3.12.3.
- 4 vCPU (AMD EPYC 7763), Linux, no GPU. `torch.set_num_threads(4)`, ORT
  `intra_op_num_threads=4`, CPUExecutionProvider.
- Classification head is random-init with `torch.manual_seed(0)`, so it is identical across
  configurations. Its logits are small (largest |logit| over the test set: 1.84).
- 32 fixed English sentences, 3 to 108 tokens. Parity is checked on all 32 as one padded batch
  and on each sentence alone at batch 1 (64 comparisons), against PyTorch eval-mode logits.
- Export traced with batch 2, length 24; batch and sequence axes dynamic.
- Latency: one 23-token sentence (median length), batch 1, 5 warmups, 20 timed runs; I report
  the median.

## What I tried

| Config | Exporter | Attention | reference_compile |
|---|---|---|---|
| dynamo-sdpa | `dynamo=True` (default) | sdpa | not passed |
| dynamo-eager | `dynamo=True` | eager | not passed |
| torchscript-sdpa | `dynamo=False`, opset 17 | sdpa | not passed |
| torchscript-eager | `dynamo=False`, opset 17 | eager | not passed |
| dynamo-sdpa-refcompile-kwarg-false | `dynamo=True` | sdpa | `from_pretrained(..., reference_compile=False)` |
| dynamo-sdpa-refcompile-config-true | `dynamo=True` | sdpa | `config.reference_compile = True` |

For each successful fp32 export I ran `quantize_dynamic(weight_type=QInt8)` four ways: as is;
after stripping value_info entries that duplicate initializers (only when the plain call
failed); with `per_channel=True`; and with the 22 MLP down-projection MatMuls (`mlp.Wo`) left
in fp32.

## Results

From `eval/results/spike-onnx-export/results.json`. PyTorch batch-1 median latency on the same
input was 66.1 to 69.4 ms across configs.

fp32:

| Config | Export s | Size MB | Max abs diff | Argmax agree | ORT ms (median) | Nodes |
|---|---|---|---|---|---|---|
| dynamo-sdpa | 24.8 | 601.3 | 8e-06 | 1.0 | 36.26 | 1250 |
| dynamo-eager | 24.0 | 600.7 | 6e-06 | 1.0 | 35.40 | 1008 |
| torchscript-sdpa | 2.6 | 599.1 | 8e-06 | 1.0 | 36.99 | 3156 |
| torchscript-eager | 2.7 | 599.1 | 6e-06 | 1.0 | 37.05 | 3030 |
| dynamo-sdpa-refcompile-kwarg-false | failed: `TypeError` | | | | | |
| dynamo-sdpa-refcompile-config-true | 23.3 | 601.3 | 8e-06 | 1.0 | 36.21 | 1250 |

int8 dynamic (on the dynamo configs the plain call fails, so these rows are on the stripped
model):

| Config | Variant | Size MB | Max abs diff | Mean abs diff | Argmax agree | ORT ms |
|---|---|---|---|---|---|---|
| dynamo-sdpa | per-tensor | 152.4 | 1.386 | 0.341 | 0.906 | 22.09 |
| dynamo-sdpa | per-channel | 153.1 | 3.579 | 0.737 | 0.094 | 22.76 |
| dynamo-sdpa | per-tensor, mlp.Wo fp32 | 210.8 | 1.072 | 0.115 | 1.0 | 24.29 |
| dynamo-eager | per-tensor | 151.8 | 1.302 | 0.323 | 0.906 | 21.66 |
| dynamo-eager | per-channel | 152.5 | 3.589 | 0.748 | 0.078 | 22.41 |
| dynamo-eager | per-tensor, mlp.Wo fp32 | 210.2 | 1.060 | 0.119 | 0.969 | 23.71 |
| torchscript-sdpa | per-tensor | 150.7 | 1.421 | 0.337 | 0.922 | 23.21 |
| torchscript-sdpa | per-channel | 151.4 | 3.579 | 0.749 | 0.063 | 24.12 |
| torchscript-sdpa | per-tensor, mlp.Wo fp32 | 209.1 | 0.762 | 0.120 | 0.984 | 25.64 |
| torchscript-eager | per-tensor | 150.7 | 1.302 | 0.323 | 0.906 | 22.66 |
| torchscript-eager | per-channel | 151.4 | 3.589 | 0.748 | 0.078 | 23.81 |
| torchscript-eager | per-tensor, mlp.Wo fp32 | 209.1 | 1.060 | 0.119 | 0.969 | 25.14 |

The refcompile-config-true int8 rows match dynamo-sdpa exactly. Total run: 309.8 s, $0.

## What failed and why

**fp32 export: nothing.** Both exporters, both attention implementations, padded batches and
single inputs all match PyTorch to under 1e-5. In PyTorch, the padded batch and the single
inputs themselves differ by up to 6e-06.

**`reference_compile`: the hazard no longer applies.** In transformers 5.20 the ModernBERT
modeling code has no `reference_compile` path; the only remaining mention is
`ModernBertConfig.to_dict()` dropping the key. Passing it to `from_pretrained` raises
`TypeError: ModernBertForSequenceClassification.__init__() got an unexpected keyword argument
'reference_compile'`. Setting it on the config does nothing: export and parity are identical
to dynamo-sdpa. So Anvil should not pass it at all.

**int8 on dynamo exports: `InferenceError` in the quantizer.** The message is
`[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (768) vs (4)`.
The dynamo exporter writes `value_info` entries for initializers, including
`model.classifier.weight` with shape [4, 768]. In dynamic mode the quantizer rewrites the
classifier `Gemm` (transB=1) into a `MatMul` and transposes the weight to [768, 4], but leaves
the stale value_info, so the next shape-inference pass fails. Dropping the value_info entries
whose names match initializers fixes it. TorchScript exports don't write those
entries and quantize as they are.

**int8 accuracy is poor, and most of the loss is in the MLP down-projection.** Per-tensor
int8 gives a mean logit error of 0.32 to 0.34 on logits whose largest magnitude is 1.84. To
find where it comes from I quantized one group of Linear layers at a time in a one-off probe
whose output I did not keep, so I am not quoting its numbers here; it pointed at the 22
`mlp.Wo` MatMuls. `mlp.Wo` reads the GeGLU output, and dynamic quantization uses one uint8
scale per tensor for activations, so my guess is that a few outliers in that activation make
the scale too coarse; I did not measure the activation range. The committed variant that keeps
`mlp.Wo` in fp32 is measured: mean error 0.11 to 0.12 and argmax agreement 0.97 to 1.0, at
+58 MB and about +2 ms. M5 adds the per-group probe to the script so the attribution is
measured, not remembered.

**Per-channel int8 is broken here.** It agrees with PyTorch on 6 to 9% of inputs, worse than
chance with 4 labels, and the result is the same whether or not the classifier head is
quantized. I did not find the root cause in the timebox. For now, don't use it.

Caveats: the head is untrained, so the margins between classes are small and argmax agreement
is a pessimistic proxy. Max tested length is 108 tokens. Latency is one 23-token input on a
shared 4-vCPU runner; the min/max spread is in results.json.

## Decision for M5

- **Exporter:** `torch.onnx.export(..., dynamo=True)` with `dynamic_shapes` for batch and
  sequence, `attn_implementation="sdpa"` (the default), no `reference_compile`. Parity and ORT
  latency match TorchScript, and the graph is smaller (1250 vs 3156 nodes). The
  TorchScript-only options are marked deprecated in `torch.onnx.export`'s docstring. The cost
  is export time (about 24 s vs 3 s), which doesn't matter for a one-off export. If dynamo
  ever breaks on a model, `dynamo=False` is a known-good fallback for ModernBERT.
- **Weights:** the dynamo exporter writes weights to `model.onnx.data` next to `model.onnx`.
  The exporter and the serving image must ship both files.
- **fp32 parity gate:** max abs logit diff <= 1e-4 against PyTorch on a fixed probe set
  (measured 8e-06, so about 12x headroom), and argmax agreement of 100%.
- **int8:** opt-in, not the default. Strip initializer value_info before `quantize_dynamic`,
  per-tensor only, with the MLP `Wo` MatMuls excluded. A logit tolerance does not work as the
  gate: measured max diffs are around 1 even in the best variant. Gate on the trained model
  instead: argmax agreement with the fp32 ONNX model >= 0.98 on the held-out set, and task
  metric drop <= 1 point. Report mean abs logit diff for information. I'll re-measure once
  there is a trained head.
- **Tests to add in M5:** a probe input longer than 512 tokens, and the value_info strip as a
  unit test on a tiny random-init ModernBERT.

## Reproduce

```bash
uv run python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export
```

It downloads the model (about 600 MB) on first run, writes ONNX files only to a temporary
directory, and takes about 5 minutes on 4 vCPUs. Raw output: `results.json`, `meta.json`,
`run.log` in `eval/results/spike-onnx-export/`.
