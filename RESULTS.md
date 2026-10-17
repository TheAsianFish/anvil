# Results

Every number here was produced by a command in CONTRIBUTING.md and its raw output is committed
under `eval/results/`. Anything not yet run is listed under "To measure" with its command and a
cost estimate. No evaluation so far has called a real teacher: there is no API key in the build
environment yet, so the M2 classification runs use each dataset's gold labels as the teacher
(model id `gold`). They measure the training loop, the student and the statistics on real data;
they are not parity-with-Claude numbers, which are listed under "To measure".

## Measured

### Classification parity: AG News and Banking77 (M2)

- Command: `make eval-classify` (`python -m eval.classify ag_news banking77`, then
  `python -m eval.render`, which writes the block below from `eval/results/classify-*/`).
- Method: a seeded sample of 2,500 distinct inputs from each dataset's train split at a pinned
  revision; the store's hash split puts about 20% in held-out. The teacher labels up to 500
  held-out and 2,000 pool inputs; a fresh student trains on the pool labels (10% of them held
  back by text hash for early stopping) and is scored against the teacher on the held-out
  labels. Kappa intervals are 95% percentile bootstrap intervals over held-out items (1,000
  resamples; the bootstrap seed is the run's student seed, so a `--seed 1` repeat also redraws
  its resamples). They cover held-out sampling only, not training randomness, and the coverage
  check in the tests is for one regime (kappa 0.7, 200 items, 5 labels); percentile intervals
  of a statistic bounded at 1 sit high when kappa is near 1 or the sample is small, which is
  how the ceiling's interval (200 items, high kappa) should be read once it is measured. The
  TF-IDF baseline is a bag-of-bigrams logistic regression (default regularisation, untuned)
  trained on all 2,000 teacher labels, including the 10% the encoder holds back to pick its
  epoch. The ceiling is the teacher labelling 200 held-out inputs again under a paraphrased
  prompt.

Findings so far (from the table below):

- On Banking77 (77 intents, ~12-word inputs) the fine-tuned encoder beats a TF-IDF + logistic
  regression student trained on the same 2,000 labels by 0.18 kappa (paired 95% interval
  [0.141, 0.223]). On AG News (4 topics, ~40-word inputs) the paired difference is +0.009
  [-0.033, 0.054]: a tie, since bag-of-words is already strong on long topical text.
- The exact-hash split cannot catch near-duplicates: 17 of 496 Banking77 held-out inputs (3%)
  and 3 of 477 AG News ones share at least 80% of their words with a labelled pool input.
  Every report counts them, so a leak would show up as the count grows rather than hide.
- Student confidence tracks agreement: on Banking77 the lowest-confidence tenth of held-out
  inputs (50 items) agrees 40% of the time and the top six tenths 98-100%; the seed-1 run
  puts the lowest tenth at 30%, so the slope is the finding, not the exact figure. That is
  the signal the uncertainty selection (M3) and the escalating cascade (M5) depend on.
- Training randomness moves the headline about as much as the held-out sample does. On AG
  News, changing only the student seed (initialisation, data order, validation slice) took
  held-out kappa from 0.844 to 0.896: a 0.053 swing, about 70% of one run's 95% interval
  width (0.073 on average). The bootstrap interval covers "which held-out items", not "which
  training run", so a single run's interval understates the uncertainty of the recipe. The
  epoch choice is part of it: seed 0's validation kappa went 0.821, 0.780, 0.821 and it kept
  epoch 1; seed 1 went 0.752, 0.707, 0.773, 0.773 and kept epoch 4. On Banking77 the two
  seeds land on the same kappa (0.842; both 419 of 496 correct) yet disagree with each other
  on 72 held-out items (15%, from the `predictions` in the two `report.json` files): a stable
  aggregate over item-level churn, so which inputs look uncertain depends on the training run.
  Two seeds show the size of the effect, not its distribution; M3 runs three per point.

<!-- BEGIN generated:classify -->
| Dataset | Teacher | Student | Teacher labels | Held-out | Student kappa [95% CI] | Accuracy | Macro F1 | Majority acc. | TF-IDF kappa [95% CI] | Student minus TF-IDF kappa [95% paired CI] | Teacher ceiling kappa | Label cost | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | gold | ModernBERT-base | 2000 | 477 | 0.844 [0.805, 0.882] | 0.883 | 0.882 | 0.260 | 0.835 [0.795, 0.873] | +0.009 [-0.033, 0.054] | n/a | $0.00 | 36 min |
| banking77 | gold | ModernBERT-base | 2000 | 496 | 0.842 [0.807, 0.873] | 0.845 | 0.814 | 0.024 | 0.662 [0.621, 0.702] | +0.180 [0.141, 0.223] | n/a | $0.00 | 57 min |

#### ag_news, teacher `gold`, student `ModernBERT-base`

- Raw output: `eval/results/classify-ag_news-gold/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.classify ag_news banking77 --teacher gold` at `74cca5e`; seed 0, 1000 bootstrap resamples; 4 CPUs; wall 2170 s; spent $0.0000
- Data: fancyzhx/ag_news@eb185aa (AG's corpus, used under its academic terms (http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)), 2500 inputs sampled (seed 0, sha256 3cf708037e01)
- Training: 1806 examples and 194 held back to pick the epoch, out of 2000 teacher labels; 3 epochs, 339 steps, best epoch 1, 2044 s
- Near-duplicates: 3 of 477 held-out inputs share at least 80% of their words with a labelled pool input (identical texts never straddle the split; these are the ones a hash cannot catch)
- The teacher here is the dataset's gold labels (no API key in this run), so this row measures the harness and how well the student learns from 2,000 clean labels; it is not a parity-with-Claude number.
- Weakest classes with at least 5 held-out items (F1 vs teacher, support): `business` 0.81 (105), `sci_tech` 0.82 (124), `world` 0.92 (130), `sports` 0.97 (118)
- Top confusions (teacher -> student): `sci_tech` -> `business` 31, `world` -> `business` 9, `world` -> `sports` 5, `business` -> `sci_tech` 4, `sci_tech` -> `world` 2
- Agreement by confidence: 0.34-0.67: 0.54 (n=48), 0.68-0.84: 0.75 (n=48), 0.85-0.93: 0.90 (n=48), 0.93-0.96: 0.90 (n=48), 0.96-0.98: 0.88 (n=48), 0.98-0.99: 0.94 (n=48), 0.99-0.99: 0.98 (n=48), 0.99-1.00: 0.96 (n=47), 1.00-1.00: 1.00 (n=47), 1.00-1.00: 1.00 (n=47)
- Agreement by length (words): 14-31: 0.85 (n=120), 32-37: 0.89 (n=119), 37-43: 0.92 (n=119), 43-107: 0.87 (n=119)

#### banking77, teacher `gold`, student `ModernBERT-base`

- Raw output: `eval/results/classify-banking77-gold/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.classify banking77 --teacher gold` at `d4f4016`; seed 0, 1000 bootstrap resamples; 4 CPUs; wall 3391 s; spent $0.0000
- Data: legacy-datasets/banking77@f541215 (CC-BY-4.0 (Casanueva et al., 2020, PolyAI)), 2500 inputs sampled (seed 0, sha256 2aeaf7290b43)
- Training: 1809 examples and 191 held back to pick the epoch, out of 2000 teacher labels; 10 epochs, 1140 steps, best epoch 8, 3320 s
- Near-duplicates: 17 of 496 held-out inputs share at least 80% of their words with a labelled pool input (identical texts never straddle the split; these are the ones a hash cannot catch)
- The teacher here is the dataset's gold labels (no API key in this run), so this row measures the harness and how well the student learns from 2,000 clean labels; it is not a parity-with-Claude number.
- Weakest classes with at least 5 held-out items (F1 vs teacher, support): `card_not_working` 0.60 (6), `pending_transfer` 0.62 (8), `atm_support` 0.67 (5), `card_delivery_estimate` 0.67 (6), `transfer_not_received_by_recipient` 0.67 (6)
- Top confusions (teacher -> student): `card_not_working` -> `declined_card_payment` 3, `transfer_timing` -> `pending_transfer` 3, `card_delivery_estimate` -> `card_arrival` 2, `card_swallowed` -> `lost_or_stolen_card` 2, `compromised_card` -> `card_payment_not_recognised` 2
- Agreement by confidence: 0.19-0.56: 0.40 (n=50), 0.57-0.78: 0.60 (n=50), 0.78-0.93: 0.64 (n=50), 0.93-0.98: 0.88 (n=50), 0.98-0.99: 0.98 (n=50), 0.99-1.00: 0.98 (n=50), 1.00-1.00: 0.98 (n=49), 1.00-1.00: 1.00 (n=49), 1.00-1.00: 1.00 (n=49), 1.00-1.00: 1.00 (n=49)
- Agreement by length (words): 4-7: 0.86 (n=124), 7-10: 0.89 (n=124), 10-13: 0.78 (n=124), 13-53: 0.85 (n=124)
<!-- END generated:classify -->

Training-seed spread (only the student seed changes: weight init, data order, validation slice):

<!-- BEGIN generated:classify-seeds -->
| Dataset | Teacher | Student | Seeds | Kappa per seed | Mean | Range (max - min) | Mean 95% CI width |
|---|---|---|---|---|---|---|---|
| ag_news | gold | ModernBERT-base | 2 | 0: 0.844, 1: 0.896 | 0.870 | 0.053 | 0.073 |
| banking77 | gold | ModernBERT-base | 2 | 0: 0.842, 1: 0.842 | 0.842 | 0.000 | 0.067 |
<!-- END generated:classify-seeds -->

### Selection on synthetic data (M3)

- Command: `make eval-select` (`python -m eval.select_study`, then `python -m eval.render`,
  which writes the block below from `eval/results/select-synthetic/`). Offline, $0.
- Method: the acquisition strategies of `anvil/select.py` with a cheap student, to look at the
  selection mechanism before the text frontier. Data: scikit-learn Gaussian blobs (10 classes,
  12 features, cluster standard deviation 2.3, data seed 0), a pool of 2,000 points and a
  separate test set of 1,000. In `blobs` every pool point is distinct; in `blobs-duplicated`
  400 distinct points each appear 5 times with tiny jitter (standard deviation 0.001), so a
  batch chosen on uncertainty alone can be several copies of one point. The true labels play
  the teacher. The student is a logistic regression (lbfgs, multinomial), retrained from
  scratch every round on the labelled points; it scores the unlabelled pool with its class
  probabilities, and the raw features serve as embeddings. Cold start: 20 points drawn at
  random, the same for every strategy within a seed; then batches of 10 up to 400 labels.
  20 seeds. "Labels to 0.95" is the smallest label count whose test accuracy reaches 0.95; the
  mean and range are over the seeds that reach it, and the next column counts those that do
  not by 400 labels. Accuracy columns are means over seeds.

<!-- BEGIN generated:select -->
- Raw output: `eval/results/select-synthetic/` (`report.json`, `meta.json`)
- Command: `python -m eval.select_study` at `c1a7a44`; 20 seeds, data seed 0; 4 CPUs; wall 25 s; spent $0.00

**blobs**: 2,000 pool points (2,000 distinct), 1,000 test points; a student trained on the whole pool scores 0.973 test accuracy (95% of it: 0.924).

| Strategy | Labels to 0.95 accuracy, mean [min-max] | Seeds not reaching 0.95 by 400 | Labels to 0.924 (95% of full pool), mean [min-max] | Seeds not reaching it | Accuracy at 50 | Accuracy at 100 | Accuracy at 200 | Accuracy at 400 |
|---|---|---|---|---|---|---|---|---|
| random | 107 [60-170] | 0 of 20 | 68.5 [40-140] | 0 of 20 | 0.897 | 0.944 | 0.962 | 0.969 |
| uncertainty | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |
| margin | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |
| diverse | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |

**blobs-duplicated**: 2,000 pool points (400 distinct), 1,000 test points; a student trained on the whole pool scores 0.966 test accuracy (95% of it: 0.918).

| Strategy | Labels to 0.95 accuracy, mean [min-max] | Seeds not reaching 0.95 by 400 | Labels to 0.918 (95% of full pool), mean [min-max] | Seeds not reaching it | Accuracy at 50 | Accuracy at 100 | Accuracy at 200 | Accuracy at 400 |
|---|---|---|---|---|---|---|---|---|
| random | 100.5 [40-190] | 0 of 20 | 60.5 [40-90] | 0 of 20 | 0.899 | 0.948 | 0.960 | 0.966 |
| uncertainty | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |
| margin | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |
| diverse | not implemented yet: run `make eval-select` once it is written | | | |  |  |  |  |
<!-- END generated:select -->

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
| Parity with Claude on AG News and Banking77: student vs `claude-haiku-5-5` kappa with CIs, the teacher's self-agreement ceiling (200 items, paraphrased prompt) and teacher vs gold. Labels are written to `eval/labels/` so later runs replay them offline | `make eval-classify` (with `ANTHROPIC_API_KEY` set; `TEACHER=auto` then calls Claude once and replays afterwards) | key; about 1.5 h CPU for the two students | about $0.35 with Haiku 5.5 effort `low` (5,400 labels at ~180 input tokens for AG News, ~540 for Banking77 with its 77 label names, ~50 output tokens each); under $1 if thinking doubles the output |
| A third training seed on both datasets (seeds 0 and 1 are in the table above) | `uv run --group eval python -m eval.classify ag_news banking77 --teacher gold --seed 2`, then `make results` | about 1.5 h CPU (2 ModernBERT-base runs) | $0 |

The toy task is synthetic and only exercises the pipeline; real parity numbers come from the
M2 datasets.
