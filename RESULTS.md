# Results

Every number here was produced by a command in CONTRIBUTING.md and its raw output is committed
under `eval/results/`. Anything not yet run is listed under "To measure" with its command and a
cost estimate. No evaluation so far has called Claude: there is no API key in the build
environment yet. Most runs use each dataset's gold labels as the teacher (model id `gold`); they
measure the training loop, the student and the statistics on real data, not parity with Claude,
which is listed under "To measure". The M8 runs (model id `qwen2.5:3b`) use a real but fallible
teacher, a 3B open model served by Ollama on the same CPU, whose labels are committed under
`eval/labels/` and replay for free.

## Spend

What each committed run cost in teacher dollars and wall time, from the `meta.json` of every
directory under `eval/results/`.

<!-- BEGIN generated:ledger -->
| Run | Eval | Teacher | Command | Commit | Wall time | Teacher $ |
|---|---|---|---|---|---|---|
| `bench-banking77-gold` | bench | gold | `python -m eval.bench --teacher gold` | `d0cfa4c` | 19m 27s | $0.00 |
| `calibration-ag_news-gold` | eval-calibration | gold | `python -m eval.calibration ag_news --logits eval/results/calibration-ag_news-gold/logits.npz` | `52dd15a` | 1m 33s | $0.00 |
| `calibration-ag_news-qwen2.5-3b` | eval-calibration | qwen2.5:3b | `python -m eval.calibration ag_news --logits eval/results/calibration-ag_news-qwen2.5-3b/logits.npz` | `c388f6d` | 1m 14s | $0.00 |
| `calibration-banking77-gold` | eval-calibration | gold | `python -m eval.calibration banking77 --logits eval/results/calibration-banking77-gold/logits.npz` | `52dd15a` | 0m 57s | $0.00 |
| `calibration-banking77-qwen2.5-3b` | eval-calibration | qwen2.5:3b | `python -m eval.calibration banking77 --logits eval/results/calibration-banking77-qwen2.5-3b/logits.npz` | `c388f6d` | 0m 46s | $0.00 |
| `cascade-extract-mit_restaurant-gold` | eval-extract-cascade | gold | `python -m eval.extract --predictions eval/results/extract-mit_restaurant-gold/report.json` | `fe621a5` | 0m 20s | $0.00 |
| `classify-ag_news-gold` | eval-classify | gold | `python -m eval.classify ag_news banking77 --teacher gold` | `74cca5e` | 36m 10s | $0.00 |
| `classify-ag_news-gold-seed1` | eval-classify | gold | `python -m eval.classify ag_news --teacher gold --seed 1` | `1c900f7` | 47m 59s | $0.00 |
| `classify-ag_news-qwen2.5-3b` | eval-classify | qwen2.5:3b | `python -m eval.classify ag_news banking77 --teacher ollama` | `e25575f` | 41m 06s | $0.00 |
| `classify-banking77-gold` | eval-classify | gold | `python -m eval.classify banking77 --teacher gold` | `d4f4016` | 56m 31s | $0.00 |
| `classify-banking77-gold-seed1` | eval-classify | gold | `python -m eval.classify banking77 --teacher gold --seed 1` | `7c205b0` | 45m 20s | $0.00 |
| `classify-banking77-qwen2.5-3b` | eval-classify | qwen2.5:3b | `python -m eval.classify ag_news banking77 --teacher ollama` | `e25575f` | 1h 07m 55s | $0.00 |
| `coverage-simulation` | eval-coverage | n/a | `python -m eval.coverage --workers 2` | `e404ed4` | 0m 34s | $0.00 |
| `export-ag_news-gold` | eval-export | gold | `python -m eval.export --teacher gold` | `e65c998` (uncommitted changes) | 47m 30s | $0.00 |
| `export-banking77-gold` | eval-export | gold | `python -m eval.export --teacher gold` | `e65c998` (uncommitted changes) | 1h 03m 02s | $0.00 |
| `extract-mit_restaurant-gold` | eval-extract | gold | `python -m eval.extract mit_restaurant --teacher gold` | `7dace8b` (uncommitted changes) | 15m 01s | $0.00 |
| `flag-ag_news-qwen2.5-3b` | eval-flag | qwen2.5:3b | `python -m eval.flag` | `1dfc2df` | 0m 04s | $0.00 |
| `flag-banking77-qwen2.5-3b` | eval-flag | qwen2.5:3b | `python -m eval.flag` | `1dfc2df` | 0m 02s | $0.00 |
| `frontier-ag_news-gold` | eval-frontier | gold | `python -m eval.frontier --teacher gold` | `f929d74` | 1h 27m 18s | $0.00 |
| `frontier-banking77-gold` | eval-frontier | gold | `python -m eval.frontier banking77 --teacher gold` | `278462c` | 27m 58s | $0.00 |
| `generate-wikisql-gold` | eval-generate | gold | `python -m eval.generate wikisql --teacher gold` | `795cc7d` | 1h 43m 21s | $0.00 |
| `noise-ag_news-qwen2.5-3b` | eval-noise | qwen2.5:3b | `python -m eval.noise ` | `e25575f` | 1h 17m 14s | $0.00 |
| `noise-banking77-qwen2.5-3b` | eval-noise | qwen2.5:3b | `python -m eval.noise ` | `e25575f` | 1h 11m 30s | $0.00 |
| `select-synthetic` | eval-select | n/a | `python -m eval.select_study` | `c1a7a44` | 0m 25s | $0.00 |
| `shift-banking77-gold` | eval-shift | gold | `python -m eval.shift --teacher gold` | `773e679` | 10m 59s | $0.00 |
| `spike-onnx-export` | spike-onnx | n/a | `uv run python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export` | `94389b5` | 5m 10s | $0.00 |
| **Total (26 runs)** | | | | | 13h 49m 21s | $0.00 |

Dollars are the sum of the `usage` of each teacher call, priced in `anvil/teacher/pricing.py`; a run whose teacher is `gold` makes no calls and costs $0. Wall time is the invocation's `wall_seconds`; for the frontier runs that is the figure the frontier section below treats as the run's time, and the training + selection time it also lists is not added again. A run that resumed an earlier invocation counts only its own. Totals skip runs that did not record a field.
<!-- END generated:ledger -->

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
| ag_news | qwen2.5:3b | ModernBERT-base | 2000 | 477 | 0.847 [0.808, 0.884] | 0.889 | 0.871 | 0.319 | 0.732 [0.684, 0.781] | +0.115 [0.067, 0.166] | 0.937 [0.895, 0.978] | $0.00 | 41 min |
| banking77 | gold | ModernBERT-base | 2000 | 496 | 0.842 [0.807, 0.873] | 0.845 | 0.814 | 0.024 | 0.662 [0.621, 0.702] | +0.180 [0.141, 0.223] | n/a | $0.00 | 57 min |
| banking77 | qwen2.5:3b | ModernBERT-base | 2000 | 496 | 0.687 [0.641, 0.726] | 0.694 | 0.614 | 0.036 | 0.499 [0.454, 0.544] | +0.188 [0.145, 0.229] | 0.831 [0.779, 0.882] | $0.00 | 68 min |

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

#### ag_news, teacher `qwen2.5:3b`, student `ModernBERT-base`

- Raw output: `eval/results/classify-ag_news-qwen2.5-3b/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.classify ag_news banking77 --teacher ollama` at `e25575f`; seed 0, 1000 bootstrap resamples; 4 CPUs; wall 2466 s; spent $0.0000
- Data: fancyzhx/ag_news@eb185aa (AG's corpus, used under its academic terms (http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)), 2500 inputs sampled (seed 0, sha256 3cf708037e01)
- Training: 1806 examples and 194 held back to pick the epoch, out of 2000 teacher labels; 3 epochs, 339 steps, best epoch 1, 2319 s
- Near-duplicates: 3 of 477 held-out inputs share at least 80% of their words with a labelled pool input (identical texts never straddle the split; these are the ones a hash cannot catch)
- Teacher: `qwen2.5:3b` through Ollama, weights sha256 `357c53fb659c` (pinned in the eval config); 2477 labels took 1.81 h of server time (median 2.58 s per label), the ceiling's second pass 9 min more, $0
- Against gold: teacher 0.719 [0.672, 0.764], student 0.696 [0.645, 0.743]
- Student minus teacher, agreement with gold on the same 477 inputs (paired bootstrap): kappa -0.023 [-0.059, 0.013], accuracy -0.017 [-0.044, 0.010]
- On the 101 held-out inputs the teacher got wrong, the student gave the teacher's label on 74 (73%), the gold label on 18 (18%) and another on 9 (9%); on the 376 it got right, the student is wrong on 26 (7%)
- The teacher's top confusions (gold -> teacher label: how often the teacher / the student makes it): `sci_tech` -> `business` 59 / 55, `business` -> `world` 12 / 21, `sci_tech` -> `world` 9 / 16, `world` -> `sports` 6 / 5, `business` -> `sci_tech` 4 / 2, `sci_tech` -> `sports` 4 / 3
- On the ceiling's 200 inputs: teacher vs itself 0.937 [0.895, 0.978], student vs teacher 0.861 [0.799, 0.916]; the student agrees with the teacher on 90.6% of 191 inputs where the teacher repeated its label and on 77.8% of 9 where it did not
- Weakest classes with at least 5 held-out items (F1 vs teacher, support): `sci_tech` 0.75 (59), `business` 0.88 (152), `world` 0.89 (138), `sports` 0.96 (128)
- Top confusions (teacher -> student): `business` -> `world` 15, `sci_tech` -> `business` 8, `business` -> `sci_tech` 7, `sci_tech` -> `world` 7, `sports` -> `world` 5
- Agreement by confidence: 0.31-0.58: 0.52 (n=48), 0.58-0.77: 0.71 (n=48), 0.77-0.90: 0.79 (n=48), 0.90-0.97: 0.92 (n=48), 0.97-0.99: 0.98 (n=48), 0.99-1.00: 0.98 (n=48), 1.00-1.00: 1.00 (n=48), 1.00-1.00: 1.00 (n=47), 1.00-1.00: 1.00 (n=47), 1.00-1.00: 1.00 (n=47)
- Agreement by length (words): 14-31: 0.87 (n=120), 32-37: 0.86 (n=119), 37-43: 0.91 (n=119), 43-107: 0.92 (n=119)

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

#### banking77, teacher `qwen2.5:3b`, student `ModernBERT-base`

- Raw output: `eval/results/classify-banking77-qwen2.5-3b/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.classify ag_news banking77 --teacher ollama` at `e25575f`; seed 0, 1000 bootstrap resamples; 4 CPUs; wall 4075 s; spent $0.0000
- Data: legacy-datasets/banking77@f541215 (CC-BY-4.0 (Casanueva et al., 2020, PolyAI)), 2500 inputs sampled (seed 0, sha256 2aeaf7290b43)
- Training: 1809 examples and 191 held back to pick the epoch, out of 2000 teacher labels; 9 epochs, 1026 steps, best epoch 7, 3990 s
- Near-duplicates: 17 of 496 held-out inputs share at least 80% of their words with a labelled pool input (identical texts never straddle the split; these are the ones a hash cannot catch)
- Teacher: `qwen2.5:3b` through Ollama, weights sha256 `357c53fb659c` (pinned in the eval config); 2496 labels took 1.26 h of server time (median 1.70 s per label), the ceiling's second pass 6 min more, $0
- Against gold: teacher 0.544 [0.501, 0.586], student 0.540 [0.497, 0.586]
- Student minus teacher, agreement with gold on the same 496 inputs (paired bootstrap): kappa -0.004 [-0.037, 0.030], accuracy -0.004 [-0.036, 0.030]
- On the 223 held-out inputs the teacher got wrong, the student gave the teacher's label on 112 (50%), the gold label on 39 (17%) and another on 72 (32%); on the 273 it got right, the student is wrong on 41 (15%)
- The teacher's top confusions (gold -> teacher label: how often the teacher / the student makes it): `card_payment_wrong_exchange_rate` -> `wrong_exchange_rate_for_cash_withdrawal` 9 / 10, `card_payment_fee_charged` -> `extra_charge_on_statement` 6 / 3, `get_physical_card` -> `passcode_forgotten` 6 / 4, `order_physical_card` -> `get_physical_card` 6 / 6, `balance_not_updated_after_bank_transfer` -> `transfer_not_received_by_recipient` 5 / 4, `card_arrival` -> `card_delivery_estimate` 5 / 5
- On the ceiling's 200 inputs: teacher vs itself 0.831 [0.779, 0.882], student vs teacher 0.667 [0.600, 0.733]; the student agrees with the teacher on 74.9% of 167 inputs where the teacher repeated its label and on 30.3% of 33 where it did not
- Weakest classes with at least 5 held-out items (F1 vs teacher, support): `card_acceptance` 0.22 (5), `refund_not_showing_up` 0.36 (7), `supported_cards_and_currencies` 0.38 (9), `reverted_card_payment` 0.40 (5), `pending_transfer` 0.45 (10)
- Top confusions (teacher -> student): `card_not_working` -> `card_delivery_estimate` 3, `extra_charge_on_statement` -> `top_up_by_card_charge` 3, `get_disposable_virtual_card` -> `get_physical_card` 3, `apple_pay_or_google_pay` -> `top_up_by_card_charge` 2, `card_arrival` -> `card_about_to_expire` 2
- Agreement by confidence: 0.16-0.49: 0.24 (n=50), 0.49-0.69: 0.48 (n=50), 0.70-0.80: 0.62 (n=50), 0.81-0.91: 0.50 (n=50), 0.91-0.97: 0.68 (n=50), 0.97-0.99: 0.84 (n=50), 0.99-1.00: 0.78 (n=49), 1.00-1.00: 0.92 (n=49), 1.00-1.00: 0.90 (n=49), 1.00-1.00: 1.00 (n=49)
- Agreement by length (words): 4-7: 0.72 (n=124), 7-10: 0.70 (n=124), 10-13: 0.71 (n=124), 13-53: 0.65 (n=124)
<!-- END generated:classify -->

Training-seed spread (only the student seed changes: weight init, data order, validation slice):

<!-- BEGIN generated:classify-seeds -->
| Dataset | Teacher | Student | Seeds | Kappa per seed | Mean | Range (max - min) | Mean 95% CI width |
|---|---|---|---|---|---|---|---|
| ag_news | gold | ModernBERT-base | 2 | 0: 0.844, 1: 0.896 | 0.870 | 0.053 | 0.073 |
| banking77 | gold | ModernBERT-base | 2 | 0: 0.842, 1: 0.842 | 0.842 | 0.000 | 0.067 |
<!-- END generated:classify-seeds -->

### The first fallible teacher: a local model on AG News and Banking77 (M8)

The same harness, sample, split and ModernBERT-base student as the table above, with the
teacher swapped for `qwen2.5:3b` served by Ollama on the same CPU (`make eval-classify
TEACHER=ollama`). Unlike gold, this teacher can be wrong, so each row puts the student's
agreement with it next to the teacher's own agreement with gold and with itself.

<!-- BEGIN generated:classify-teacher -->
| Dataset | Teacher | Student | Teacher vs gold | Teacher vs itself (ceiling) | Student vs teacher | Student vs teacher, ceiling inputs | Student vs gold | Student minus teacher vs gold [95% paired CI] | Same student on gold labels |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | qwen2.5:3b | ModernBERT-base | 0.719 [0.672, 0.764] | 0.937 [0.895, 0.978] | 0.847 [0.808, 0.884] | 0.861 [0.799, 0.916] | 0.696 [0.645, 0.743] | -0.023 [-0.059, 0.013] | 0.844 [0.805, 0.882] |
| banking77 | qwen2.5:3b | ModernBERT-base | 0.544 [0.501, 0.586] | 0.831 [0.779, 0.882] | 0.687 [0.641, 0.726] | 0.667 [0.600, 0.733] | 0.540 [0.497, 0.586] | -0.004 [-0.037, 0.030] | 0.842 [0.807, 0.873] |

All kappas are on the same held-out inputs (the ceiling and its column on the first 200 of them, which the teacher labelled under two prompts). "Same student on gold labels" is the gold-teacher run of the table above: the same sample, split and student trained on clean labels.
<!-- END generated:classify-teacher -->

Findings (each per-run block under the M2 table above holds the full detail):

- **The student did not beat its teacher.** The question was whether training averages a
  fallible teacher's mistakes away, as it would random label noise. It did not on either
  dataset. Student minus teacher, kappa with gold on the same inputs (paired bootstrap):
  -0.023 [-0.059, 0.013] on AG News and -0.004 [-0.037, 0.030] on Banking77. The plan
  expected at least +0.05 on AG News; that is refuted.
- **It copies the teacher's mistakes.** Of the teacher's held-out errors, the student repeats
  74 of 101 (73%) on AG News and 112 of 223 (50%) on Banking77; it recovers the gold label on
  18% and 17%. One confusion dominates AG News: the teacher labels 59 `sci_tech` stories
  `business` (stories about technology companies), and the student says `business` on 47 of
  those 59 (55 `sci_tech` stories in all).
- **Parity stays under the ceiling, and far from the truth.** Student vs teacher on the
  ceiling's 200 inputs (point estimates; on AG News the two intervals overlap): 0.861 against the teacher's self-agreement of 0.937 on AG News, 0.667
  against 0.831 on Banking77. The same ModernBERT trained on gold reaches 0.844 and 0.842
  against gold. Trained on this teacher it reaches 0.696 and 0.540: the teacher's error is
  inherited almost whole, and a parity number alone (0.847 on AG News) would hide that.
- **Where the teacher repeats itself, the student follows it.** On Banking77 the student
  agrees with the teacher on 74.9% of the 167 ceiling inputs where the teacher gave the same
  label under both prompts, and on 30.3% of the 33 where it did not. Disagreement is far more
  likely where the teacher is unstable (70% against 25%), but those inputs are few: 42 of the
  65 disagreements on the ceiling inputs are on inputs where the teacher repeated itself.
- The control below tests whether the inherited errors come from the confusions themselves
  or from random noise.

### Noise control: what kind of noise is the local teacher's? (M8)

`make eval-noise`: the frontier's MiniLM-L6 student trained on the same 2,000 pool inputs
four ways per seed (gold; the teacher's labels; gold with as many labels wrong as the teacher
got wrong, moved uniformly; gold with the teacher's own confusion matrix dealt to random
inputs of each class), each scored against gold on the same held-out inputs.

<!-- BEGIN generated:noise -->
#### ag_news, teacher `qwen2.5:3b`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/noise-ag_news-qwen2.5-3b/` (`report.json` with every run's held-out predictions, `meta.json`)
- Command: `python -m eval.noise ` at `e25575f`; seeds 0, 1, 2; 4 CPUs; wall 1h 17m 14s; $0
- Training labels: 2000 pool inputs; the teacher got 476 (24%) wrong (kappa with gold 0.683 [0.660, 0.706]), so each corrupted set has exactly that many wrong labels
- Scored on 477 held-out inputs, where the teacher's own kappa with gold is 0.719 [0.672, 0.764]

| Trained on | Student vs gold kappa, mean (range) | Student vs teacher kappa | Teacher's held-out errors the student repeats |
|---|---|---|---|
| gold labels | 0.863 (0.857-0.869) | 0.722 (0.715-0.730) | 22 (19-25)% |
| the teacher's labels | 0.676 (0.655-0.688) | 0.839 (0.820-0.858) | 77 (75-79)% |
| gold with uniform noise | 0.811 (0.785-0.826) | 0.685 (0.659-0.703) | 23 (20-25)% |
| gold with the teacher's confusions | 0.666 (0.644-0.678) | 0.700 (0.674-0.718) | 52 (50-54)% |

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold with uniform noise (paired bootstrap, per seed): seed 0 -0.130 [-0.190, -0.070], seed 1 -0.141 [-0.192, -0.087], seed 2 -0.133 [-0.182, -0.082]

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold with the teacher's confusions (paired bootstrap, per seed): seed 0 -0.023 [-0.072, 0.024], seed 1 +0.010 [-0.034, 0.057], seed 2 +0.044 [-0.006, 0.102]

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold labels (paired bootstrap, per seed): seed 0 -0.208 [-0.263, -0.155], seed 1 -0.183 [-0.232, -0.134], seed 2 -0.169 [-0.216, -0.121]

#### banking77, teacher `qwen2.5:3b`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/noise-banking77-qwen2.5-3b/` (`report.json` with every run's held-out predictions, `meta.json`)
- Command: `python -m eval.noise ` at `e25575f`; seeds 0, 1, 2; 4 CPUs; wall 1h 11m 30s; $0
- Training labels: 2000 pool inputs; the teacher got 868 (43%) wrong (kappa with gold 0.560 [0.539, 0.582]), so each corrupted set has exactly that many wrong labels
- Scored on 496 held-out inputs, where the teacher's own kappa with gold is 0.544 [0.501, 0.586]

| Trained on | Student vs gold kappa, mean (range) | Student vs teacher kappa | Teacher's held-out errors the student repeats |
|---|---|---|---|
| gold labels | 0.842 (0.830-0.848) | 0.498 (0.485-0.505) | 3 (3-4)% |
| the teacher's labels | 0.503 (0.501-0.509) | 0.684 (0.680-0.687) | 55 (54-55)% |
| gold with uniform noise | 0.567 (0.554-0.576) | 0.388 (0.371-0.406) | 9 (7-11)% |
| gold with the teacher's confusions | 0.476 (0.468-0.484) | 0.561 (0.540-0.579) | 39 (35-44)% |

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold with uniform noise (paired bootstrap, per seed): seed 0 -0.070 [-0.125, -0.015], seed 1 -0.076 [-0.129, -0.022], seed 2 -0.045 [-0.101, 0.009]

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold with the teacher's confusions (paired bootstrap, per seed): seed 0 +0.016 [-0.015, 0.049], seed 1 +0.033 [-0.002, 0.072], seed 2 +0.033 [-0.000, 0.067]

- Kappa with gold, the student trained on the teacher's labels minus the one trained on gold labels (paired bootstrap, per seed): seed 0 -0.346 [-0.391, -0.299], seed 1 -0.348 [-0.395, -0.299], seed 2 -0.321 [-0.370, -0.274]
<!-- END generated:noise -->

Findings:

- **The local teacher's noise costs what its confusions cost, not what random noise costs.**
  At the teacher's exact error count, uniform noise costs the student little on AG News
  (0.811 against 0.863 for gold, 24% of labels wrong) and 0.28 of kappa on Banking77 (0.567
  against 0.842, 43% wrong). The teacher's own labels cost more: 0.676 and 0.503. Teacher
  minus uniform is negative on all six seeds, and its paired interval excludes 0 on five of
  them (-0.130 to -0.141 on AG News).
- **On kappa with gold, the teacher's labels and its confusion matrix cannot be told apart.**
  The same confusions dealt to random inputs of each class score 0.666 and 0.476, within a
  few hundredths of the real teacher's labels. Teacher minus pair ranges from -0.023 to
  +0.044, and every paired interval includes 0 (Banking77's seed 2 only just). That is
  consistent with the confusion matrix carrying the cost, not proof of it. Keeping each
  class's error count also means the pair condition puts many of its errors where the teacher
  put its own, which pulls the two together.
- **But part of the teacher's noise is tied to the inputs.** The student trained on the
  teacher's labels repeats 77% (AG News) and 55% (Banking77) of the teacher's held-out
  mistakes, against 52% and 39% for the one trained on the same confusions placed at random,
  with no overlap between the seed ranges. A class-level confusion cannot teach a student
  which unseen inputs to get wrong; the teacher's own labels do, because it mislabels the same
  kind of input the same way (AG News's stories about technology companies filed as
  `business`). The ModernBERT student above copying 73% of the errors is consistent with this,
  though the control ran on MiniLM.
- **Some of the teacher's errors are not clear-cut.** On AG News, the student trained on
  *gold* still makes the teacher's mistake on 22% of the teacher's held-out errors (3% on
  Banking77). Those inputs sit near a class boundary, where the teacher and the student
  trained on gold disagree with the gold label in the same way.
- The control's wall times include contention: it shared the 4 cores with the ModernBERT
  training of the table above.

### Can the student flag its teacher's mistakes? (M13)

`make eval-flag` (`python -m eval.flag`): the frontier's MiniLM-L6 student trained out of fold
(3 folds, one random split shared by every condition) on each M8 dataset's 2,000 pool labels,
three ways: the teacher's labels, and the `eval-noise` seed-0 corruptions of gold at the
teacher's error count (uniform, and the teacher's confusions dealt to random inputs of each
class). Each label is scored by the probability the student gives it without having trained on
it (the self-confidence of confident learning, Northcutt et al. 2021); low means suspect. The
flags are scored against the labels that really are wrong (given label differs from gold), and
the student is retrained on the 80% not flagged and scored against gold on the held-out split,
paired with the committed seed-0 `eval-noise` student trained on the same labels unfiltered.
For the teacher's labels two more students make that comparison readable: one on a random 80%
(is a change the filtering, or just less data?) and the unfiltered one retrained here (does the
committed reference reproduce?). `python -m eval.flag --oof eval/results/flag-<dataset>-qwen2.5-3b/oof.npz`
re-runs the flag analysis from the committed probabilities without a model.

<!-- BEGIN generated:flag -->
#### ag_news, teacher `qwen2.5:3b`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/flag-ag_news-qwen2.5-3b/` (`report.json`, `meta.json`, `oof.npz` with every out-of-fold probability)
- Command: `python -m eval.flag` at `1dfc2df`; 2000 pool labels, 3 folds (seed 0); 0 students trained in this invocation; 4 CPUs; wall 0m 04s; $0

| Training labels | Wrong | Out-of-fold AUROC for wrong labels | Lowest 20.0% flagged: precision | Recall (of all wrong) | Recall at precision >= 0.7 (share flagged) | Flagged wrong labels where the student's top class is gold |
|---|---|---|---|---|---|---|
| the teacher's labels | 476 (24%) | 0.711 [0.686, 0.738] | 0.490 | 0.412 (at most 0.840) | 0.015 (0.4%) | 52.6% |
| gold with uniform noise | 476 (24%) | 0.898 [0.883, 0.913] | 0.693 | 0.582 (at most 0.840) | 0.471 (16.0%) | 88.1% |
| gold with the teacher's confusions | 476 (24%) | 0.724 [0.696, 0.753] | 0.480 | 0.403 (at most 0.840) | 0.000 (0.0%) | 64.1% |

Retrained on what was not flagged, scored against gold on 477 held-out inputs (seed 0; the reference is the committed seed-0 `eval-noise` student trained on the same labels unfiltered):

| Training labels | Trained on | Wrong among them | Kappa with gold | Minus the unfiltered reference (paired) |
|---|---|---|---|---|
| the teacher's labels | everything (committed reference) | 476 | 0.655 [0.604, 0.706] | |
| the teacher's labels | the 80.0% not flagged (1600) | 280 | 0.672 [0.621, 0.721] | +0.017 [-0.019, 0.053] |
| the teacher's labels | a random share of the same size (1600) | 384 | 0.688 [0.638, 0.735] | +0.033 [0.006, 0.061] |
| the teacher's labels | everything, retrained here (2000) | 476 | 0.697 [0.649, 0.742] | +0.042 [0.017, 0.068] |
| gold with uniform noise | everything (committed reference) | 476 | 0.785 [0.739, 0.826] | |
| gold with uniform noise | the 80.0% not flagged (1600) | 199 | 0.838 [0.796, 0.876] | +0.053 [0.015, 0.096] |
| gold with the teacher's confusions | everything (committed reference) | 476 | 0.678 [0.630, 0.727] | |
| gold with the teacher's confusions | the 80.0% not flagged (1600) | 284 | 0.640 [0.592, 0.689] | -0.038 [-0.074, -0.004] |

- The teacher's labels: kappa with gold, filtered minus random drop (paired, both trained here): -0.016 [-0.046, 0.014]
- The teacher's labels: kappa with gold, filtered minus unfiltered (paired, both trained here): -0.025 [-0.060, 0.009]
- The unfiltered student retrained here predicts the same label as the committed reference on 450 of 477 held-out inputs

#### banking77, teacher `qwen2.5:3b`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/flag-banking77-qwen2.5-3b/` (`report.json`, `meta.json`, `oof.npz` with every out-of-fold probability)
- Command: `python -m eval.flag` at `1dfc2df`; 2000 pool labels, 3 folds (seed 0); 0 students trained in this invocation; 4 CPUs; wall 0m 02s; $0

| Training labels | Wrong | Out-of-fold AUROC for wrong labels | Lowest 20.0% flagged: precision | Recall (of all wrong) | Recall at precision >= 0.7 (share flagged) | Flagged wrong labels where the student's top class is gold |
|---|---|---|---|---|---|---|
| the teacher's labels | 868 (43%) | 0.682 [0.659, 0.706] | 0.765 | 0.353 (at most 0.461) | 0.475 (29.4%) | 35.0% |
| gold with uniform noise | 868 (43%) | 0.938 [0.927, 0.947] | 0.927 | 0.427 (at most 0.461) | 0.977 (60.6%) | 56.1% |
| gold with the teacher's confusions | 868 (43%) | 0.729 [0.705, 0.754] | 0.748 | 0.344 (at most 0.461) | 0.499 (30.9%) | 34.1% |

Retrained on what was not flagged, scored against gold on 496 held-out inputs (seed 0; the reference is the committed seed-0 `eval-noise` student trained on the same labels unfiltered):

| Training labels | Trained on | Wrong among them | Kappa with gold | Minus the unfiltered reference (paired) |
|---|---|---|---|---|
| the teacher's labels | everything (committed reference) | 868 | 0.501 [0.457, 0.547] | |
| the teacher's labels | the 80.0% not flagged (1600) | 562 | 0.542 [0.501, 0.586] | +0.041 [0.012, 0.070] |
| the teacher's labels | a random share of the same size (1600) | 692 | 0.488 [0.445, 0.533] | -0.012 [-0.041, 0.014] |
| the teacher's labels | everything, retrained here (2000) | 868 | 0.509 [0.466, 0.554] | +0.008 [-0.016, 0.031] |
| gold with uniform noise | everything (committed reference) | 868 | 0.570 [0.523, 0.614] | |
| gold with uniform noise | the 80.0% not flagged (1600) | 497 | 0.615 [0.568, 0.654] | +0.045 [0.004, 0.086] |
| gold with the teacher's confusions | everything (committed reference) | 868 | 0.484 [0.439, 0.527] | |
| gold with the teacher's confusions | the 80.0% not flagged (1600) | 569 | 0.486 [0.444, 0.528] | +0.002 [-0.029, 0.035] |

- The teacher's labels: kappa with gold, filtered minus random drop (paired, both trained here): +0.053 [0.025, 0.084]
- The teacher's labels: kappa with gold, filtered minus unfiltered (paired, both trained here): +0.033 [0.002, 0.065]
- The unfiltered student retrained here predicts the same label as the committed reference on 426 of 496 held-out inputs
<!-- END generated:flag -->

Provenance: the 28 trainings ran in two earlier invocations of this harness, `python -m
eval.flag` for Banking77 (1h 33m wall, started at `a00f163` with `eval/flag.py` not yet
committed) and `python -m eval.flag ag_news` at `a5f11b8` (2h 00m wall, sharing the 4 vCPUs
with the first, since torch used about two of them). Every prediction is cached by its inputs,
so the committed reports are a re-run at a clean commit that trained nothing. Its reports are
identical to the two originals, except for the filtered-minus-unfiltered line, which was added
afterwards. $0.

Findings, against the plan's hypotheses:

- **Uniform noise is easy to flag; the teacher's noise is not (hypothesis held on Banking77,
  not on AG News).** At the same error count, the out-of-fold self-confidence finds uniformly
  placed wrong labels with AUROC 0.938 (Banking77) and 0.898 (AG News), and the teacher's own
  wrong labels with 0.682 and 0.711. Recall at precision 0.7 is 0.977 under uniform noise on
  Banking77, above the predicted 0.6, but 0.471 on AG News. With 24% of labels wrong there,
  precision 0.7 is a stricter bar than with 43%, so the plan's "both datasets" did not hold.
- **The hypothesis that input-tied errors are what defeats the flags was refuted.** The plan
  predicted the teacher's labels would flag at least 0.15 worse in recall than the same
  confusions dealt at random. They flag about the same: recall at precision 0.7 is 0.475
  against 0.499 on Banking77 (the predicted ceiling was 0.35), and 0.015 against 0.000 on
  AG News. The AUROCs are 0.682 against 0.729 and 0.711 against 0.724. The teacher's labels
  are a little harder to flag on Banking77, but nowhere near the gap to uniform noise. What
  defeats confident learning here is the class-level structure. When one class is
  systematically labelled as another, the other folds carry the same confusion, the student
  learns it, and a wrong label looks like a normal one. Whether the error also follows the
  input adds little on top. M8's copy rates (77% against 52%) were about which held-out
  inputs a student gets wrong; they don't make training labels harder to flag. The plan's
  aside that AG News might do better was also refuted at the precision bar: there the
  teacher's flags never reach precision 0.7 beyond the first 0.4% of labels, though its
  AUROC (0.711) is a little higher than Banking77's.
- **The flags are still informative where the errors are dense.** Flagging the lowest 20% of
  Banking77's teacher labels is 76.5% precise against a 43% base rate, and catches 306 of the
  868 wrong labels (the cap is 400). On AG News it is 49.0% precise against 24%. On the caught
  errors, the student's own top class is gold 35% (Banking77) and 53% (AG News) of the time,
  so relabelling with the student would fix about one caught error in three or two, and break
  the rest.
- **Dropping the flagged 20% helps on Banking77 and not on AG News (the hypothesis, under
  +0.03, held on AG News and was borderline on Banking77).** Read the paired lines, which
  compare students from the same invocation. On Banking77, filtered minus unfiltered is
  +0.033 [0.002, 0.065] and filtered minus a random 20% drop is +0.053 [0.025, 0.084]. On AG
  News they are -0.025 [-0.060, 0.009] and -0.016 [-0.046, 0.014]. The confusion-pair control
  agrees: +0.002 on Banking77 and -0.038 on AG News against its committed reference. On AG
  News the flag removes 196 of the 476 wrong labels but also 204 right ones, and that does not
  beat the data it costs.
- **A single-seed retrain is noisier than its interval says.** The unfiltered student
  retrained here, with the same seed and labels as the committed `eval-noise` student,
  predicts the same label on 426 of 496 (Banking77) and 450 of 477 (AG News) held-out inputs.
  On AG News it scores 0.697 against the reference's 0.655 (+0.042 [0.017, 0.068]), outside
  the paired interval, which counts only which test inputs were drawn and not training
  randomness. M9 already found students that do not reproduce across CPUs. This is the same
  machine type, and the likely cause (not tested) is thread scheduling while the two flag
  processes shared the CPU. So the "minus the unfiltered reference" column is confounded by
  that drift, the same-invocation lines above are the comparisons to read, and differences of
  a few hundredths on AG News sit inside training noise (`eval-noise`'s teacher seeds span
  0.655-0.688 there, against 0.501-0.509 on Banking77).
- **So, is the student a third option to gold labels or a better teacher?** Partly. It can
  rank its teacher's mistakes better than chance (AUROC about 0.7), and on a 77-class task
  where 43% of labels are wrong, dropping its most suspect 20% buys about 0.03-0.05 kappa
  with gold. But it cannot see the errors that matter most, the teacher's systematic
  confusions, and on AG News it buys nothing. It is a triage list for a human reviewer, not a
  replacement for gold.

### Label-count frontier: AG News and Banking77 (M3)

- Command: `make eval-frontier TEACHER=gold` (`python -m eval.frontier --teacher gold`, then
  `python -m eval.render`, which writes the block below from `eval/results/frontier-*/`).
- Method: the `make eval-classify` sample of each dataset (2,500 inputs; the same prompt and
  label files, so a Claude run labels once for both evals). Every implemented strategy runs
  one `anvil loop` per seed (0, 1, 2) over budgets of 100, 200, 400, 800 and 1,600 pool labels.
  Round 0 is a random draw that depends only on the seed, so all strategies of a seed share
  their first student. Every round trains a fresh MiniLM-L6 student
  (`sentence-transformers/all-MiniLM-L6-v2`, 64 tokens, batch 16, learning rate 5e-4, at least
  600 optimizer steps, no early stopping) on every label so far and scores it against the
  teacher on the same held-out inputs (kappa with a 95% percentile bootstrap interval,
  1,000 resamples). Cells are the mean kappa over the three seeds with the lowest and highest
  seed in brackets. "Labels to kappa t" reads the grid two ways: the smallest budget whose mean
  kappa reaches t, and the smallest at which every seed's interval lies above t; neither
  interpolates between budgets. The learning rate and step count come from a pilot on
  Banking77 scored on pool inputs outside the training slice, never on this held-out split;
  the pilot's output was not kept, so no number from it is reported. "Wall" is the clock
  time of the invocation named in the command line; "training + selection" sums every
  round's `train_seconds` and `select_seconds` in `frontier.jsonl`, so it covers a run
  finished over several invocations (the Banking77 run resumed an interrupted one).
- With no key in the build environment the teacher is each dataset's gold labels, so these
  are curves of how the student learns from clean labels, not parity with Claude; the
  strategies other than `random` are not written yet. Both are under "To measure".

<!-- BEGIN generated:frontier -->
#### ag_news, teacher `gold`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/frontier-ag_news-gold/` (`frontier.jsonl`, `report.json`, `meta.json`)
- Command: `python -m eval.frontier --teacher gold` at `f929d74`; seeds 0, 1, 2; 4 CPUs; wall 87 min (training + selection 86 min); spent $0.0000
- Data: fancyzhx/ag_news@eb185aa, 2500 inputs sampled (sha256 3cf708037e01): a pool of 2023 to select from, 477 held-out inputs scored every round

| Strategy | 100 labels | 200 labels | 400 labels | 800 labels | 1600 labels | Labels to kappa 0.8 (mean / every seed's CI) | Labels to kappa 0.9 (mean / every seed's CI) | Label cost at 1600 |
|---|---|---|---|---|---|---|---|---|
| random | 0.721 [0.709, 0.733] | 0.795 [0.787, 0.810] | 0.817 [0.807, 0.824] | 0.844 [0.837, 0.851] | 0.859 [0.855, 0.863] | 400 / 1600 | > 1600 / > 1600 | $0.00 |
| uncertainty | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |
| margin | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |
| diverse | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |

Classes seen in the training labels (random, mean over seeds): 4.0 at 100, 4.0 at 200, 4.0 at 400, 4.0 at 800, 4.0 at 1600; training time per round: 312 s, 335 s, 342 s, 340 s, 338 s.

The teacher here is the dataset's gold labels (no API key in this run): the curve measures how the student learns from clean labels, not parity with Claude.

#### banking77, teacher `gold`, student `all-MiniLM-L6-v2`

- Raw output: `eval/results/frontier-banking77-gold/` (`frontier.jsonl`, `report.json`, `meta.json`)
- Command: `python -m eval.frontier banking77 --teacher gold` at `278462c`; seeds 0, 1, 2; 4 CPUs; wall 28 min (training + selection 56 min, so this invocation resumed an earlier one); spent $0.0000
- Data: legacy-datasets/banking77@f541215, 2500 inputs sampled (sha256 2aeaf7290b43): a pool of 2004 to select from, 496 held-out inputs scored every round

| Strategy | 100 labels | 200 labels | 400 labels | 800 labels | 1600 labels | Labels to kappa 0.8 (mean / every seed's CI) | Labels to kappa 0.9 (mean / every seed's CI) | Label cost at 1600 |
|---|---|---|---|---|---|---|---|---|
| random | 0.416 [0.396, 0.433] | 0.532 [0.507, 0.572] | 0.681 [0.671, 0.691] | 0.768 [0.761, 0.775] | 0.820 [0.812, 0.830] | 1600 / > 1600 | > 1600 / > 1600 | $0.00 |
| uncertainty | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |
| margin | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |
| diverse | not implemented yet | not implemented yet | not implemented yet | not implemented yet | not implemented yet | n/a | n/a | n/a |

Classes seen in the training labels (random, mean over seeds): 55.7 at 100, 68.0 at 200, 76.3 at 400, 77.0 at 800, 77.0 at 1600; training time per round: 207 s, 222 s, 218 s, 218 s, 221 s.

The teacher here is the dataset's gold labels (no API key in this run): the curve measures how the student learns from clean labels, not parity with Claude.
<!-- END generated:frontier -->

### Extraction parity: MIT Restaurant (M4)

- Command: `make eval-extract TEACHER=gold` (`python -m eval.extract mit_restaurant --teacher
  gold`, then `python -m eval.render`, which writes the block below from
  `eval/results/extract-*/`).
- Method: a seeded sample of 2,000 distinct sentences from the MIT Restaurant train split
  (T-NER copy at a pinned revision; words joined by single spaces, gold BIO tags turned into
  character spans). The store's hash split puts about a quarter in held-out. The teacher
  labels up to 500 held-out and 1,500 pool sentences as mentions (text, type, claimed start
  offset); every answer is aligned to its sentence and to the student's tokens, and training
  refuses to start if more than 5% of mentions are lost. A fresh `EncoderTagger`
  (ModernBERT-base, BIO head) trains on the pool (10% held back by text hash to pick the epoch
  by entity F1) and is scored against the teacher's held-out answers. Entity F1 is exact match
  on (start, end, type), micro over all entities; token kappa is Cohen's kappa over the BIO tag
  of every whitespace word. Both intervals are 95% percentile bootstraps that resample
  sentences (1,000 resamples), not words or entities. The gazetteer baseline tags every
  occurrence of a surface form the teacher tagged in the training pool (most frequent type,
  longest match first); the student-minus-gazetteer interval is paired on the same sentences.
- The stress table corrupts the held-out gold mentions the ways a language model gets offsets
  wrong and counts the spans plain verification (exact offset or nothing) and the repair
  ladder recover. It is synthetic noise on real sentences, run on every invocation; it
  measures the alignment code, not a teacher. Only the spans a corruption actually changes are
  scored (offset zero leaves a span that starts at 0 alone), and each changed span is matched
  to at most one aligned span: recovered when one equals it, misplaced when one of its type
  overlaps it or carries the same words elsewhere, lost otherwise. MIT Restaurant queries are
  lower-case and carry no punctuation, so the title-case and trailing-full-stop rows are a
  smoke check of those two rules: on a corpus with sentence-final full stops, a mention that
  really ends a sentence ("late.") would match as written and come back one character long,
  which the ladder cannot tell from a correct answer, by design.

Findings so far (from the block below; the teacher is the gold annotation, not Claude):

- Plain verification of offsets is useless against the errors language models make: an off
  by one, a few characters off, a word index or offset 0 leaves 0 of the ~960 changed spans
  usable, and the repair ladder recovers every one of them. Two rules earn the last few: a
  shifted match prefers an occurrence on word boundaries (a two-letter "in" is otherwise found
  inside "dining" and lost to the overlap rule), and two identical mentions take two
  occurrences when the sentence has two (both "pancakes" moved to offset 0 would otherwise
  collapse into one). An earlier run without those rules recovered 99.8-99.9% on the
  word-index and offset-zero rows.
- The tagger reaches entity F1 0.727 [0.696, 0.757] and token kappa 0.831 [0.810, 0.850]
  from 1,442 clean labels, 0.254 [0.222, 0.288] above a gazetteer of the same training
  spans. The gap is widest on open-ended types (`Restaurant_Name` 0.82 vs 0.35, `Dish` 0.68 vs
  0.38) and narrowest on small closed vocabularies (`Price` 0.85 vs 0.78, `Cuisine` 0.83 vs
  0.72), where memorising surface forms nearly suffices.
- Boundaries, not types, are where the student fails: of the 237 teacher entities it does not
  match exactly, 127 have the right type and an overlapping span with the wrong extent, 43 the
  right extent and the wrong type, 25 both wrong, and 42 are missed outright. `Hours` (0.50)
  and `Amenity` (0.55) are the weakest types; their gold spans are loose phrases ("open late",
  "good for kids") whose edges vary across the annotation. Not every boundary error is the
  annotation's: 26 of the student's 1,058 spans are sub-word fragments ("up" and "scale" for
  "upscale") where neighbouring subword tokens got different tags, which a word-boundary
  snap at decode time would remove (not done; the number above is without it). Exact-match
  F1 charges every boundary error in full; token kappa is on a different scale (4,508 word
  tags, 62% of them `O`, against 976 entities), so the two are not directly comparable.
- Longer queries are harder: F1 falls from 0.80 on 1-7 word queries to 0.62 on 11-23 words.

<!-- BEGIN generated:extract -->
| Dataset | Teacher | Student | Teacher labels | Held-out | Entity F1 [95% CI] | Precision | Recall | Token kappa [95% CI] | All-O token acc. | Gazetteer F1 [95% CI] | Student minus gazetteer F1 [95% paired CI] | Teacher ceiling F1 | Unaligned (pool) | Label cost | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mit_restaurant | gold | ModernBERT-base | 1442 | 500 | 0.727 [0.696, 0.757] | 0.698 | 0.757 | 0.831 [0.810, 0.850] | 0.623 | 0.472 [0.445, 0.499] | +0.254 [0.222, 0.288] | n/a | 0.0% | $0.00 | 15 min |

#### mit_restaurant, teacher `gold`, student `ModernBERT-base`

- Raw output: `eval/results/extract-mit_restaurant-gold/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.extract mit_restaurant --teacher gold` at `7dace8b` (dirty tree); seed 0, 1000 bootstrap resamples (by input); 4 CPUs; wall 901 s; spent $0.0000
- Data: tner/mit_restaurant@5386634 `dataset/train.json` (MIT Restaurant corpus (Liu et al., 2013, MIT CSAIL SLS), T-NER copy; the dataset card lists the licence as other), 2000 sentences sampled (seed 0, sha256 6548296409fe)
- Training: 1291 examples and 151 held back to pick the epoch (entity F1), out of 1442 teacher labels; 6 epochs, 486 steps, best epoch 4, 882 s
- Alignment, pool: 2895 teacher mentions; matched exact 2895; dropped none (unaligned 0.0%); 0 duplicates; 0 snapped to a token boundary
- Alignment, heldout: 976 teacher mentions; matched exact 976; dropped none (unaligned 0.0%); 0 duplicates; 0 snapped to a token boundary
- Near-duplicates: 7 of 500 held-out sentences share at least 80% of their words with a labelled pool sentence
- The teacher here is the dataset's gold annotation (no API key in this run), so this row measures the harness and how well the tagger learns from 1,442 clean labels; it is not a parity-with-Claude number.
- Entity F1 by type (student / gazetteer, held-out support, types with at least 5): `Hours` 0.50 / 0.35 (56), `Amenity` 0.55 / 0.33 (164), `Rating` 0.63 / 0.61 (70), `Dish` 0.68 / 0.38 (105), `Restaurant_Name` 0.82 / 0.35 (148), `Location` 0.83 / 0.50 (235), `Cuisine` 0.83 / 0.72 (149), `Price` 0.85 / 0.78 (49)
- Teacher entities by outcome: correct 739, wrong boundary 127, wrong type 43, wrong both 25, missed 42; spurious student entities 65
- Entity F1 by length (words): 1-7: 0.80 (n=125), 7-9: 0.79 (n=125), 9-11: 0.75 (n=125), 11-23: 0.62 (n=125)

Alignment stress on the 976 held-out gold spans: every mention corrupted the same way; only the spans a corruption changes are scored, and a span counts when alignment returns it exactly:

| Corruption | Spans changed | Verify only | Repair ladder | Repair: misplaced |
|---|---|---|---|---|
| off_by_one | 976 | 0 (0.0%) | 976 (100.0%) | 0 |
| off_by_2_to_5 | 976 | 0 (0.0%) | 976 (100.0%) | 0 |
| word_index_as_offset | 958 | 0 (0.0%) | 958 (100.0%) | 0 |
| offset_zero | 958 | 0 (0.0%) | 958 (100.0%) | 0 |
| title_case | 973 | 0 (0.0%) | 973 (100.0%) | 0 |
| trailing_full_stop | 976 | 0 (0.0%) | 976 (100.0%) | 0 |
<!-- END generated:extract -->

### The tagger in the cascade: MIT Restaurant (M12)

The classification cascade answers an input when the student's confidence reaches a threshold
chosen so that kappa with the teacher on the answered inputs keeps a target. M12 asks whether
that rule ports to the tagger, where agreement is entity F1 over sentences, and whether the
tagger's confidence (its minimum token probability, which `eval.extract` already stores per
held-out sentence) sees the errors it makes.

- Command: `make eval-extract-cascade` (`python -m eval.extract --predictions
  eval/results/extract-mit_restaurant-gold/report.json`, then `python -m eval.render`). It reads
  the committed M4 predictions (teacher spans, student spans and confidence for the 500
  held-out sentences) and needs no model: 20 s of CPU, $0. A full `make eval-extract` runs it on
  its own report.
- Method: `choose_threshold` is generalised to any statistic pooled over inputs
  (`anvil/calibrate.py`: `PooledStatistic`, `choose_statistic_threshold`). Entity F1 is one such
  statistic (`evaluate.extraction.entity_f1_statistic`): each sentence contributes its (tp, fp,
  fn), a set of sentences scores the F1 of their sum, and the bootstrap resamples whole
  sentences, so a sentence's entities stay together. The `point`, `lower` (one-sided 95%
  percentile bound) and `bca` rules work for any such statistic. The count rules (`exact`,
  `wilson`, the default for classification since M11) do not: F1 is not a share of independent
  trials, so there is no binomial count to bound. Kappa goes through the same scan
  (`KappaStatistic`), and re-running the four committed calibration analyses gives reports
  byte-identical to the unmodified code on the same machine. The split study, the error kinds
  (from M4's own matcher, `errors_by_input`) and the AUROCs are described in the block below.
- `anvil calibrate` and `anvil serve` stay classify-only: nothing serves this threshold yet.

Findings, against the plan's hypotheses:

- **The lower-bound rule answers 41.8% of sentences at F1 0.9 (hypothesis 35-45% confirmed;
  point estimate 46.0%, against the predicted 50%).** It serves 209 sentences at F1 0.925 with a
  bound of 0.903, and the cascade, with the other 291 sentences sent to the teacher, scores F1
  0.973 overall. At 0.95 it answers 23.2% (116 sentences, F1 0.972). BCa lands a few sentences
  lower (41.6% and 21.6%).
- **On unseen halves the bound holds 84.0% of the time at 0.9 against a predicted 85%
  (hypothesis not confirmed; the study cannot tell the two apart).** 168 of 200 splits kept F1
  0.9 on the unseen half's answered sentences (exact 95% interval 78.2-88.8%, which holds 85%); BCa kept it in 86.5%
  and the point rule in 44.0%. At 0.95 the percentile rule kept 70.8% of the splits where it
  found a threshold and BCa 78.6%, so 28.5% and 16.5% of all splits served a cascade that
  missed. The 0.9 figure is below the classification one (90.5% for the percentile rule on
  Banking77 at kappa 0.9). Three causes are plausible, and I did not separate them: the
  percentile bound under-covers (M10 measured 93% for kappa); the search picks the largest
  coverage whose bound clears, which favours lucky fitting halves; and "kept" is judged on about
  100 answered sentences of the other half, whose own F1 estimate is noisy. The count rule that
  fixed this for kappa has no analogue for F1, so the rule ports, but without M11's guarantee.
- **Errors above the threshold are the ones the confidence misses, and boundary errors are
  not the main survivors.** Above the 0.9 cut sit 36.3% of the teacher's entities but only
  8.7% of the boundary errors (11 of 127), against 16.3% of the type errors (7 of 43) and
  16.7% of the missed entities (7 of 42). Above the 0.95 cut no type error remains and missed
  entities are the largest kind left (4 of 42, against 2 of 127 boundary errors).
- **Boundary errors are not detectably easier to see than type errors once both are compared
  with the same clean sentences (hypothesis not confirmed).** Against the 280 clean sentences
  the AUROC of low confidence is 0.863 [0.822, 0.900] for boundary-error sentences and 0.822
  [0.754, 0.884] for type-error ones, a paired difference of +0.041 [-0.021, +0.108]: under the
  predicted 0.05, and the interval includes zero, which the plan expected possible with 38
  type-error sentences. The pilot's gap (0.778 against 0.682) ranked each kind against every
  other sentence, and that difference, +0.096 [+0.008, +0.187], does exclude zero. But in that
  comparison each kind's negatives include the sentences holding the other errors, which also
  have low confidence: 106 boundary-only sentences among the type kind's negatives, against 29
  type-only ones among the boundary kind's (9 sentences hold both). My reading, not a separate
  measurement, is that most of the pilot's gap reflects how large the other error class is
  rather than how visible each kind is. The kind the confidence sees worst is the missed entity
  (0.785 against clean, 0.619 against the rest). A plausible mechanism, untested here: a
  minimum token probability only drops on a token the tagger hesitated over, and a missed
  entity may be one it tagged O without hesitating.

Whether this confidence is worth serving: at F1 0.9 the tagger can answer about two sentences in
five with an unseen-half hold rate of 84-87%. That is a usable cascade, with a weaker guarantee
than the classification one. Its residual errors are skewed toward missed entities, the kind
this confidence cannot see. The tagger's export and serving remain a later milestone.

<!-- BEGIN generated:extract-cascade -->
#### mit_restaurant, teacher `gold`, student `ModernBERT-base`: the tagger in the cascade

- Raw output: `eval/results/cascade-extract-mit_restaurant-gold/` (`report.json`, `meta.json`), from `eval/results/extract-mit_restaurant-gold/report.json` (sha256 `d235ec00397d`, student run `20261018-080309-n1442`)
- Command: `python -m eval.extract --predictions eval/results/extract-mit_restaurant-gold/report.json` at `fe621a5`; 500 held-out sentences, entity F1 0.727 against the teacher; confidence: the minimum token probability, rounded to 4 decimals (463 distinct values); at least 30 answered sentences; 1000 sentence resamples; wall 20 s; no model, spent $0.00

Threshold each rule serves on all held-out sentences (the cascade F1 counts escalated sentences as answered by the teacher):

| Target F1 | Rule | Threshold | Sentences answered | F1 on answered | Lower bound | F1 on escalated | Cascade F1 |
|---|---|---|---|---|---|---|---|
| 0.9 | point estimate | 0.8779 | 230 (46.0%) | 0.901 | n/a | 0.613 | 0.960 |
| 0.9 | 95% percentile lower bound | 0.9032 | 209 (41.8%) | 0.925 | 0.903 | 0.621 | 0.973 |
| 0.9 | 95% BCa lower bound | 0.9036 | 208 (41.6%) | 0.927 | 0.902 | 0.621 | 0.974 |
| 0.95 | point estimate | 0.9599 | 147 (29.4%) | 0.953 | n/a | 0.659 | 0.989 |
| 0.95 | 95% percentile lower bound | 0.9714 | 116 (23.2%) | 0.972 | 0.951 | 0.673 | 0.995 |
| 0.95 | 95% BCa lower bound | 0.9756 | 108 (21.6%) | 0.979 | 0.962 | 0.677 | 0.996 |

Unseen halves: 200 random splits into 250 sentences to fit the threshold on and 250 to test it on. "Kept" counts the splits whose unseen half's answered sentences reach the target among those where a threshold was found (exact 95% interval); "missed" is the share of all splits that served a cascade missing the target.

| Target F1 | Rule | Threshold found | Unseen half: kept the target | Missed (all splits) | Unseen half: coverage | Unseen half: F1 on answered |
|---|---|---|---|---|---|---|
| 0.9 | point estimate | 100.0% | 88/200 (44.0%; 37.0%-51.2%) | 56.0% | 47.4% [40.8%, 59.2%] | 0.894 [0.848, 0.947] |
| 0.9 | 95% percentile lower bound | 100.0% | 168/200 (84.0%; 78.2%-88.8%) | 16.0% | 38.6% [26.8%, 46.5%] | 0.927 [0.877, 0.973] |
| 0.9 | 95% BCa lower bound | 100.0% | 173/200 (86.5%; 81.0%-90.9%) | 13.5% | 35.6% [23.2%, 44.8%] | 0.929 [0.885, 0.983] |
| 0.95 | point estimate | 100.0% | 74/200 (37.0%; 30.3%-44.1%) | 63.0% | 30.0% [22.8%, 42.4%] | 0.944 [0.903, 0.984] |
| 0.95 | 95% percentile lower bound | 97.5% | 138/195 (70.8%; 63.8%-77.0%) | 28.5% | 22.4% [18.8%, 31.3%] | 0.970 [0.920, 0.994] |
| 0.95 | 95% BCa lower bound | 77.0% | 121/154 (78.6%; 71.2%-84.8%) | 16.5% | 21.2% [9.6%, 29.7%] | 0.980 [0.922, 1.000] |

Error kinds of M4's matcher either side of the 95% percentile lower bound threshold (counts of teacher spans, spurious ones of student spans; each row adds up to the committed report's breakdown):

| | All | Above F1 0.9 cut | Below | Above F1 0.95 cut | Below |
|---|---|---|---|---|---|
| Sentences | 500 | 209 | 291 | 116 | 384 |
| Teacher entities | 976 | 354 (36.3%) | 622 (63.7%) | 182 (18.6%) | 794 (81.4%) |
| Entity F1 | 0.727 | 0.925 | 0.621 | 0.972 | 0.673 |
| boundary (same type, other extent) | 127 | 11 (8.7%) | 116 (91.3%) | 2 (1.6%) | 125 (98.4%) |
| type (same extent, other type) | 43 | 7 (16.3%) | 36 (83.7%) | 0 (0.0%) | 43 (100.0%) |
| both | 25 | 2 (8.0%) | 23 (92.0%) | 0 (0.0%) | 25 (100.0%) |
| missed (no overlapping span) | 42 | 7 (16.7%) | 35 (83.3%) | 4 (9.5%) | 38 (90.5%) |
| spurious (overlaps no teacher span) | 65 | 7 (10.8%) | 58 (89.2%) | 2 (3.1%) | 63 (96.9%) |

Does low confidence find each kind? AUROC of the confidence for "this sentence holds an error of this kind", against the clean sentences (no false positive or negative; 280 of them) and against every other sentence, with 95% sentence-resampled intervals. A sentence can hold several kinds (9 hold both a boundary and a type error).

| Error kind | Sentences | AUROC against clean | AUROC against every other sentence |
|---|---|---|---|
| any error | 220 | 0.841 [0.805, 0.873] | n/a |
| boundary (same type, other extent) | 115 | 0.863 [0.822, 0.900] | 0.778 [0.732, 0.820] |
| type (same extent, other type) | 38 | 0.822 [0.754, 0.884] | 0.682 [0.605, 0.758] |
| both | 23 | 0.907 [0.841, 0.955] | 0.797 [0.710, 0.871] |
| missed (no overlapping span) | 36 | 0.785 [0.708, 0.849] | 0.619 [0.539, 0.693] |
| spurious (overlaps no teacher span) | 54 | 0.855 [0.795, 0.907] | 0.754 [0.687, 0.817] |
| boundary minus type (paired) | | +0.041 [-0.021, 0.108] | +0.096 [0.008, 0.187] |

F1 against coverage on all held-out sentences:

| Coverage | Threshold | F1 on answered | Cascade F1 |
|---|---|---|---|
| 100.0% | 0.2221 | 0.727 | 0.727 |
| 90.0% | 0.5046 | 0.764 | 0.789 |
| 80.0% | 0.5748 | 0.798 | 0.841 |
| 70.0% | 0.6770 | 0.835 | 0.887 |
| 60.0% | 0.7751 | 0.871 | 0.927 |
| 50.0% | 0.8531 | 0.896 | 0.953 |
| 40.0% | 0.9121 | 0.927 | 0.975 |
| 30.0% | 0.9556 | 0.948 | 0.987 |
| 20.0% | 0.9807 | 0.987 | 0.998 |

The teacher here is the dataset's gold spans (no API key in this run): this measures the tagger's confidence against clean labels, not against Claude.
<!-- END generated:extract-cascade -->

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
logits are small (largest 1.84) and int8 agreement is a pessimistic proxy; M5 re-measured it on
the trained students (the export section above).

### Calibration and the cascade: AG News and Banking77 (M5)

- Command: `make eval-calibration TEACHER=gold` (`python -m eval.calibration banking77 ag_news
  --teacher gold`, then `python -m eval.render`). It reuses the M2 students: retraining both
  from scratch reproduced the committed training logs step for step and the held-out
  predictions and confidences item for item, so these are the same models as the parity
  table, not look-alikes. The held-out logits are committed (`logits.npz`), so
  `python -m eval.calibration <dataset> --logits eval/results/calibration-<dataset>-gold/logits.npz`
  re-runs the analysis in about a minute without the model.
- Method: "correct" means "agrees with the teacher". The temperature minimises held-out NLL
  (bisection on its gradient in 1/T, checked against scipy); ECE uses 15 equal-width bins.
  The threshold is the largest coverage whose kappa on the answered inputs reaches the target,
  with at least 30 answered inputs, either on the point estimate or on a one-sided 95%
  bootstrap lower bound (1,000 resamples). The first table is in-sample: what `anvil
  calibrate` writes for the server. The split study is the out-of-sample check: 200 times,
  shuffle the held-out split, fit the temperature and the threshold on one half and score the
  other half; "kappa reaches target" is the share of those 200 splits in which the inputs the
  student answered on the unseen half kept the target kappa.

Findings:

- **A threshold chosen on the estimate is a coin flip on new data.** Picked where the kappa
  estimate first reaches 0.9, it kept 0.9 on the unseen half in 55.5% of splits on Banking77
  and 46.0% on AG News: the threshold sits where the sample's noise happens to clear the
  target, so fresh inputs land on either side about equally. Requiring the 95% bootstrap lower
  bound to clear it held the target in 90.5% and 87.0% of splits, for 7 points less coverage on
  Banking77 (median 80.2% against 87.1%) and 23 on AG News (65.1% against 88.3%). That is why
  `anvil calibrate` used the percentile lower bound by default through M10 (since M11 it uses
  the exact count bound, see "A bound on the count itself"). It still falls short of the nominal 95%,
  and the gap is mostly in what the study scores: whether the kappa realised on about 240
  fresh inputs clears the target, a statement about one noisy sample rather than the
  population statement a lower confidence bound makes, so a threshold whose true kappa sits
  just above 0.9 misses on some fresh halves. The bound itself also falls short: M10's
  simulation (below) measures the one-sided percentile bound covering 93% rather than 95% at
  these sizes, and the search over thresholds holding its target 91% of the time against the
  population. The BCa rows (M10) were added by re-running the analysis from the committed
  `logits.npz`; every other row is unchanged. The shares also move by a few points
  with the study seed (`--seed`), so read them as about nine in ten against about one in two.
- **Temperature scaling makes the confidences honest but buys no coverage.** Both students
  are slightly overconfident (T 1.24 and 1.22); on the unseen halves scaling cut the median ECE
  from 0.070 to 0.054 (Banking77) and 0.051 to 0.047 (AG News). Coverage and hit rate with
  uncalibrated confidences are the same within noise (Banking77 at 0.9: 87.9% and 53.0%
  uncalibrated, 87.1% and 55.5% calibrated), because the threshold only uses the order of
  the confidences, which one temperature barely changes. Calibration matters for reading a
  confidence as a probability, not for the cascade's coverage.
- **The product number.** At the served setting (target 0.9, the percentile lower bound in
  M5) the Banking77 student answers 82.3% of held-out inputs at kappa 0.923 (lower bound
  0.900) and escalates the other 17.7%; for 0.95 it answers 70.8%. AG News, whose student has
  almost the same overall kappa (0.844 against 0.842), keeps less (75.7% and 41.9%): its
  errors are spread further up the confidence range (agreement on its most confident half is
  0.971, against 0.992 for Banking77), so a strict target has to give up more inputs to reach
  it. The same overall kappa can mean very different cascades. Under the M11 default (the
  exact count bound) the served coverage is 80.8% and 70.0% on Banking77 and 66.9% and none
  on AG News; the summary table below shows the served rule, the per-rule tables every rule.

<!-- BEGIN generated:calibration -->
| Dataset | Teacher | Student | Held-out | Kappa (all inputs) | Temperature | ECE before -> after (all) | ECE before -> after (unseen half, median) | Coverage at kappa 0.9 | Coverage at kappa 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | gold | ModernBERT-base | 477 | 0.844 | 1.22 | 0.035 -> 0.031 | 0.051 -> 0.047 | 66.9% | n/a |
| ag_news | qwen2.5:3b | ModernBERT-base | 477 | 0.847 | 1.01 | 0.038 -> 0.040 | 0.046 -> 0.048 | 82.4% | 68.8% |
| banking77 | gold | ModernBERT-base | 496 | 0.842 | 1.24 | 0.062 -> 0.043 | 0.070 -> 0.054 | 80.8% | 70.0% |
| banking77 | qwen2.5:3b | ModernBERT-base | 496 | 0.687 | 1.60 | 0.160 -> 0.079 | 0.165 -> 0.091 | 13.7% | n/a |

The cascade on a fallible teacher, scored against gold (the threshold is the one served, rule `exact`, chosen for agreement with the teacher; the cascade answers with the student above it and the teacher below):

| Dataset | Teacher | Target | Coverage | Answered: student vs teacher | Answered: student vs gold | Answered: teacher vs gold | Student minus teacher (paired) | Cascade vs gold | Teacher alone vs gold | Cascade minus teacher |
|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | qwen2.5:3b | 0.9 | 82.4% (393) | 0.931 | 0.772 [0.724, 0.820] | 0.776 [0.730, 0.823] | -0.004 [-0.033, 0.021] | 0.716 [0.669, 0.764] | 0.719 [0.672, 0.764] | -0.003 [-0.025, 0.019] |
| ag_news | qwen2.5:3b | 0.95 | 68.8% (328) | 0.978 | 0.864 [0.819, 0.907] | 0.860 [0.810, 0.903] | +0.004 [-0.014, 0.026] | 0.721 [0.674, 0.768] | 0.719 [0.672, 0.764] | +0.003 [-0.008, 0.014] |
| banking77 | qwen2.5:3b | 0.9 | 13.7% (68) | 0.969 | 0.725 [0.616, 0.827] | 0.710 [0.589, 0.814] | +0.015 [-0.000, 0.046] | 0.546 [0.503, 0.590] | 0.544 [0.501, 0.586] | +0.002 [-0.000, 0.006] |
| banking77 | qwen2.5:3b | 0.95 | no threshold | | | | | | 0.544 [0.501, 0.586] | |

The same comparison at fixed coverage (the most confident inputs), and how well the student's confidence finds the inputs the teacher got wrong:

| Dataset | Coverage | Inputs | Student vs gold | Teacher vs gold | Student minus teacher (paired) |
|---|---|---|---|---|---|
| ag_news | 90.0% | 429 | 0.760 [0.711, 0.807] | 0.751 [0.704, 0.795] | +0.009 [-0.022, 0.038] |
| ag_news | 70.0% | 334 | 0.858 [0.811, 0.901] | 0.850 [0.802, 0.893] | +0.008 [-0.009, 0.030] |
| ag_news | 50.0% | 238 | 0.939 [0.900, 0.973] | 0.933 [0.890, 0.972] | +0.007 [0.000, 0.022] |
| ag_news | 30.0% | 143 | 0.927 [0.852, 0.985] | 0.927 [0.852, 0.985] | +0.000 [0.000, 0.000] |
| banking77 | 90.0% | 446 | 0.572 [0.520, 0.617] | 0.556 [0.508, 0.601] | +0.016 [-0.018, 0.050] |
| banking77 | 70.0% | 347 | 0.628 [0.575, 0.677] | 0.619 [0.565, 0.671] | +0.009 [-0.027, 0.047] |
| banking77 | 50.0% | 248 | 0.692 [0.637, 0.748] | 0.684 [0.625, 0.741] | +0.008 [-0.029, 0.045] |
| banking77 | 30.0% | 149 | 0.752 [0.677, 0.820] | 0.718 [0.640, 0.785] | +0.034 [-0.000, 0.075] |

| Dataset | Teacher wrong (held-out) | AUROC, max softmax | AUROC, probability of the teacher's label |
|---|---|---|---|
| ag_news | 101 (21%) | 0.799 [0.753, 0.845] | 0.812 [0.764, 0.857] |
| banking77 | 223 (45%) | 0.668 [0.621, 0.718] | 0.736 [0.689, 0.782] |

#### ag_news, teacher `gold`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-ag_news-gold/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration ag_news --logits eval/results/calibration-ag_news-gold/logits.npz` at `52dd15a`; student run `20261020-012041-n2000`; 200 random splits into 238 inputs to fit on and 239 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 93 s; spent $0.00
- Fitted temperature over the splits: 1.22 [1.10, 1.34]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 46.0% | 88.3% [75.7%, 95.4%] | 0.897 [0.847, 0.948] | 0.911 [0.855, 0.961] |
| 0.9 | 95% percentile lower bound, calibrated | 100.0% | 87.0% | 65.1% [46.4%, 85.5%] | 0.932 [0.873, 0.988] | 0.961 [0.889, 0.994] |
| 0.9 | 95% BCa lower bound, calibrated | 100.0% | 87.5% | 58.6% [43.9%, 85.4%] | 0.933 [0.883, 1.000] | 0.961 [0.904, 1.000] |
| 0.9 | 95% exact count bound, calibrated | 92.0% | 87.5% | 52.9% [40.3%, 81.9%] | 0.946 [0.893, 0.985] | 0.972 [0.916, 0.994] |
| 0.9 | 95% Wilson count bound, calibrated | 95.0% | 88.4% | 54.0% [39.9%, 83.5%] | 0.946 [0.893, 0.986] | 0.972 [0.913, 0.994] |
| 0.9 | point estimate, uncalibrated | 100.0% | 50.0% | 88.3% [76.6%, 95.4%] | 0.900 [0.847, 0.948] | 0.913 [0.855, 0.961] |
| 0.95 | point estimate, calibrated | 100.0% | 50.5% | 54.0% [43.9%, 77.0%] | 0.951 [0.901, 0.987] | 0.977 [0.927, 0.994] |
| 0.95 | 95% percentile lower bound, calibrated | 100.0% | 82.0% | 36.6% [17.2%, 55.6%] | 0.970 [0.926, 1.000] | 0.992 [0.966, 1.000] |
| 0.95 | 95% BCa lower bound, calibrated | 100.0% | 85.5% | 23.8% [16.7%, 52.7%] | 0.969 [0.930, 1.000] | 0.994 [0.966, 1.000] |
| 0.95 | 95% exact count bound, calibrated | 16.0% | 75.0% | 40.8% [33.9%, 48.8%] | 0.957 [0.935, 0.966] | 0.983 [0.972, 0.989] |
| 0.95 | 95% Wilson count bound, calibrated | 25.0% | 80.0% | 38.9% [34.3%, 51.8%] | 0.962 [0.932, 0.968] | 0.989 [0.969, 0.989] |
| 0.95 | point estimate, uncalibrated | 100.0% | 50.5% | 54.8% [44.3%, 77.1%] | 0.952 [0.901, 0.987] | 0.977 [0.927, 0.994] |

Threshold each rule serves on all held-out inputs:

| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |
|---|---|---|---|---|---|
| 0.9 | point estimate | 0.656 | 88.3% | 0.901 | n/a |
| 0.95 | point estimate | 0.947 | 54.7% | 0.950 | n/a |
| 0.9 | 95% percentile lower bound | 0.834 | 75.7% | 0.928 | 0.901 |
| 0.95 | 95% percentile lower bound | 0.964 | 41.9% | 0.978 | 0.955 |
| 0.9 | 95% BCa lower bound | 0.836 | 75.5% | 0.928 | 0.900 |
| 0.95 | 95% BCa lower bound | 0.966 | 40.9% | 0.977 | 0.954 |
| 0.9 | 95% exact (Clopper-Pearson) count bound | 0.901 | 66.9% | 0.938 | 0.905 |
| 0.95 | 95% exact (Clopper-Pearson) count bound | none | n/a | n/a | n/a |
| 0.9 | 95% Wilson count bound | 0.862 | 72.7% | 0.932 | 0.901 |
| 0.95 | 95% Wilson count bound | none | n/a | n/a | n/a |

Coverage-vs-parity curve on all held-out inputs (calibrated confidence; the cascade kappa counts escalated inputs as answered by the teacher):

| Coverage | Threshold | Kappa on answered | Agreement on answered | Cascade kappa |
|---|---|---|---|---|
| 100.0% | 0.327 | 0.844 | 0.883 | 0.844 |
| 95.0% | 0.539 | 0.874 | 0.905 | 0.880 |
| 89.9% | 0.623 | 0.894 | 0.921 | 0.905 |
| 84.9% | 0.731 | 0.904 | 0.928 | 0.919 |
| 79.9% | 0.789 | 0.918 | 0.940 | 0.936 |
| 74.8% | 0.846 | 0.927 | 0.947 | 0.947 |
| 69.8% | 0.879 | 0.929 | 0.949 | 0.952 |
| 60.0% | 0.928 | 0.935 | 0.955 | 0.964 |
| 49.9% | 0.955 | 0.957 | 0.971 | 0.980 |

The teacher here is the dataset's gold labels (no API key in this run): this measures the student's confidence against clean labels, not against Claude.

#### ag_news, teacher `qwen2.5:3b`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-ag_news-qwen2.5-3b/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration ag_news --logits eval/results/calibration-ag_news-qwen2.5-3b/logits.npz` at `c388f6d`; student run `20261022-000137-n2000`; 200 random splits into 238 inputs to fit on and 239 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 74 s; spent $0.00
- Fitted temperature over the splits: 0.99 [0.87, 1.10]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 42.0% | 91.6% [84.9%, 96.7%] | 0.893 [0.846, 0.945] | 0.903 [0.854, 0.954] |
| 0.9 | 95% percentile lower bound, calibrated | 100.0% | 81.0% | 81.2% [74.5%, 90.0%] | 0.933 [0.880, 0.976] | 0.948 [0.896, 0.983] |
| 0.9 | 95% BCa lower bound, calibrated | 100.0% | 84.5% | 80.8% [74.5%, 89.1%] | 0.935 [0.880, 0.977] | 0.948 [0.896, 0.983] |
| 0.9 | 95% exact count bound, calibrated | 100.0% | 89.5% | 78.2% [73.6%, 87.9%] | 0.945 [0.891, 0.984] | 0.959 [0.907, 0.988] |
| 0.9 | 95% Wilson count bound, calibrated | 100.0% | 88.0% | 79.1% [73.6%, 88.3%] | 0.939 [0.890, 0.983] | 0.954 [0.906, 0.988] |
| 0.9 | point estimate, uncalibrated | 100.0% | 42.0% | 91.6% [84.5%, 96.7%] | 0.893 [0.851, 0.945] | 0.903 [0.857, 0.954] |
| 0.95 | point estimate, calibrated | 100.0% | 49.0% | 77.8% [72.8%, 85.8%] | 0.946 [0.895, 0.984] | 0.960 [0.914, 0.988] |
| 0.95 | 95% percentile lower bound, calibrated | 100.0% | 80.0% | 69.5% [60.2%, 78.2%] | 0.973 [0.929, 1.000] | 0.983 [0.948, 1.000] |
| 0.95 | 95% BCa lower bound, calibrated | 100.0% | 87.0% | 68.4% [53.1%, 77.4%] | 0.974 [0.936, 1.000] | 0.983 [0.954, 1.000] |
| 0.95 | 95% exact count bound, calibrated | 100.0% | 90.5% | 62.3% [45.2%, 75.4%] | 0.980 [0.943, 1.000] | 0.989 [0.960, 1.000] |
| 0.95 | 95% Wilson count bound, calibrated | 100.0% | 88.5% | 63.6% [45.6%, 77.0%] | 0.980 [0.937, 1.000] | 0.989 [0.954, 1.000] |
| 0.95 | point estimate, uncalibrated | 100.0% | 47.0% | 77.8% [73.2%, 85.4%] | 0.946 [0.896, 0.984] | 0.960 [0.914, 0.988] |

Threshold each rule serves on all held-out inputs:

| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |
|---|---|---|---|---|---|
| 0.9 | point estimate | 0.563 | 91.0% | 0.901 | n/a |
| 0.95 | point estimate | 0.807 | 76.9% | 0.953 | n/a |
| 0.9 | 95% percentile lower bound | 0.680 | 84.5% | 0.926 | 0.902 |
| 0.95 | 95% percentile lower bound | 0.844 | 74.6% | 0.967 | 0.950 |
| 0.9 | 95% BCa lower bound | 0.698 | 82.8% | 0.928 | 0.902 |
| 0.95 | 95% BCa lower bound | 0.865 | 73.0% | 0.971 | 0.950 |
| 0.9 | 95% exact (Clopper-Pearson) count bound | 0.707 | 82.4% | 0.931 | 0.900 |
| 0.95 | 95% exact (Clopper-Pearson) count bound | 0.903 | 68.8% | 0.978 | 0.953 |
| 0.9 | 95% Wilson count bound | 0.707 | 82.4% | 0.931 | 0.901 |
| 0.95 | 95% Wilson count bound | 0.879 | 71.7% | 0.974 | 0.951 |

Coverage-vs-parity curve on all held-out inputs (calibrated confidence; the cascade kappa counts escalated inputs as answered by the teacher):

| Coverage | Threshold | Kappa on answered | Agreement on answered | Cascade kappa |
|---|---|---|---|---|
| 100.0% | 0.313 | 0.847 | 0.889 | 0.847 |
| 95.0% | 0.500 | 0.881 | 0.914 | 0.888 |
| 89.9% | 0.581 | 0.903 | 0.930 | 0.914 |
| 84.9% | 0.676 | 0.923 | 0.946 | 0.937 |
| 79.9% | 0.769 | 0.939 | 0.958 | 0.954 |
| 74.8% | 0.837 | 0.968 | 0.978 | 0.977 |
| 69.8% | 0.897 | 0.974 | 0.982 | 0.983 |
| 60.0% | 0.965 | 0.989 | 0.993 | 0.994 |
| 49.9% | 0.990 | 0.993 | 0.996 | 0.997 |

The ceiling on the inputs the cascade answers: of the 200 held-out inputs the teacher labelled under both prompts, those whose calibrated confidence clears each threshold, with the teacher's agreement with itself and the student's agreement with the teacher on the same inputs:

| Threshold | Coverage (all held-out) | Ceiling items | Teacher self-agreement kappa [95% CI] | Student vs teacher kappa [95% CI] |
|---|---|---|---|---|
| top 100.0% | 100.0% | 200 | 0.937 [0.894, 0.972] | 0.861 [0.803, 0.917] |
| top 95.0% | 95.0% | 187 | 0.940 [0.898, 0.977] | 0.903 [0.851, 0.948] |
| top 90.0% | 89.9% | 178 | 0.952 [0.910, 0.984] | 0.913 [0.864, 0.960] |
| top 85.0% | 84.9% | 169 | 0.966 [0.931, 0.992] | 0.932 [0.888, 0.974] |
| top 80.0% | 80.1% | 163 | 0.965 [0.923, 0.991] | 0.947 [0.903, 0.982] |
| top 75.0% | 75.1% | 154 | 0.972 [0.935, 1.000] | 0.962 [0.924, 0.991] |
| top 70.0% | 70.0% | 147 | 0.970 [0.938, 1.000] | 0.970 [0.931, 1.000] |
| top 60.0% | 60.0% | 124 | 0.988 [0.962, 1.000] | 0.988 [0.962, 1.000] |
| top 50.0% | 49.9% | 101 | 0.984 [0.949, 1.000] | 0.984 [0.949, 1.000] |
| served at kappa 0.9 | 82.4% | 166 | 0.965 [0.930, 0.991] | 0.939 [0.894, 0.974] |
| served at kappa 0.95 | 68.8% | 143 | 0.969 [0.928, 1.000] | 0.979 [0.949, 1.000] |

#### banking77, teacher `gold`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-banking77-gold/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration banking77 --logits eval/results/calibration-banking77-gold/logits.npz` at `52dd15a`; student run `20261020-001737-n2000`; 200 random splits into 248 inputs to fit on and 248 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 57 s; spent $0.00
- Fitted temperature over the splits: 1.24 [1.15, 1.31]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 55.5% | 87.1% [81.0%, 92.8%] | 0.905 [0.857, 0.942] | 0.918 [0.869, 0.955] |
| 0.9 | 95% percentile lower bound, calibrated | 100.0% | 90.5% | 80.2% [75.8%, 85.9%] | 0.933 [0.887, 0.968] | 0.947 [0.905, 0.976] |
| 0.9 | 95% BCa lower bound, calibrated | 100.0% | 88.0% | 81.0% [75.8%, 85.9%] | 0.930 [0.887, 0.963] | 0.945 [0.905, 0.971] |
| 0.9 | 95% exact count bound, calibrated | 100.0% | 94.5% | 79.4% [74.2%, 84.3%] | 0.937 [0.899, 0.972] | 0.951 [0.918, 0.979] |
| 0.9 | 95% Wilson count bound, calibrated | 100.0% | 93.5% | 79.4% [74.2%, 84.7%] | 0.937 [0.897, 0.973] | 0.951 [0.914, 0.979] |
| 0.9 | point estimate, uncalibrated | 100.0% | 53.0% | 87.9% [81.0%, 92.7%] | 0.901 [0.856, 0.943] | 0.914 [0.868, 0.955] |
| 0.95 | point estimate, calibrated | 100.0% | 56.5% | 76.6% [69.4%, 81.5%] | 0.952 [0.918, 0.982] | 0.963 [0.934, 0.988] |
| 0.95 | 95% percentile lower bound, calibrated | 100.0% | 89.5% | 69.0% [64.1%, 75.8%] | 0.975 [0.940, 0.994] | 0.984 [0.955, 0.996] |
| 0.95 | 95% BCa lower bound, calibrated | 100.0% | 85.5% | 69.8% [64.9%, 76.6%] | 0.970 [0.940, 0.994] | 0.979 [0.955, 0.996] |
| 0.95 | 95% exact count bound, calibrated | 100.0% | 96.0% | 67.1% [59.7%, 73.0%] | 0.981 [0.954, 1.000] | 0.988 [0.967, 1.000] |
| 0.95 | 95% Wilson count bound, calibrated | 100.0% | 93.0% | 67.7% [60.1%, 73.4%] | 0.981 [0.945, 1.000] | 0.988 [0.959, 1.000] |
| 0.95 | point estimate, uncalibrated | 100.0% | 50.0% | 75.4% [71.0%, 81.0%] | 0.950 [0.917, 0.978] | 0.963 [0.934, 0.984] |

Threshold each rule serves on all held-out inputs:

| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |
|---|---|---|---|---|---|
| 0.9 | point estimate | 0.542 | 87.5% | 0.902 | n/a |
| 0.95 | point estimate | 0.730 | 76.6% | 0.952 | n/a |
| 0.9 | 95% percentile lower bound | 0.639 | 82.3% | 0.923 | 0.900 |
| 0.95 | 95% percentile lower bound | 0.832 | 70.8% | 0.965 | 0.950 |
| 0.9 | 95% BCa lower bound | 0.639 | 82.3% | 0.923 | 0.903 |
| 0.95 | 95% BCa lower bound | 0.830 | 71.4% | 0.966 | 0.951 |
| 0.9 | 95% exact (Clopper-Pearson) count bound | 0.663 | 80.8% | 0.926 | 0.901 |
| 0.95 | 95% exact (Clopper-Pearson) count bound | 0.844 | 70.0% | 0.971 | 0.951 |
| 0.9 | 95% Wilson count bound | 0.663 | 80.8% | 0.926 | 0.902 |
| 0.95 | 95% Wilson count bound | 0.844 | 70.0% | 0.971 | 0.952 |

Coverage-vs-parity curve on all held-out inputs (calibrated confidence; the cascade kappa counts escalated inputs as answered by the teacher):

| Coverage | Threshold | Kappa on answered | Agreement on answered | Cascade kappa |
|---|---|---|---|---|
| 100.0% | 0.155 | 0.842 | 0.845 | 0.842 |
| 95.0% | 0.367 | 0.868 | 0.870 | 0.875 |
| 89.9% | 0.496 | 0.893 | 0.895 | 0.904 |
| 84.9% | 0.579 | 0.911 | 0.912 | 0.924 |
| 79.8% | 0.678 | 0.936 | 0.937 | 0.949 |
| 75.0% | 0.769 | 0.956 | 0.957 | 0.967 |
| 70.0% | 0.844 | 0.971 | 0.971 | 0.980 |
| 59.9% | 0.941 | 0.990 | 0.990 | 0.994 |
| 50.0% | 0.976 | 0.992 | 0.992 | 0.996 |

The teacher here is the dataset's gold labels (no API key in this run): this measures the student's confidence against clean labels, not against Claude.

#### banking77, teacher `qwen2.5:3b`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-banking77-qwen2.5-3b/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration banking77 --logits eval/results/calibration-banking77-qwen2.5-3b/logits.npz` at `c388f6d`; student run `20261022-004245-n2000`; 200 random splits into 248 inputs to fit on and 248 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 46 s; spent $0.00
- Fitted temperature over the splits: 1.61 [1.53, 1.67]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 99.5% | 39.7% | 39.5% [30.2%, 56.6%] | 0.894 [0.834, 0.960] | 0.959 [0.905, 0.988] |
| 0.9 | 95% percentile lower bound, calibrated | 75.0% | 82.7% | 29.8% [11.1%, 36.7%] | 0.926 [0.855, 1.000] | 0.979 [0.950, 1.000] |
| 0.9 | 95% BCa lower bound, calibrated | 92.5% | 83.2% | 29.0% [11.7%, 37.0%] | 0.932 [0.857, 1.000] | 0.979 [0.950, 1.000] |
| 0.9 | 95% exact count bound, calibrated | 46.5% | 88.2% | 14.5% [8.1%, 33.7%] | 0.923 [0.893, 1.000] | 0.988 [0.965, 1.000] |
| 0.9 | 95% Wilson count bound, calibrated | 53.5% | 88.8% | 26.2% [8.1%, 35.0%] | 0.921 [0.864, 1.000] | 0.979 [0.957, 1.000] |
| 0.9 | point estimate, uncalibrated | 100.0% | 42.0% | 40.1% [29.8%, 57.7%] | 0.893 [0.824, 0.960] | 0.959 [0.905, 0.988] |
| 0.95 | point estimate, calibrated | 91.5% | 38.8% | 18.5% [11.3%, 34.6%] | 0.932 [0.893, 1.000] | 0.988 [0.963, 1.000] |
| 0.95 | 95% percentile lower bound, calibrated | 45.5% | 53.8% | 12.9% [8.1%, 21.0%] | 0.959 [0.891, 1.000] | 0.996 [0.979, 1.000] |
| 0.95 | 95% BCa lower bound, calibrated | 46.5% | 50.5% | 13.3% [8.1%, 24.6%] | 0.955 [0.896, 1.000] | 0.996 [0.975, 1.000] |
| 0.95 | 95% exact count bound, calibrated | 0.0% | n/a | n/a | n/a | n/a |
| 0.95 | 95% Wilson count bound, calibrated | 0.5% | 0.0% | 28.6% [28.6%, 28.6%] | 0.897 [0.897, 0.897] | 0.971 [0.971, 0.971] |
| 0.95 | point estimate, uncalibrated | 98.0% | 40.3% | 22.6% [13.7%, 34.3%] | 0.942 [0.879, 1.000] | 0.988 [0.962, 1.000] |

Threshold each rule serves on all held-out inputs:

| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |
|---|---|---|---|---|---|
| 0.9 | point estimate | 0.903 | 34.7% | 0.904 | n/a |
| 0.95 | point estimate | 0.970 | 14.3% | 0.955 | n/a |
| 0.9 | 95% percentile lower bound | 0.918 | 31.9% | 0.934 | 0.901 |
| 0.95 | 95% percentile lower bound | 0.973 | 13.3% | 0.984 | 0.951 |
| 0.9 | 95% BCa lower bound | 0.918 | 31.9% | 0.934 | 0.902 |
| 0.95 | 95% BCa lower bound | 0.973 | 13.3% | 0.984 | 0.951 |
| 0.9 | 95% exact (Clopper-Pearson) count bound | 0.973 | 13.7% | 0.969 | 0.905 |
| 0.95 | 95% exact (Clopper-Pearson) count bound | none | n/a | n/a | n/a |
| 0.9 | 95% Wilson count bound | 0.973 | 13.7% | 0.969 | 0.910 |
| 0.95 | 95% Wilson count bound | none | n/a | n/a | n/a |

Coverage-vs-parity curve on all held-out inputs (calibrated confidence; the cascade kappa counts escalated inputs as answered by the teacher):

| Coverage | Threshold | Kappa on answered | Agreement on answered | Cascade kappa |
|---|---|---|---|---|
| 100.0% | 0.085 | 0.687 | 0.694 | 0.687 |
| 95.0% | 0.202 | 0.718 | 0.724 | 0.732 |
| 89.9% | 0.282 | 0.739 | 0.744 | 0.765 |
| 84.9% | 0.371 | 0.757 | 0.762 | 0.794 |
| 79.8% | 0.433 | 0.765 | 0.770 | 0.813 |
| 75.0% | 0.484 | 0.783 | 0.788 | 0.837 |
| 70.0% | 0.533 | 0.799 | 0.804 | 0.860 |
| 59.9% | 0.664 | 0.851 | 0.855 | 0.911 |
| 50.0% | 0.768 | 0.876 | 0.879 | 0.938 |

The ceiling on the inputs the cascade answers: of the 200 held-out inputs the teacher labelled under both prompts, those whose calibrated confidence clears each threshold, with the teacher's agreement with itself and the student's agreement with the teacher on the same inputs:

| Threshold | Coverage (all held-out) | Ceiling items | Teacher self-agreement kappa [95% CI] | Student vs teacher kappa [95% CI] |
|---|---|---|---|---|
| top 100.0% | 100.0% | 200 | 0.831 [0.774, 0.882] | 0.667 [0.604, 0.732] |
| top 95.0% | 95.0% | 192 | 0.840 [0.786, 0.887] | 0.696 [0.626, 0.758] |
| top 90.0% | 89.9% | 180 | 0.863 [0.811, 0.914] | 0.721 [0.652, 0.783] |
| top 85.0% | 85.1% | 169 | 0.860 [0.805, 0.908] | 0.745 [0.676, 0.811] |
| top 80.0% | 80.0% | 157 | 0.869 [0.815, 0.921] | 0.758 [0.685, 0.822] |
| top 75.0% | 75.0% | 144 | 0.879 [0.821, 0.928] | 0.786 [0.713, 0.849] |
| top 70.0% | 70.0% | 133 | 0.892 [0.836, 0.945] | 0.799 [0.720, 0.868] |
| top 60.0% | 60.1% | 117 | 0.885 [0.823, 0.938] | 0.868 [0.804, 0.929] |
| top 50.0% | 50.0% | 101 | 0.898 [0.826, 0.948] | 0.908 [0.836, 0.959] |
| served at kappa 0.9 | 13.7% | 34 | 0.968 [0.900, 1.000] | 0.936 [0.837, 1.000] |
| served at kappa 0.95 | no threshold found | | | |
<!-- END generated:calibration -->

With the local teacher (M8, `make eval-calibration TEACHER=ollama`, on the students of the
fallible-teacher table): "correct" now means "agrees with `qwen2.5:3b`", which is wrong
against gold on 21% of AG News and 45% of Banking77 held-out inputs.

- **The plan's prediction failed: a kappa-0.9 threshold exists on Banking77.** The plan
  expected none, since the teacher's self-agreement over all 200 ceiling inputs is 0.831. The
  percentile rule (M8's default) finds one at 31.9% coverage and the exact rule (M11's) at
  13.7%, because the teacher is more consistent on the inputs the student is sure of. On the
  34 ceiling inputs above the exact rule's served threshold, its self-agreement is 0.968
  [0.900, 1.000], and the student's agreement with it is 0.936. (M8 read the same on the 66
  inputs above the percentile threshold: 0.936 and 0.952.) The ceiling has to be read on the
  inputs the cascade answers, not on the whole split.
- **The rule holds up less well against a noisier teacher.** At kappa 0.9, the percentile
  lower-bound threshold kept the target on the unseen half in 82.7% of the splits where it found one on
  Banking77 (it found one in 75.0% of the 200, so 62% of all splits) and 81.0% on AG News, against 90.5% and 87.0% with gold. The
  point-estimate threshold kept it in 39.7% and 42.0%.
- **This is consistency with the teacher, not correctness.** Inputs that a confident student
  and a self-consistent teacher agree on can still be wrong in the same way, like AG News's
  `sci_tech` stories labelled `business`. A cascade calibrated against a fallible teacher
  serves the teacher's systematic errors at full confidence.
- **Against gold, the cascade is exactly as good as its teacher (M13, the table at the top of
  this block).** Joining the two committed `logits.npz` files to the gold labels of the
  matching `classify-*` report: at the served threshold the student agrees with the teacher at
  0.969 (Banking77, 13.7% of inputs) and 0.931 (AG News, 82.4%), but with gold at 0.725 and
  0.772, while the teacher itself scores 0.710 and 0.776 with gold on the same inputs. The
  paired student-minus-teacher difference is +0.015 [-0.000, 0.046] and -0.004 [-0.033, 0.021].
  The whole cascade (student above the threshold, teacher below) scores 0.546 and 0.716 with
  gold against 0.544 and 0.719 for the teacher alone; the paired differences are within 0.003
  of zero with intervals no wider than [-0.025, 0.019]. At fixed coverages of 90%, 70%, 50%
  and 30% the gap stays within 0.035 on both datasets, and every interval includes zero or
  touches it. So the cascade saves teacher calls without adding errors, and does not remove
  any either: the ceiling it reaches is the teacher's correctness, not 0.9. The student's
  confidence does see where the teacher is wrong (AUROC 0.668 for max softmax and 0.736 for
  the probability of the teacher's own label on Banking77; 0.799 and 0.812 on AG News), but
  the threshold sorts by that confidence, so above it the teacher is also more often right.
  Whether that signal can clean the training labels is M13's flagging study.

### Does the interval cover? The bound behind the threshold rule (M10)

The cascade's threshold rule serves the largest coverage whose one-sided 95% bootstrap lower
bound of kappa clears the target, so the rule is only as honest as that bound. M10 asks whether
the bound covers, and whether BCa (bias-corrected and accelerated) would do better than the
percentile method at the sizes the rule actually meets.

- Command: `make eval-coverage` (`python -m eval.coverage`, then `python -m eval.render`). It is
  a seeded simulation, offline and free. It runs in about 30 s on two cores, and CI re-runs it
  with `--check`, under a 60 s timeout, against the committed counts.
- Method, fixed sample: a population with known kappa (the student copies the teacher's
  uniformly drawn label with probability kappa, else draws its own, so the population kappa is
  exactly that probability). For each cell of kappa {0.85, 0.9, 0.95} x items {100, 250, 500}
  x labels {4, 77}, there are 500 replications, each with the 1,000 resamples
  `choose_threshold` draws. "Covers" means the bound is at or below the population kappa. The
  Monte Carlo standard error of a coverage near 93% is about 1.1 points per cell.
- Method, threshold search: inputs with a confidence c uniform on [0, 1] and an agreement
  probability of 1 - 0.8 (1 - c)^4. Its kappa on the top u of inputs is 1 - 0.16 u^4, which
  tracks the Banking77 gold student's committed curve within 0.011 at every grid
  coverage (0.84 on everything, 0.934 at 80% against 0.936, 0.990 at 50% against 0.992).
  Each replication runs the whole search for every rule on one sample, then reads the
  population kappa at the chosen threshold. The search cells are 30 replications each, so
  read them pooled (about 360 per rule). Target 0.85 is left out, because there the oracle
  threshold already answers 98%.
- The coverage cost on real data comes from re-running `make eval-calibration`'s analysis
  from the four committed `logits.npz` files (`python -m eval.calibration <dataset> --logits
  ...`). Those tables are in the calibration section above: the BCa rows of the split study
  and "Threshold each rule serves".

Findings, against the plan's hypotheses:

- **Neither bound reaches 95%, and the shortfall is small.** Pooled over 9,000 replications,
  the one-sided percentile bound covers 93.1% and BCa 93.0%. The worst percentile cells are
  4 labels at kappa 0.9-0.95 (89.4% at 100 items, 89.6% at 250). The plan's 88-92% for the
  percentile bound at kappa 0.95 and 250 items holds for 4 labels (89.6%) but not for 77
  (94.6%).
- **BCa is not the upgrade the plan expected (hypothesis: 93-96%).** It helps where the
  percentile bound is weakest: 4 labels pooled, 92.9% against 92.1%, with 92.8% against 89.4%
  at 100 items and kappa 0.9. It hurts with 77 labels: pooled 93.0% against 94.2%, and 88.6%
  against 92.4% at 100 items and kappa 0.95. The cause is visible in the "BCa level" column.
  Near kappa 1 the statistic is nearly a function of the disagreement count, which is
  discrete. With 77 labels, each resample's chance term comes out slightly above the
  sample's, because duplicated items inflate the product of the marginals. So the replicates
  that tie the sample on disagreements all land just below the estimate. BCa reads that as
  bias (z0 > 0) and moves the 5% endpoint up to the 5.7-7.0% level, a less conservative
  bound.
  With 4 labels the effect reverses (4.0-4.9%).
- **The estimator is not biased (hypothesis refuted).** The plan expected a bias at 77 labels
  and 100 items that no interval fixes. The measured bias is +0.0000 ± 0.0010 there, and
  at most 0.002 in every cell. The bootstrap's spread matches the estimate's true spread to
  within about 5%. What under-covers is not a shifted centre but the discrete tail. A sample with
  only one or two disagreements in 100 items yields a bound above the truth however it is
  read; I found that in an exploratory run, and it is not in the committed report.
- **The threshold search costs about two points (hypothesis confirmed, small).** Picking the
  best of many thresholds, the percentile rule's choice truly kept the target in 325 of 358
  searches that found one (90.8%), and BCa's in 324 of 359 (90.3%). The 2 and 1 searches
  that found none escalate everything, which keeps the target trivially. That compares with 93% for one bound at a
  fixed size. The point-estimate rule kept it in 179 of 360 (49.7%), the coin flip the M5
  split study saw on real data. For that, the bound rules answer about 10 points fewer inputs
  than the oracle threshold (percentile 10.8, BCa 9.8), while the point rule gives up 0.2.
- **On the committed logits BCa's cost depends on the dataset.** On the whole held-out
  split, the threshold `anvil calibrate` would serve moves by at most 1.7 points, inside the
  plan's "at most 3 points". Banking77 (gold) is the same at 0.9 (82.3%) and BCa is 0.6
  points higher at 0.95; AG News (`qwen2.5:3b`) is 1.7 points lower at 0.9. On the unseen
  halves the picture splits by dataset. On Banking77 (gold), BCa kept 0.9 in 88.0% of splits
  against 90.5%, and 0.95 in 85.5% against 89.5%, which matches the 77-label simulation.
  With the `qwen2.5:3b` teacher on Banking77 it is mixed (0.9: 83.2% against 82.7%; 0.95:
  50.5% against 53.8%). On AG News (gold, 4 labels) BCa kept the target more often (87.5%
  against 87.0%; 85.5% against 82.0%), but paid in coverage: median 58.6% against 65.1% at
  0.9, and 23.8% against 36.6% at 0.95.

**The default stayed the percentile bound in M10 (`rule: lower`; M11 below replaced it).**
Pooled, the table shows no
coverage gain from BCa (93.0% against 93.1%) and no gain in the search (90.3% against 90.8%).
In the regime the served Banking77 cascade lives in (77 labels, kappa 0.9-0.95), BCa is the
worse bound in the simulation and on the gold Banking77 splits. Its advantage is confined to
4-label tasks, and there it pays heavily in coverage on the unseen AG News halves. BCa stays available as `anvil calibrate --rule bca`,
and `eval.calibration` keeps reporting it. Neither method fixes the remaining 2-point
shortfall, which comes from the discreteness at few disagreements. A bound built on the
disagreement count itself, such as an exact binomial bound on 1 - agreement, is the next
thing to try.

<!-- BEGIN generated:coverage -->
- Raw output: `eval/results/coverage-simulation/` (`report.json`, `meta.json`)
- Command: `python -m eval.coverage --workers 2` at `e404ed4`; 500 replications per fixed-sample cell, 30 per threshold-search cell; 1000 bootstrap resamples; 2 workers; wall 34 s; spent $0.00
- Search population: agreement probability 1 - 0.8 * (1 - c) ** 4.0 for confidence c uniform on [0, 1]; population kappa at coverage (100.0% -> 0.840, 90.0% -> 0.895, 80.0% -> 0.934, 70.0% -> 0.962, 60.0% -> 0.979, 50.0% -> 0.990)

#### Fixed sample: does the one-sided 95% lower bound cover the population kappa?

| Labels | Items | Kappa | Percentile bound covers | BCa bound covers | Exact bound covers | Wilson bound covers | Mean bound: Percentile / BCa / Exact / Wilson | BCa level for the 5% bound (mean) | Two-sided 95%: percentile / BCa | Bias of the estimate (± s.e.) | SD of the estimate / bootstrap SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 100 | 0.85 | 93.6% | 93.4% | 96.8% | 96.2% | 0.775 / 0.773 / 0.760 / 0.765 | 4.7% | 94.4% / 95.8% | -0.0016 ± 0.0019 | 0.0428 / 0.0422 |
| 4 | 100 | 0.9 | 89.4% | 92.8% | 98.2% | 94.4% | 0.838 / 0.835 / 0.821 / 0.826 | 4.5% | 92.6% / 93.8% | -0.0009 ± 0.0016 | 0.0365 / 0.0349 |
| 4 | 100 | 0.95 | 90.6% | 92.8% | 98.6% | 98.6% | 0.903 / 0.900 / 0.884 / 0.888 | 4.0% | 89.4% / 95.6% | -0.0020 ± 0.0012 | 0.0260 / 0.0251 |
| 4 | 250 | 0.85 | 93.6% | 93.6% | 95.4% | 95.4% | 0.804 / 0.804 / 0.798 / 0.800 | 4.8% | 93.0% / 93.4% | -0.0004 ± 0.0012 | 0.0277 / 0.0266 |
| 4 | 250 | 0.9 | 93.0% | 93.4% | 96.6% | 96.6% | 0.863 / 0.862 / 0.856 / 0.858 | 4.7% | 94.4% / 94.2% | +0.0006 ± 0.0010 | 0.0229 / 0.0221 |
| 4 | 250 | 0.95 | 89.6% | 90.6% | 94.6% | 94.6% | 0.923 / 0.922 / 0.916 / 0.917 | 4.4% | 93.4% / 93.6% | +0.0004 ± 0.0007 | 0.0161 / 0.0158 |
| 4 | 500 | 0.85 | 93.8% | 94.2% | 96.2% | 96.2% | 0.818 / 0.818 / 0.815 / 0.816 | 4.9% | 94.4% / 94.8% | -0.0002 ± 0.0009 | 0.0193 / 0.0189 |
| 4 | 500 | 0.9 | 92.4% | 92.6% | 94.4% | 94.4% | 0.874 / 0.874 / 0.871 / 0.871 | 4.8% | 93.0% / 93.2% | +0.0003 ± 0.0007 | 0.0161 / 0.0156 |
| 4 | 500 | 0.95 | 92.6% | 93.0% | 95.4% | 95.4% | 0.931 / 0.930 / 0.927 / 0.928 | 4.6% | 92.8% / 93.0% | +0.0002 ± 0.0005 | 0.0117 / 0.0112 |
| 77 | 100 | 0.85 | 94.2% | 93.6% | 98.8% | 98.8% | 0.788 / 0.793 / 0.777 / 0.781 | 6.7% | 96.8% / 95.4% | -0.0004 ± 0.0015 | 0.0342 / 0.0359 |
| 77 | 100 | 0.9 | 93.4% | 91.6% | 97.2% | 97.2% | 0.849 / 0.854 / 0.838 / 0.841 | 6.7% | 93.8% / 93.0% | +0.0013 ± 0.0013 | 0.0301 / 0.0299 |
| 77 | 100 | 0.95 | 92.4% | 88.6% | 96.0% | 96.0% | 0.912 / 0.916 / 0.898 / 0.901 | 7.0% | 94.4% / 94.6% | +0.0000 ± 0.0010 | 0.0217 / 0.0216 |
| 77 | 250 | 0.85 | 94.6% | 94.4% | 95.8% | 95.8% | 0.811 / 0.813 / 0.807 / 0.808 | 6.0% | 94.4% / 95.2% | -0.0004 ± 0.0010 | 0.0231 / 0.0228 |
| 77 | 250 | 0.9 | 95.6% | 94.4% | 98.0% | 98.0% | 0.867 / 0.869 / 0.862 / 0.864 | 6.0% | 96.6% / 96.6% | -0.0005 ± 0.0008 | 0.0189 / 0.0191 |
| 77 | 250 | 0.95 | 94.6% | 93.0% | 97.6% | 97.6% | 0.924 / 0.926 / 0.919 / 0.920 | 6.1% | 95.0% / 94.6% | -0.0017 ± 0.0006 | 0.0135 / 0.0140 |
| 77 | 500 | 0.85 | 96.4% | 96.2% | 96.6% | 96.6% | 0.822 / 0.823 / 0.820 / 0.821 | 5.7% | 95.0% / 95.4% | -0.0007 ± 0.0007 | 0.0157 / 0.0161 |
| 77 | 500 | 0.9 | 92.8% | 92.4% | 95.0% | 95.0% | 0.878 / 0.879 / 0.876 / 0.877 | 5.7% | 95.2% / 95.0% | +0.0009 ± 0.0006 | 0.0131 / 0.0134 |
| 77 | 500 | 0.95 | 93.6% | 92.6% | 95.0% | 95.0% | 0.933 / 0.934 / 0.931 / 0.931 | 5.7% | 92.8% / 92.6% | -0.0001 ± 0.0004 | 0.0101 / 0.0098 |

Pooled over all 18 cells (9000 replications), the one-sided bound covers: Percentile 93.1%, BCa 93.0%, Exact 96.5%, Wilson 96.2%.
- 4 labels: Percentile 92.1%, BCa 92.9%, Exact 96.2%, Wilson 95.8%
- 77 labels: Percentile 94.2%, BCa 93.0%, Exact 96.7%, Wilson 96.7%
- Worst cell: Percentile 89.4% (4 labels, 100 items, kappa 0.9); BCa 88.6% (77 labels, 100 items, kappa 0.95); Exact 94.4% (4 labels, 500 items, kappa 0.9); Wilson 94.4% (4 labels, 100 items, kappa 0.9)

#### Threshold search: how often does the chosen threshold truly keep the target?

| Labels | Items | Target | Oracle coverage | Point: met | Percentile: met | BCa: met | Exact: met | Wilson: met | Point: true coverage | Percentile: true coverage | BCa: true coverage | Exact: true coverage | Wilson: true coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 100 | 0.9 | 88.9% | 15/30 (50.0%) | 27/30 (90.0%) | 26/30 (86.7%) | 27/27 (100.0%) | 29/29 (100.0%) | 89.1% | 74.9% | 76.6% | 67.8% | 68.2% |
| 4 | 100 | 0.95 | 74.8% | 14/30 (46.7%) | 25/28 (89.3%) | 26/29 (89.7%) | 0/1 (0.0%) | 3/5 (60.0%) | 74.9% | 58.2% | 58.8% | 84.0% | 73.8% |
| 4 | 250 | 0.9 | 88.9% | 17/30 (56.7%) | 27/30 (90.0%) | 28/30 (93.3%) | 28/30 (93.3%) | 28/30 (93.3%) | 87.8% | 78.2% | 78.0% | 75.9% | 76.3% |
| 4 | 250 | 0.95 | 74.8% | 14/30 (46.7%) | 27/30 (90.0%) | 28/30 (93.3%) | 28/28 (100.0%) | 30/30 (100.0%) | 74.8% | 64.3% | 63.6% | 57.4% | 56.6% |
| 4 | 500 | 0.9 | 88.9% | 14/30 (46.7%) | 28/30 (93.3%) | 28/30 (93.3%) | 30/30 (100.0%) | 30/30 (100.0%) | 88.8% | 82.2% | 82.2% | 81.1% | 81.4% |
| 4 | 500 | 0.95 | 74.8% | 14/30 (46.7%) | 27/30 (90.0%) | 27/30 (90.0%) | 30/30 (100.0%) | 29/30 (96.7%) | 75.2% | 68.2% | 67.8% | 65.0% | 65.3% |
| 77 | 100 | 0.9 | 88.9% | 13/30 (43.3%) | 28/30 (93.3%) | 27/30 (90.0%) | 30/30 (100.0%) | 30/30 (100.0%) | 88.0% | 76.5% | 77.7% | 70.4% | 72.3% |
| 77 | 100 | 0.95 | 74.8% | 14/30 (46.7%) | 27/30 (90.0%) | 27/30 (90.0%) | 8/8 (100.0%) | 12/12 (100.0%) | 74.6% | 53.1% | 60.4% | 66.6% | 63.2% |
| 77 | 250 | 0.9 | 88.9% | 18/30 (60.0%) | 27/30 (90.0%) | 27/30 (90.0%) | 29/30 (96.7%) | 28/30 (93.3%) | 88.5% | 79.8% | 80.5% | 77.9% | 78.1% |
| 77 | 250 | 0.95 | 74.8% | 15/30 (50.0%) | 27/30 (90.0%) | 25/30 (83.3%) | 26/29 (89.7%) | 27/30 (90.0%) | 74.9% | 65.4% | 66.9% | 62.4% | 62.2% |
| 77 | 500 | 0.9 | 88.9% | 16/30 (53.3%) | 28/30 (93.3%) | 28/30 (93.3%) | 29/30 (96.7%) | 29/30 (96.7%) | 88.9% | 83.5% | 83.8% | 82.7% | 82.8% |
| 77 | 500 | 0.95 | 74.8% | 15/30 (50.0%) | 27/30 (90.0%) | 27/30 (90.0%) | 28/30 (93.3%) | 28/30 (93.3%) | 74.7% | 67.4% | 68.2% | 65.8% | 66.1% |

Pooled over all 12 cells: point estimate met 179/360 (49.7%), coverage given up versus the oracle 0.2%; 95% percentile lower bound met 325/358 (90.8%), coverage given up versus the oracle 10.8%; 95% BCa lower bound met 324/359 (90.3%), coverage given up versus the oracle 9.8%; 95% exact (Clopper-Pearson) count bound met 293/303 (96.7%), coverage given up versus the oracle 12.3%; 95% Wilson count bound met 303/316 (95.9%), coverage given up versus the oracle 12.1%.

- 4 labels: Point found a threshold in 180/180, met 48.9%, true coverage 81.8% (81.8% counting a search with no threshold as 0); Percentile found a threshold in 178/180, met 90.4%, true coverage 71.1% (70.4% counting a search with no threshold as 0); BCa found a threshold in 179/180, met 91.1%, true coverage 71.2% (70.8% counting a search with no threshold as 0); Exact found a threshold in 146/180, met 97.9%, true coverage 69.7% (56.6% counting a search with no threshold as 0); Wilson found a threshold in 154/180, met 96.8%, true coverage 69.7% (59.6% counting a search with no threshold as 0).
- 77 labels: Point found a threshold in 180/180, met 50.6%, true coverage 81.6% (81.6% counting a search with no threshold as 0); Percentile found a threshold in 180/180, met 91.1%, true coverage 71.0% (71.0% counting a search with no threshold as 0); BCa found a threshold in 180/180, met 89.4%, true coverage 72.9% (72.9% counting a search with no threshold as 0); Exact found a threshold in 157/180, met 95.5%, true coverage 71.6% (62.5% counting a search with no threshold as 0); Wilson found a threshold in 162/180, met 95.1%, true coverage 71.6% (64.5% counting a search with no threshold as 0).

#### Pointwise: how often does each count bound cover the agreement rate itself?

Exact binomial sums, no sampling: for m answered inputs and each of 2000 agreement rates p from 0.8 to 0.999, the probability that the 95% bound on p_o is at or below p.

| Answered inputs | Bound | Worst coverage | At agreement rate | Mean coverage | Share of rates under 95% |
|---|---|---|---|---|---|
| 30 | Exact | 95.01% | 0.851 | 98.63% | 0.0% |
| 30 | Wilson | 92.50% | 0.917 | 97.66% | 18.6% |
| 100 | Exact | 95.00% | 0.911 | 97.17% | 0.0% |
| 100 | Wilson | 93.10% | 0.974 | 96.42% | 22.6% |
| 250 | Exact | 95.00% | 0.928 | 96.41% | 0.0% |
| 250 | Wilson | 93.30% | 0.989 | 95.85% | 24.2% |
| 500 | Exact | 95.00% | 0.869 | 96.01% | 0.0% |
| 500 | Wilson | 93.59% | 0.995 | 95.58% | 25.6% |
<!-- END generated:coverage -->

### A bound on the count itself (M11)

M10 ended with both bootstrap bounds covering 93% where they claim 95%, an unbiased estimator,
and a named next step: bound the agreement count directly. On the m inputs a threshold accepts,
the student agrees with the teacher on A of them; if each agrees with probability p_o, A is
binomial, and the binomial distribution gives a lower bound on p_o with no resampling. Two
rules join `choose_threshold` (`anvil/evaluate/binomial.py`):

- `exact`: the one-sided Clopper-Pearson bound, the p at which seeing A or more agreements has
  probability 5%. It inverts the binomial test itself, so it covers p_o at least 95% at every p.
- `wilson`: the one-sided Wilson score bound, the normal approximation to the same test.

Both map the bound L on p_o to kappa as (L - p_e) / (1 - p_e), where p_e is the chance term of
the accepted inputs themselves, treated as known (the plug-in). Kappa is a different statistic
from p_o, so the guarantee does not transfer exactly: the chance term's own sampling error is
ignored, and the simulation checks whether that matters.

- Command: `make eval-coverage` (the same simulation as M10, with the count bounds computed on
  the same samples; the bootstrap columns are unchanged count for count), then
  `python -m eval.calibration <dataset> --logits eval/results/calibration-<dataset>-<teacher>/logits.npz`
  on the four committed files. The simulation takes 34 s on 2 workers (31 s under `--check`,
  inside CI's 60 s); the four analyses take 30-75 s each. No model, $0.
- The pointwise table (above) is new: exact binomial sums over 2,000 agreement rates from 0.8
  to 0.999 at 30, 100, 250 and 500 answered inputs, so it has no Monte Carlo error.

Findings, against the plan's hypotheses:

- **Both count bounds reach 95% (hypothesis confirmed).** Pooled over the 18 fixed-sample cells
  (9,000 replications), the exact bound covers 96.5% and Wilson 96.2%, against 93.1% for the
  percentile bound on the same samples. No cell is under 93%: the worst is 94.4% for both
  (exact at 4 labels, 500 items, kappa 0.9; Wilson tied between that cell and 4 labels, 100
  items, kappa 0.9). Two exact cells sit under 95% (94.6% and 94.4%), both less than one Monte
  Carlo standard error (1.0 point per cell) away, so the claim is "within Monte Carlo error of
  95% in every cell", not "at least 95%". The price is a bound below the percentile one on
  average by 0.002-0.004 (exact) or 0.001-0.003 (Wilson) of kappa at 500 items, and by
  0.010-0.020 (exact) or 0.007-0.016 (Wilson) at 100 items.
- **The 4-label, 100-item cells over-cover, as predicted, but only with exact.** Exact covers
  96.8%, 98.2% and 98.6% there (kappa 0.85, 0.9, 0.95), above the plan's 96.5%. Wilson covers
  96.2%, 94.4% and 98.6%, so the prediction holds for it in one cell of three. The percentile
  bound covers 93.6%, 89.4% and 90.6%, under the plan's 92% in two of the three.
- **The plug-in chance term costs no coverage that this simulation can see.** If ignoring the
  chance term's sampling error mattered, it would show where p_e is largest and noisiest, at 4
  labels and 100 items, and exact covers 96.8-98.6% there. That evidence is weaker than it
  looks: Clopper-Pearson's own conservatism could hide a plug-in cost, and Wilson, closer to
  nominal, covers 94.4% in one of those cells. The simulated raters also have balanced
  marginals, where p_e varies least; skewed label distributions are untested.
- **Wilson is not exact, and the pointwise table shows where.** Clopper-Pearson's worst
  coverage of p_o is 95.00% at every size. Wilson's dips to 92.50% (30 inputs) to 93.59%
  (500), and it is under 95% at 19-26% of the agreement rates. Its worst rates move toward 1
  as m grows (0.917 at 30 inputs, 0.995 at 500), which is where a cascade answers. On average
  Wilson is close to nominal (95.6-97.7%), and that is what the kappa simulation sees.
- **The threshold search keeps its target (hypothesis confirmed).** The chosen threshold truly
  kept the target in 293 of 303 searches with exact (96.7%) and 303 of 316 with Wilson
  (95.9%), against 325 of 358 (90.8%) for the percentile rule. The count rules find no
  threshold more often (57 and 44 of 360 searches, against 2), mostly at 100 items and target
  0.95 (51 of the 57). At 4 labels with p_e exactly 0.25, the exact bound on kappa first
  reaches 0.95 at 79 accepted inputs that all agree; any disagreement or skewed marginals need
  more, so a 100-item sample rarely has such a prefix. At 77 labels p_e is near 0.013 and 100
  agreeing inputs give about 0.970. Counting the searches with no threshold as answering
  nothing, the
  expected served coverage at 4 labels is 56.6% (exact), 59.6% (Wilson) and 70.4%
  (percentile). At 77 labels it is 62.5%, 64.5% and 71.0%, against an oracle 81.8% for both.
- **The served coverage cost does not follow the label count; it looks like the curve's shape
  near the target.** At 0.9, the exact threshold answers 80.8% of Banking77 (gold), against 82.3% for
  the percentile rule, 1.4 points less (hypothesis: at most 2). On AG News (gold) it answers
  66.9% against 75.7%, 8.8 points less (hypothesis: at least 5); Wilson is 72.7%. At 0.95 on
  AG News neither count rule finds any threshold, where the percentile rule serves 41.9%
  (hypothesis confirmed). The `qwen2.5:3b` students break the label-count reading: Banking77
  with that teacher loses 18.1 points at 0.9 (31.9% to 13.7%, both count rules) and finds
  nothing at 0.95, while AG News with that teacher loses only 2.1 at 0.9 (84.5% to 82.4%). My
  reading is that where the kappa-vs-coverage curve is flat near the target, a bound 0.01
  lower moves the threshold a long way, and where it is steep the honest bound is nearly free.
  That is a hypothesis from four curves, not a measurement, and the qwen Banking77 case is
  confounded: its percentile threshold accepts 158 inputs (the exact one 68) against about
  400 for gold under either rule, so its bound is also wider because m is smaller.
- **On the unseen AG News halves at 0.95, the count rules decline instead of missing
  (hypothesis on the conditional rate refuted).** The percentile rule found a threshold on
  every fitting half and kept 0.95 on the unseen half in 82.0% of them, so 18% of splits
  served a cascade that missed. The exact rule found a threshold in 16.0% of splits and kept
  the target in 75.0% of those (24 of 32 splits, exact 95% interval 57-89%); Wilson found one
  in 25.0% and kept it in 80.0% (40 of 50, 66-90%). Both fall short of the predicted 90%,
  Wilson only just. Two explanations are likely, though I did not test them separately. A
  fitting half on which a 0.95 bound clears may be a lucky half, so conditioning on "found"
  selects samples that overstate. And "kept" is scored on the other half's own estimate from
  about 100 answered inputs, which misses often even when the true kappa sits just above 0.95.
  Unconditionally, the exact rule served a
  missing cascade in 4.0% of splits (16.0% x 25.0%), Wilson in 5.0% and the percentile rule
  in 18.0%. On those 238-input fitting halves the percentile rule's thresholds (median 36.6%
  coverage on the unseen half) were mostly ones the count rules could not certify. At 0.9 on Banking77 (gold), exact kept the target in 94.5% of splits
  against 90.5%, at a median unseen-half coverage of 79.4% against 80.2%.

**The default is now `rule: exact`, for both label counts.** It is the only rule whose
coverage of the agreement rate is guaranteed at every rate. Through the plug-in chance term it
also covers kappa within Monte Carlo error of 95% in every simulated cell (pooled 96.5%, worst
94.4%), and its chosen threshold
keeps the target in 96.7% of searches. At 77 labels it costs exactly what Wilson costs (the two
serve the same Banking77 threshold at both targets) and 1.4 points over the percentile rule.
The answer only differs between the two count rules at 4 labels. There Wilson is cheaper
(72.7% against 66.9% on AG News at 0.9), but it buys that by covering p_o as little as 92.5%
at some agreement rates, the shortfall this milestone set out to remove. `anvil calibrate
--rule wilson` keeps the cheaper rule for a user who accepts that, and `--rule lower` keeps
the M5-M10 behaviour. The model card names the rule its threshold was chosen with. The
committed bundle, its export and bench reports and the M9 shift study were measured on the
percentile threshold (0.6395), because the students are not committed and do not reproduce
bit for bit across CPUs. Re-exporting under the exact rule is under "To measure".

### ONNX export: fp32 and int8 parity on the trained students (M5)

- Command: `make eval-export TEACHER=gold` (`python -m eval.export <dataset> --teacher gold`,
  then `python -m eval.render`), on the same M2 students, every held-out input a parity probe,
  with the calibration from the section above in the bundle.
- Method: fp32 is `torch.onnx.export(dynamo=True)` with dynamic batch and sequence axes,
  compared with PyTorch logits on the padded batch and on 16 inputs one at a time, with the
  ONNX side tokenized by the serving tokenizer (`tokenizers`), so a tokenizer mismatch would
  fail the gate as well. int8 is ONNX Runtime's dynamic quantization (per tensor) with the MLP
  down-projections kept in fp32, compared with the fp32 graph. The probe quantizes one layer
  group at a time to see where the loss comes from.

Findings:

- **fp32 export is exact for serving purposes.** The largest logit difference over all
  held-out inputs is 2.3e-05 (Banking77) and 2.9e-05 (AG News), three to four times inside
  the 1e-4 gate, with identical argmax everywhere, so kappa is unchanged to three decimals.
- **The MLP down-projection is what int8 cannot take, now measured on a trained head.**
  Quantizing only those 22 MatMuls leaves 2.6% (Banking77) and 26% (AG News) of argmax
  decisions unchanged. Every other group alone keeps
  at least 96%, and quantizing everything is as bad as quantizing `mlp.Wo` alone. The M1 spike
  had guessed this from an untrained head; this table measures it. Keeping the
  down-projections in fp32 costs 59 MB (211 MB against 152 MB).
- **int8 flips only what the cascade escalates.** The shipped int8 graph (211 MB, against
  601 MB for fp32) agrees with fp32 on 96.4% of Banking77 inputs and 99.2% of AG News inputs,
  and on 100% of the inputs either graph answers at the served threshold (417 and 365). Kappa
  with the teacher moves by -0.002 and -0.006, both within noise. The M1 rule (98% agreement
  on every input) would have refused the Banking77 graph; the gate now counts agreement on
  the answered inputs, since a flip on an input both graphs escalate never reaches a caller
  (PLAN.md, M5). The first version of the gate took the answered inputs from fp32's
  confidence alone (408 and 361), which misses an input fp32 escalates but a served int8
  graph answers itself; counting either graph's answers adds 9 and 4 inputs and no flip.
  Since this run the gate also scores the first 16 probes served one at a time, as the fp32
  check already did: dynamic quantization scales activations per batch, so an input's int8
  logits depend on its batch mates (on the tiny test model the same text moves by 0.03
  between a batch and a single call), and a served request is often one input. The table's
  answered counts predate those runs; the re-run is under "To measure".
- Two things about the quantizer that the first version of this probe got wrong: it also
  quantizes the token embedding `Gather` by default, and it converts each `Gemm` before it
  reads its exclusion list, so the head and the classifier are quantized in every row. Both
  are now separate rows ("only embeddings", "only head and classifier") instead of being
  silently mixed into the others.

<!-- BEGIN generated:export -->
| Dataset | Student | Variant | Compared with | Probes | Max abs logit diff | Mean abs logit diff | Argmax agreement | Agreement on answered inputs | Kappa vs teacher (drop) | Size | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | ModernBERT-base | fp32 | pytorch | 477 | 2.9e-05 | 1.2e-06 | 1.000 | n/a | 0.844 (+0.000) | 601 MB | passed |
| ag_news | ModernBERT-base | int8 | onnx-fp32 | 477 | 1.2e+00 | 1.6e-01 | 0.992 | 1.000 (n=365) | 0.849 (-0.006) | 211 MB | passed |
| banking77 | ModernBERT-base | fp32 | pytorch | 496 | 2.3e-05 | 3.2e-06 | 1.000 | n/a | 0.842 (+0.000) | 601 MB | passed |
| banking77 | ModernBERT-base | int8 | onnx-fp32 | 496 | 2.4e+00 | 3.7e-01 | 0.964 | 1.000 (n=417) | 0.844 (-0.002) | 211 MB | passed |

#### ag_news: export

- Raw output: `eval/results/export-ag_news-gold/` (`report.json`, `meta.json`)
- Command: `python -m eval.export --teacher gold` at `e65c998` (uncommitted changes); student run `20261020-012041-n2000`; export and both parity checks 183 s; wall 2850 s; spent $0.00
- Gates: fp32 max |logit diff| <= 1e-04 and agreement == 1.0; int8 agreement with fp32 onnx on the probes fp32 or int8 answers at threshold 0.8336 >= 0.98 and kappa drop vs teacher <= 0.01 (22 MLP down-projections kept in fp32)
- int8 layer probe (nodes per group: embeddings 1, attention Wqkv 22, attention Wo 22, MLP Wi 22, MLP Wo 22, head and classifier 2); each row quantizes only the named group unless it says otherwise, plus the head and classifier Gemms, which the quantizer converts before it reads its exclusion list, and is compared with the fp32 graph on the same 477 held-out inputs (fp32 kappa vs teacher 0.844); answered inputs are those the fp32 or the int8 graph scores at or above the bundle's threshold:

| int8 variant | Quantized nodes | Size | Mean abs logit diff | Max abs logit diff | Argmax agreement | Agreement on answered inputs | Kappa drop |
|---|---|---|---|---|---|---|---|
| everything | 91 | 152 MB | 2.163 | 10.329 | 0.235 | 0.152 (n=361) | +0.804 |
| default: all but MLP Wo | 69 | 211 MB | 0.158 | 1.200 | 0.992 | 1.000 (n=365) | -0.006 |
| all but MLP Wo and embeddings | 68 | 327 MB | 0.153 | 0.817 | 0.992 | 1.000 (n=364) | -0.006 |
| only embeddings | 1 | 483 MB | 0.070 | 0.676 | 0.992 | 1.000 (n=364) | -0.006 |
| only attention Wqkv | 22 | 483 MB | 0.130 | 0.723 | 0.998 | 1.000 (n=362) | -0.003 |
| only attention Wo | 22 | 560 MB | 0.058 | 0.475 | 0.994 | 1.000 (n=362) | +0.003 |
| only MLP Wi | 22 | 483 MB | 0.074 | 0.728 | 0.992 | 1.000 (n=363) | -0.011 |
| only MLP Wo | 22 | 541 MB | 2.155 | 10.482 | 0.262 | 0.191 (n=361) | +0.765 |
| only head and classifier | 2 | 599 MB | 0.054 | 0.401 | 0.998 | 1.000 (n=362) | -0.003 |

#### banking77: export

- Raw output: `eval/results/export-banking77-gold/` (`report.json`, `meta.json`)
- Command: `python -m eval.export --teacher gold` at `e65c998` (uncommitted changes); student run `20261020-001737-n2000`; export and both parity checks 106 s; wall 3782 s; spent $0.00
- Gates: fp32 max |logit diff| <= 1e-04 and agreement == 1.0; int8 agreement with fp32 onnx on the probes fp32 or int8 answers at threshold 0.6395 >= 0.98 and kappa drop vs teacher <= 0.01 (22 MLP down-projections kept in fp32)
- int8 layer probe (nodes per group: embeddings 1, attention Wqkv 22, attention Wo 22, MLP Wi 22, MLP Wo 22, head and classifier 2); each row quantizes only the named group unless it says otherwise, plus the head and classifier Gemms, which the quantizer converts before it reads its exclusion list, and is compared with the fp32 graph on the same 496 held-out inputs (fp32 kappa vs teacher 0.842); answered inputs are those the fp32 or the int8 graph scores at or above the bundle's threshold:

| int8 variant | Quantized nodes | Size | Mean abs logit diff | Max abs logit diff | Argmax agreement | Agreement on answered inputs | Kappa drop |
|---|---|---|---|---|---|---|---|
| everything | 91 | 152 MB | 1.900 | 18.575 | 0.020 | 0.025 (n=408) | +0.838 |
| default: all but MLP Wo | 69 | 211 MB | 0.366 | 2.393 | 0.964 | 1.000 (n=417) | -0.002 |
| all but MLP Wo and embeddings | 68 | 327 MB | 0.343 | 1.950 | 0.964 | 1.000 (n=416) | -0.002 |
| only embeddings | 1 | 483 MB | 0.154 | 1.157 | 0.986 | 1.000 (n=412) | +0.002 |
| only attention Wqkv | 22 | 483 MB | 0.278 | 1.548 | 0.964 | 1.000 (n=418) | -0.004 |
| only attention Wo | 22 | 560 MB | 0.119 | 0.633 | 0.986 | 1.000 (n=411) | +0.002 |
| only MLP Wi | 22 | 483 MB | 0.200 | 1.541 | 0.974 | 0.998 (n=415) | +0.004 |
| only MLP Wo | 22 | 541 MB | 1.896 | 18.449 | 0.026 | 0.029 (n=408) | +0.832 |
| only head and classifier | 2 | 600 MB | 0.081 | 0.450 | 0.988 | 1.000 (n=408) | +0.000 |
<!-- END generated:export -->

### Serving speed and cost: Banking77 (M5)

- Command: `make bench TEACHER=gold` (`python -m eval.bench --teacher gold`, then
  `python -m eval.render`), on the Banking77 bundle from `make eval-export`, with nothing else
  running on the machine.
- Method: the 496 held-out messages (mean 11.7 words), cycled; 20 warm-up calls, then 400
  timed calls at batch 1 and 100 at batch 32 per engine, every engine at 4 intra-op threads.
  The HTTP rows call a real `anvil serve` subprocess (uvicorn, teacher off) over localhost.
  CPU seconds are measured as user plus system time of the process doing the work (this one,
  or the server's `/proc/<pid>/stat`). Dollars use an assumed price of $0.05 per vCPU-hour;
  pass `--vcpu-hour-usd` for your own.

Findings:

- **ONNX Runtime halves PyTorch's latency, and int8 takes off another third.** At batch 1 the
  p50 is 55.3 ms in PyTorch, 24.6 ms in ONNX Runtime fp32 and 16.2 ms in int8. The server adds
  2-3 ms (27.6 and 18.1 ms over HTTP). With four concurrent clients one 4-core replica answers
  36 inputs/s in fp32 and 55 in int8.
- **On a CPU, batching makes it worse.** Batch 32 costs more per input than batch 1 for every
  engine: 232 against 110 CPU seconds per 1,000 inputs for ONNX fp32. A batch is padded to
  its longest input and the encoder is compute-bound, so the padding is real work; batching
  only pays when the hardware is idle between requests (a GPU) or when inputs are grouped by
  length, which the server does not do yet.
- **Cost.** At the cheaper batch size the served int8 student uses 81 CPU seconds per 1,000
  inputs, $0.0011 at the assumed price. The teacher side of the comparison (Claude latency and
  dollars per 1,000 inputs) needs a key and is under "To measure"; at the served threshold the
  cascade sends 17.7% of Banking77 inputs to it.

<!-- BEGIN generated:bench -->
#### banking77: student `ModernBERT-base`, 4 threads

- Raw output: `eval/results/bench-banking77-gold/` (`report.json`, `meta.json`)
- Command: `python -m eval.bench --teacher gold` at `d0cfa4c`; AMD EPYC 7763 64-Core Processor, 4 CPUs; inputs: the 496 held-out texts (mean 11.7 words), cycled; wall 1167 s

| Engine | Batch 1: p50 / p95 ms | Batch 1: inputs/s | Batch 32: p50 / p95 ms | Batch 32: inputs/s | CPU seconds per 1k inputs (batch 1 / 32) | $ per 1k at $0.05/vCPU-hour, cheaper batch |
|---|---|---|---|---|---|---|
| PyTorch (in process) | 55.3 / 92.2 | 17 | 2423.2 / 3544.3 | 13 | 236.1 / 260.0 | 0.00328 |
| ONNX Runtime fp32 (in process) | 24.6 / 53.6 | 36 | 1748.6 / 2610.8 | 17 | 109.6 / 232.5 | 0.00152 |
| `anvil serve` fp32 (HTTP) | 27.6 / 55.4 | 33 | 1754.7 / 2621.3 | 17 | 119.8 / 232.9 | 0.00166 |
| ONNX Runtime int8 (in process) | 16.2 / 35.4 | 54 | 1178.2 / 1763.6 | 25 | 73.6 / 157.2 | 0.00102 |
| `anvil serve` int8 (HTTP) | 18.1 / 38.0 | 49 | 1180.7 / 1754.6 | 25 | 80.8 / 156.7 | 0.00112 |

Four concurrent HTTP clients at batch 1:

- `anvil serve` fp32 (HTTP): 36 inputs/s, p50 / p95 99.2 / 186.7 ms
- `anvil serve` int8 (HTTP): 55 inputs/s, p50 / p95 65.5 / 132.6 ms

The CPU price is an assumption (`--vcpu-hour-usd`), not a measurement; the CPU seconds are measured. Teacher `gold`: no recorded Claude labels for this dataset yet, so its cost per 1k is under To measure; its latency needs `--teacher-live N` with a key (To measure).
At the bundle's threshold (0.6395, target kappa 0.9, rule `lower`) the student answers 82.3% of held-out inputs, so the cascade calls the teacher for 17.7% of traffic.
<!-- END generated:bench -->

### The cascade off-distribution: Banking77 (M9)

- Command: `make eval-shift TEACHER=gold` (`python -m eval.shift --teacher gold`, then
  `python -m eval.render`). It scores the exported fp32 bundle through `anvil/runtime.py`,
  as `anvil serve` does, on three sets pinned by revision in `eval/shift.yaml` (licences in
  the table). Everything after the forward passes re-runs from the committed `scores.npz`
  in seconds (`--scores`).
- Which bundle: the eval workspace had to be rebuilt, and retraining the Banking77 student
  on this machine (AMD EPYC 9V74) did not reproduce the M5 student (trained on an EPYC
  7763). The two runs share the data, seed and code, and their losses agree to six digits
  at step 1 (gradient norm 39.770115 against 39.770176), then drift apart; early stopping
  kept epoch 5 here and epoch 8 there. Float kernels differ between CPU generations, so
  "bit for bit" (M5) holds on one machine only. This bundle has temperature 1.28, threshold
  0.691 and 78.6% held-out coverage, against 1.24, 0.639 and 82.3% for M5's. The
  hypotheses below are read against this bundle's own held-out numbers.
- Method: the served decision is "answer when the calibrated max softmax reaches the
  threshold". Every input of CLINC150's out-of-scope test set and of the AG News sample that
  clears it gets a Banking77 intent, so it is wrong by construction. Four other scores come
  from the same request: max logit, energy (logsumexp of the logits), cosine similarity to
  the 5th nearest of the 2,000 training embeddings, and the Mahalanobis distance to the
  nearest class mean (Ledoit-Wolf covariance). The embeddings are the mean-pooled last
  hidden layer of the PyTorch student, because the bundle returns logits only. Each score is
  compared by AUROC against the Banking77 test split (1,000 bootstrap resamples, paired
  across scores), and as a second gate. The gate's cut is set on the 496 held-out inputs to
  give up 3% of them, then read on the test split and the other sets.

Findings:

- **The threshold transfers to the dataset's own test split.** On 3,079 test inputs the
  student answers 78.4% [76.9%, 79.8%] (held-out: 78.6%) at kappa 0.946 [0.937, 0.956] with
  gold, so the 0.9 target holds with room to spare. The test kappa is higher than the
  held-out 0.922 because the lower-bound rule is conservative: it stops where the bound, not
  the estimate, reaches 0.9. Near-copies of training rows do not explain the gap: 182 test
  inputs share at least 80% of their words with a sampled training text and are answered
  more often (93.4%), but the other 2,897 give the same kappa (0.945). Temperature scaling
  holds too (ECE 0.038 at T = 1, 0.023 at the bundle's temperature).
- **The cascade escalates almost everything that is not banking; the 40% hypothesis was
  wrong by more than an order of magnitude.** The student answers 2.6% [1.6%, 3.6%] of
  CLINC150's out-of-scope requests and 0.7% of AG News items. With 77 intents, an input
  that fits none spreads its probability mass: 80% of out-of-scope requests get a top
  probability below 0.3, against a 0.691 threshold that the kappa target forced up. A
  threshold chosen for agreement on hard in-distribution inputs is also a decent
  out-of-distribution filter. Whether that holds for a 4-way student (AG News), where an
  unrelated input needs far less mass to clear a threshold, is not measured here.
- **What does get through is adjacent, not random.** The out-of-scope requests answered most
  confidently are the ones a bank customer could write: "i need to update my address" goes
  to `edit_personal_details` at 0.990, "how much of a refund will i get this year" (a tax
  refund) to `refund_not_showing_up`, and a kindergarten enrolment question to `age_limit`.
  Six of the seven news items that clear the threshold go to
  `card_payment_wrong_exchange_rate`; the most confident are interest-rate and price stories. The risk to plan for is near-domain traffic
  that shares the label's words, not gibberish.
- **Energy and max logit are better signals than max softmax, but only a little; embedding
  distances are no better.** AUROC against CLINC out-of-scope: max softmax 0.969, max logit
  0.982, energy 0.985 (+0.016 [+0.013, +0.019] paired). Against AG News: 0.981, 0.992,
  0.995. The intervals exclude zero, but the gain is a third of the 0.05 the hypothesis
  asked for. The gain shows more in the tail: at 95% of test inputs kept, energy lets
  through 5.5% of out-of-scope requests against 15.6% for max softmax. The training-
  embedding distances are level with max softmax on CLINC (k-NN -0.004 [-0.009, +0.001],
  Mahalanobis +0.006 [+0.000, +0.011]) and clearly worse on AG News (-0.024 and -0.040).
  The vector is the mean-pooled last layer, which is the classification head's input (the
  student pools by mean); the head's own output (dense, GELU, LayerNorm), the layer Lee et
  al. use, was not tried. On that vector, the logits carry the out-of-distribution signal
  better than the pooled hidden state does.
- **As a gate, energy halves the leak at the same coverage cost, from a small base.** Spending
  about 3 points of held-out coverage on a stricter max-softmax threshold takes CLINC
  acceptance from 2.6% to 2.0%; spending it on an energy cut takes it to 1.1% (test cost 2.6
  points). That is 9 fewer wrong answers per 1,000 out-of-scope requests. The served rule
  stays max softmax: the gain is real but small, and energy needs its own cut and a second
  calibration step that `anvil calibrate` does not have.
- **Hypotheses:** H1 held (coverage moved 0.3 points, kappa 0.946). H2 was refuted on both
  sets (2.6% and 0.7%, not 40% or more). H3 was refuted: no free score gains 0.05 AUROC, and
  energy's +0.016 is the best. The model card now says that its coverage was measured in
  distribution (`anvil/hub.py`, `IN_DISTRIBUTION_NOTE`).

<!-- BEGIN generated:shift -->
#### banking77, teacher `gold`, student `ModernBERT-base`: the served bundle off-distribution

- Raw output: `eval/results/shift-banking77-gold/` (`report.json`, `meta.json`, `scores.npz` with every input's logits and distance scores)
- Command: `python -m eval.shift --teacher gold` at `773e679`; student run `20261022-083846-n2000` (2000 training inputs); bundle `fp32` through `anvil/runtime.py` at temperature 1.281 and threshold 0.6907; 4 CPUs; wall 10m 59s; spent $0.00
- Re-analysis from the committed scores, no model: `python -m eval.shift --scores eval/results/shift-banking77-gold/scores.npz`

| Set | Role | Source (pinned) | Licence | Inputs | Answered by the student [95% CI] | Mean confidence | Most frequent answers |
|---|---|---|---|---|---|---|---|
| banking77_test | in distribution | `legacy-datasets/banking77`@`f541215` test | CC-BY-4.0 (Casanueva et al., 2020, PolyAI) | 3079 | 78.4% [76.9%, 79.8%] | 0.839 | - |
| clinc_oos | out of distribution | `clinc/clinc_oos`@`155b9c7` plus/test, label `oos` | CC-BY-3.0 (Larson et al., 2019) | 1000 | 2.6% [1.6%, 3.6%] | 0.227 | `automatic_top_up` 4, `age_limit` 4, `refund_not_showing_up` 3, `edit_personal_details` 2, `card_payment_wrong_exchange_rate` 2 |
| ag_news | out of distribution | `fancyzhx/ag_news`@`eb185aa` test | AG's corpus, used under its academic terms (http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) | 1000 | 0.7% [0.2%, 1.3%] | 0.189 | `card_payment_wrong_exchange_rate` 6, `apple_pay_or_google_pay` 1 |

In distribution: the threshold chosen on 496 held-out inputs, read on 3079 banking77_test inputs (kappa and ECE against the gold labels).

| | Held-out (threshold chosen here) | Test split |
|---|---|---|
| Coverage | 78.6% | 78.4% [76.9%, 79.8%] |
| Kappa on answered inputs [95% CI] | 0.922 | 0.946 [0.937, 0.956] |
| Accuracy on answered inputs | | 94.7% |
| Kappa on all inputs | | 0.851 |
| ECE (15 bins), T = 1 -> bundle temperature | | 0.038 -> 0.023 |
| Test inputs that are not near-copies of a training-sample text | | 2897: coverage 77.4%, kappa on answered 0.945 [0.935, 0.955] |
| Test inputs that are near-copies | | 182: coverage 93.4% |

Test inputs identical to one of the 2500 sampled training rows: 2; near-duplicates (word Jaccard >= 0.8): 182.

Served confidence (max softmax at the bundle temperature), share of each set per bin; the threshold is 0.691:

| Confidence | banking77_test | clinc_oos | ag_news |
|---|---|---|---|
| 0.0-0.1 | 0.0% | 11.0% | 17.1% |
| 0.1-0.2 | 1.3% | 46.9% | 52.2% |
| 0.2-0.3 | 2.6% | 22.0% | 17.8% |
| 0.3-0.4 | 3.7% | 9.7% | 6.6% |
| 0.4-0.5 | 4.7% | 3.9% | 3.2% |
| 0.5-0.6 | 4.7% | 2.8% | 1.6% |
| 0.6-0.7 | 4.9% | 1.1% | 0.8% |
| 0.7-0.8 | 5.0% | 1.2% | 0.4% |
| 0.8-0.9 | 10.5% | 0.7% | 0.3% |
| 0.9-1.0 | 62.5% | 0.7% | 0.0% |

Separating banking77_test from each other set: AUROC (in-distribution inputs scoring higher) with 95% bootstrap intervals, its paired difference from max softmax, and the share of the other set at or above the score that keeps 95% of the test split (FPR at 95% TPR):

| Score | clinc_oos: AUROC | clinc_oos: minus max softmax | clinc_oos: FPR95 | ag_news: AUROC | ag_news: minus max softmax | ag_news: FPR95 |
|---|---|---|---|---|---|---|
| max softmax (served) | 0.969 [0.964, 0.974] | - | 15.6% | 0.981 [0.978, 0.985] | - | 9.8% |
| max logit | 0.982 [0.978, 0.986] | +0.013 [+0.011, +0.015] | 8.0% | 0.992 [0.990, 0.994] | +0.011 [+0.009, +0.013] | 3.5% |
| energy | 0.985 [0.981, 0.988] | +0.016 [+0.013, +0.019] | 5.5% | 0.995 [0.993, 0.996] | +0.013 [+0.011, +0.016] | 1.8% |
| k-NN cosine to training embeddings | 0.964 [0.959, 0.970] | -0.004 [-0.009, +0.001] | 19.7% | 0.958 [0.952, 0.963] | -0.024 [-0.029, -0.018] | 31.0% |
| Mahalanobis to class means | 0.974 [0.969, 0.979] | +0.006 [+0.000, +0.011] | 12.6% | 0.941 [0.935, 0.948] | -0.040 [-0.046, -0.033] | 40.8% |

Each score as a second gate on top of the served threshold, its cut set to give up 3% of the held-out inputs, then read on the test split and the other sets:

| Gate | Held-out coverage | Test coverage (cost) | Test kappa on answered | clinc_oos answered | ag_news answered |
|---|---|---|---|---|---|
| served threshold only | 78.6% | 78.4% | 0.946 | 2.6% | 0.7% |
| + max softmax (served) | 75.8% | 75.6% (2.8 points) | 0.954 | 2.0% | 0.6% |
| + max logit | 75.8% | 76.3% (2.1 points) | 0.949 | 1.3% | 0.3% |
| + energy | 75.8% | 75.8% (2.6 points) | 0.950 | 1.1% | 0.2% |
| + k-NN cosine to training embeddings | 75.8% | 76.2% (2.1 points) | 0.950 | 1.9% | 0.6% |
| + Mahalanobis to class means | 75.8% | 75.3% (3.1 points) | 0.951 | 1.2% | 0.5% |

The clinc_oos inputs the student is surest of (each answered with a banking77 label):

- "i need to update my address" -> `edit_personal_details` (0.990)
- "is there a way to change your houses address" -> `edit_personal_details` (0.989)
- "tell me where to go to get my son enrolled in public kindergarten" -> `age_limit` (0.985)
- "how long do i have to wait to get my refund" -> `refund_not_showing_up` (0.984)
- "how much of a refund will i get this year" -> `refund_not_showing_up` (0.973)

The ag_news inputs the student is surest of (each answered with a banking77 label):

- "US rates seen headed up despite soft-spot worries Federal Reserve policy-makers were expected to raise US i..." -> `card_payment_wrong_exchange_rate` (0.899)
- "Producer Price Surge Fuels Inflation Fears Producer prices surged 1.7 percent in October, their sharpest mo..." -> `card_payment_wrong_exchange_rate` (0.892)
- "New Zealand bank raises key interest rate WELLINGTON New Zealand #39;s central bank raised the benchmark in..." -> `card_payment_wrong_exchange_rate` (0.850)
- "Apple beefs up .Mac storage limits Customers of Apple #39;s .Mac Internet service can hit the delete button..." -> `apple_pay_or_google_pay` (0.777)
- "Crude oil prices fall to their lowest level in seven weeks Crude oil futures closed below 48 dollars a barr..." -> `card_payment_wrong_exchange_rate` (0.774)

Hypotheses (PLAN.md, M9), decided by the numbers above:

- H1 **held**: on the test split coverage stays within 3 points of the held-out 78.6% (moved 0.3 points) and kappa on the answered inputs stays at or above 0.9 (0.946).
- H2 **refuted** for clinc_oos: the student answers at least 40% above the served threshold (2.6%).
- H2 **refuted** for ag_news: the student answers at least 40% above the served threshold (0.7%).
- H3 **refuted**: a free score beats max softmax by at least 0.05 AUROC on every other set while its gate costs under 3 points of test coverage (scores that do: none).

The distance scores use the PyTorch student's embeddings; its probabilities differ from the ONNX bundle's by at most 8.9e-05 on any input scored.
<!-- END generated:shift -->

### Generation parity: WikiSQL text-to-SQL (M6)

- Command: `make eval-generate TEACHER=gold` (`python -m eval.generate wikisql --teacher gold`,
  then `python -m eval.render`).
- Method: 1,340 distinct questions drawn (seed 0) from WikiSQL's train split at a pinned
  parquet commit, tables of at most 12 columns; the store's hash split holds out about a
  quarter, of which 300 are scored. Each input is the table's column names and types and the
  question; the answer is one SQLite query over a table named `t`. The student is
  Qwen2.5-0.5B-Instruct with a rank-16 LoRA adapter on every linear layer, trained on 1,000
  teacher queries in its own chat template (prompt tokens masked), 10% held back to pick the
  epoch by greedy exact match, and decoded greedily. Both queries run in a sandboxed SQLite
  copy of the question's table (text compared case-insensitively, as WikiSQL's evaluator
  does); intervals resample questions. Execution match is also reported on the table with
  every row doubled, which separates queries that only agree because the table has one
  matching row.

<!-- BEGIN generated:generate -->
| Dataset | Teacher | Student | Teacher labels | Held-out | Execution match [95% CI] | Exec. match, rows doubled too | Exact SQL | Exec. match, non-empty reference | Zero-shot base [95% CI] | Student minus zero-shot [95% paired CI] | Teacher ceiling | Merged export | Label cost | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wikisql | gold | Qwen2.5-0.5B-Instruct | 1000 | 300 | 0.767 [0.717, 0.813] | 0.743 [0.693, 0.793] | 0.490 | 0.763 | 0.210 [0.167, 0.253] | +0.557 [0.497, 0.617] | n/a | 64/64 identical, max diff 8.8e-05 | $0.00 | 103 min |

#### wikisql, teacher `gold`, student `Qwen2.5-0.5B-Instruct`

- Raw output: `eval/results/generate-wikisql-gold/` (`report.json`, `meta.json`, `train-metrics.jsonl`)
- Command: `python -m eval.generate wikisql --teacher gold` at `795cc7d`; seed 0, 1000 bootstrap resamples (by question); 4 CPUs (cpu); wall 6201 s; spent $0.0000
- Rescored: `python -m eval.generate wikisql --teacher gold --rescore --out eval/results/generate-wikisql-gold` at `bce6485` (the stored queries scored again by the current scorer; training and decoding not repeated)
- Data: Salesforce/wikisql@48cfb60 `default/train/0000.parquet` (WikiSQL (Zhong et al., 2017, Salesforce); BSD-3-Clause in the GitHub repo, the Hub card lists it as unknown), 1340 questions sampled (seed 0, tables of at most 12 columns, sha256 8220f6e980c2); held-out inputs (column names and question) sharing at least 80% of their words with a pool input: 7
- Training: LoRA r=16, alpha=32 on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj; 892 examples and 108 held back to pick the epoch (greedy exact match), 2 epochs, 224 steps, best epoch 2, 5031 s; chat template: 1000 answers aligned, 0 dropped, longest sequence 288 tokens
- Decoding: greedy, 1318 ms per held-out question (batches of 16, length-sorted)
- Execution match on the 270 questions whose reference finds rows: 0.763; references that find nothing (no rows, an aggregate over no matching rows: NULL, or 0 from COUNT): 30 of 300 (any query that also finds nothing matches them, and doubling the rows cannot tell them apart)
- Coincidental matches: 7 queries match on the table but not once every row is doubled (a missing aggregate over one row, a count that equals a cell)
- Execution vs exact match: 83 queries match by result but differ as SQL, 0 the other way; with string literals compared case-insensitively (as the sandbox compares text) exact SQL is 0.630 and 42 differ; student execution errors: OperationalError 4
- The teacher here is WikiSQL's own query for each question (no API key in this run), so this row measures the harness and how well the student learns from 1,000 clean labels; it is not a parity-with-Claude number.
- Zero-shot base model (no adapter, same prompt): execution match 0.210 [0.167, 0.253], exact SQL 0.003, 0 answers wrapped in a code fence; errors: OperationalError 176
- Execution match by aggregate: none: 0.79 (n=209), aggregate: 0.71 (n=91)
- Execution match by WHERE conditions: 0: 1.00 (n=3), 1: 0.78 (n=202), 2: 0.74 (n=76), 3+: 0.68 (n=19)
- Execution match by input length (words): <= 26: 0.79 (n=92), 27-30: 0.72 (n=64), 31-35: 0.80 (n=79), > 35: 0.74 (n=65)
- Judge sample: skipped (no recorded verdicts at eval/labels/wikisql/judge-claude-haiku-5-5-9e7b905fc21cd29e.jsonl and no ANTHROPIC_API_KEY)
- Merged export: 64/64 held-out completions identical to the adapter's, max |logit diff| 8.8e-05 on 8 prompts; 1988 MB, 8,798,208 adapter parameters folded in
<!-- END generated:generate -->

What the numbers say (all from the table above):

- 1,000 labels and two CPU epochs of LoRA take the 0.5B model from 0.210 to 0.767 execution
  match (+0.557, paired interval [0.497, 0.617]). Most of the zero-shot loss is queries that
  do not run (176 of 300 raise an error), not wrong answers.
- Exact SQL (0.490) understates the student: 83 queries return the same rows as the
  reference but are written differently, and none goes the other way. Most of that is the
  case of one string literal (WikiSQL writes `'sunbeam'` in one query and `'North Island'`
  in another, and the sandbox compares text case-insensitively): with literals compared
  that way, exact SQL is 0.630 and 42 queries are real rewrites (reordered conditions, a
  different quoting of the same value).
- 30 of the 300 references (10%) find nothing in their table: 22 are an aggregate over no
  matching rows, which returns one row holding `NULL` (WikiSQL's own `SELECT MIN("Races")
  FROM t WHERE "Podiums" < 2 AND "FLaps" > 0 AND "Season" > 2008`, where no row qualifies,
  so `MIN` matches `AVG`), and 8 are a `COUNT` over no matching rows, which returns 0 and
  matches any `COUNT` with a wrong filter. Doubling the rows cannot expose either. The first
  scoring counted only zero-row results as empty and reported all 300 references as
  non-empty; the scorer now runs each reference on a copy of its table with no rows and
  calls it empty when the result is the same. That leaves 270 references with evidence,
  on which the student scores 0.763. The stored queries were rescored (`--rescore`) instead
  of being trained again.
- Doubling every row breaks 7 matches (0.767 to 0.743): the reference and the student agree
  only because the table has one matching row.
- A known limit of the gold rendering: WikiSQL keeps comma-formatted numbers (`'55,045'`)
  as text, so a gold `>` or `<` against such a cell compares strings, not numbers, and the
  student's numeric form (`> 55,045`, a syntax error) is counted wrong. Both sides run the
  same engine, so the parity number is self-consistent, but on those questions the
  reference is not the question's answer.

## To measure

| What | Command | Needs | Estimated cost |
|---|---|---|---|
| Live teacher test and the replay recordings (8 toy inputs) | `make record` | `ANTHROPIC_API_KEY` | under $0.01 with `claude-haiku-5-5` |
| Student vs Claude on the toy task (91 held-out + 200 pool labels) | set `teacher.model` in `examples/toy/anvil.yaml`, then `anvil label -i inputs.jsonl -n 200 && anvil train && anvil eval` from `examples/toy/` | `ANTHROPIC_API_KEY` | about $0.04 with Haiku 5.5, about $1.60 with Opus 5.5 (300 labels at roughly 300 input and 200 output tokens each) |
| Parity with Claude on AG News and Banking77: student vs `claude-haiku-5-5` kappa with CIs, the teacher's self-agreement ceiling (200 items, paraphrased prompt) and teacher vs gold. Labels are written to `eval/labels/` so later runs replay them offline | `make eval-classify` (with `ANTHROPIC_API_KEY` set; `TEACHER=auto` then calls Claude once and replays afterwards) | key; about 1.5 h CPU for the two students | about $0.35 with Haiku 5.5 effort `low` (5,400 labels at ~180 input tokens for AG News, ~540 for Banking77 with its 77 label names, ~50 output tokens each); under $1 if thinking doubles the output |
| Extraction parity with Claude on MIT Restaurant: entity F1 and token kappa against `claude-haiku-5-5`, the alignment report on real Claude offsets (how many needed repair, how many were lost), the paraphrase ceiling (200 sentences) and teacher vs gold | `make eval-extract` with `ANTHROPIC_API_KEY` set (`TEACHER=auto` calls Claude once and replays afterwards) | key; about 15 min CPU | about $0.30 with Haiku 5.5 effort `low` (2,200 requests at ~350 input tokens and ~170 output tokens); under $0.60 if thinking doubles the output |
| The frontier with Claude Haiku as the teacher (both datasets, every implemented strategy) | `make eval-classify` with `ANTHROPIC_API_KEY` set records the labels, then `make eval-frontier` replays them | key; about 2.5 h CPU per strategy | $0 beyond the `make eval-classify` labels: the frontier uses the same 2,500 inputs and prompt per dataset |
| The `uncertainty`, `margin` and `diverse` curves, synthetic and text | write them in `anvil/select.py` (the tests in `tests/test_select.py` turn from expected failures into passes), then `make eval-select` and `make eval-frontier TEACHER=gold` | about 2.5 h CPU per strategy for the text frontier; seconds for the synthetic study | $0 |
| Teacher latency and cost per 1,000 inputs next to the student in the bench table, and the cascade's cost at its measured coverage | `make eval-classify` with a key (records the Haiku labels and their cost), then `uv run --group eval python -m eval.bench --teacher-live 50` | key | the eval's ~$0.35, plus about $0.01 for 50 uncached timed Haiku calls |
| Calibration and the cascade against Claude labels (does the 0.9 threshold hold when the teacher is a model, not gold?) | after the keyed `make eval-classify`, `make eval-calibration eval-export` (`TEACHER=auto` replays the recorded labels; the students are retrained on them, about 1.5 h CPU) | key for the labels | $0 beyond the `make eval-classify` labels |
| The export table with the int8 gate as it now stands (the first 16 probes also served one at a time; the committed run scored batched probes only) and the bench on that bundle | `make eval-export TEACHER=gold`, then `make bench TEACHER=gold` | about 2 h CPU for the two exports with the layer probe, 20 min for the bench | $0 |
| Generation parity with Claude on WikiSQL: execution match of the Qwen2.5-0.5B LoRA student against `claude-haiku-5-5`'s queries, the paraphrase ceiling (100 questions) and teacher vs WikiSQL's own queries. Labels go to `eval/labels/wikisql/` and replay afterwards | `make eval-generate` with `ANTHROPIC_API_KEY` set (`TEACHER=auto` calls Claude once) | key; about 2 h CPU, or the Colab notebook on a T4 | about $0.10 with Haiku 5.5 effort `low` (1,400 requests at ~250 input and ~60 output tokens); under $0.25 if thinking triples the output |
| The 100-item judge sample: `claude-haiku-5-5` judges whether the student's query answers the question as the teacher's does, compared with execution match item by item (how often a small table makes execution match wrong) | any `make eval-generate` run with `ANTHROPIC_API_KEY` set; the verdicts are written to `eval/labels/wikisql/judge-*.jsonl` and replay for free | key | about $0.01 (100 requests at ~500 input and ~80 output tokens) |
| The generation run on a GPU (fp16 autocast on a T4): wall time and whether mixed precision changes the measured parity | `notebooks/anvil_gpu.ipynb` on Colab, then commit the downloaded `eval/results/generate-*` | a Colab T4 | $0 on the free tier |
| The shift study on a 4-label student: does a threshold that filters out-of-scope inputs for 77 intents still filter them for AG News's 4 topics? | copy `eval/shift.yaml` with `dataset: ag_news`, AG News's test split as the in-distribution set and CLINC150 out-of-scope plus Banking77's test split as the others, then `uv run --group eval python -m eval.shift --teacher gold --config <file>` | about 1 h CPU for the AG News student if the workspace lacks it, 15 min for the study | $0 |
| The export, bench and shift reports on the M11 threshold (the committed bundle carries the percentile rule's 0.6395; the exact rule serves 0.6635 on the same logits) | `make eval-calibration TEACHER=gold` (retrains the students if the workspace lacks them, writing the exact rule's `calibration.json`), then `make eval-export TEACHER=gold`, `make bench TEACHER=gold` and `make eval-shift` | about 4.5 h CPU | $0 |
| A third training seed on both datasets (seeds 0 and 1 are in the table above) | `uv run --group eval python -m eval.classify ag_news banking77 --teacher gold --seed 2`, then `make results` | about 1.5 h CPU (2 ModernBERT-base runs) | $0 |

The toy task is synthetic and only exercises the pipeline; real parity numbers come from the
M2 datasets.
