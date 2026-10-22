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
| `calibration-ag_news-gold` | eval-calibration | gold | `python -m eval.calibration --teacher gold` | `b127db2` | 2m 02s | $0.00 |
| `calibration-ag_news-qwen2.5-3b` | eval-calibration | qwen2.5:3b | `python -m eval.calibration --teacher ollama` | `ef873d0` | 1m 45s | $0.00 |
| `calibration-banking77-gold` | eval-calibration | gold | `python -m eval.calibration --teacher gold` | `b127db2` | 0m 58s | $0.00 |
| `calibration-banking77-qwen2.5-3b` | eval-calibration | qwen2.5:3b | `python -m eval.calibration --teacher ollama` | `ef873d0` | 0m 58s | $0.00 |
| `classify-ag_news-gold` | eval-classify | gold | `python -m eval.classify ag_news banking77 --teacher gold` | `74cca5e` | 36m 10s | $0.00 |
| `classify-ag_news-gold-seed1` | eval-classify | gold | `python -m eval.classify ag_news --teacher gold --seed 1` | `1c900f7` | 47m 59s | $0.00 |
| `classify-ag_news-qwen2.5-3b` | eval-classify | qwen2.5:3b | `python -m eval.classify ag_news banking77 --teacher ollama` | `e25575f` | 41m 06s | $0.00 |
| `classify-banking77-gold` | eval-classify | gold | `python -m eval.classify banking77 --teacher gold` | `d4f4016` | 56m 31s | $0.00 |
| `classify-banking77-gold-seed1` | eval-classify | gold | `python -m eval.classify banking77 --teacher gold --seed 1` | `7c205b0` | 45m 20s | $0.00 |
| `classify-banking77-qwen2.5-3b` | eval-classify | qwen2.5:3b | `python -m eval.classify ag_news banking77 --teacher ollama` | `e25575f` | 1h 07m 55s | $0.00 |
| `export-ag_news-gold` | eval-export | gold | `python -m eval.export --teacher gold` | `e65c998` (uncommitted changes) | 47m 30s | $0.00 |
| `export-banking77-gold` | eval-export | gold | `python -m eval.export --teacher gold` | `e65c998` (uncommitted changes) | 1h 03m 02s | $0.00 |
| `extract-mit_restaurant-gold` | eval-extract | gold | `python -m eval.extract mit_restaurant --teacher gold` | `7dace8b` (uncommitted changes) | 15m 01s | $0.00 |
| `frontier-ag_news-gold` | eval-frontier | gold | `python -m eval.frontier --teacher gold` | `f929d74` | 1h 27m 18s | $0.00 |
| `frontier-banking77-gold` | eval-frontier | gold | `python -m eval.frontier banking77 --teacher gold` | `278462c` | 27m 58s | $0.00 |
| `generate-wikisql-gold` | eval-generate | gold | `python -m eval.generate wikisql --teacher gold` | `795cc7d` | 1h 43m 21s | $0.00 |
| `noise-ag_news-qwen2.5-3b` | eval-noise | qwen2.5:3b | `python -m eval.noise ` | `e25575f` | 1h 17m 14s | $0.00 |
| `noise-banking77-qwen2.5-3b` | eval-noise | qwen2.5:3b | `python -m eval.noise ` | `e25575f` | 1h 11m 30s | $0.00 |
| `select-synthetic` | eval-select | n/a | `python -m eval.select_study` | `c1a7a44` | 0m 25s | $0.00 |
| `spike-onnx-export` | spike-onnx | n/a | `uv run python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export` | `94389b5` | 5m 10s | $0.00 |
| **Total (21 runs)** | | | | | 13h 38m 36s | $0.00 |

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
  `anvil calibrate` uses the lower bound by default. It still falls short of the nominal 95%,
  and the gap is mostly in what the study scores: whether the kappa realised on about 240
  fresh inputs clears the target, a statement about one noisy sample rather than the
  population statement a lower confidence bound makes, so a threshold whose true kappa sits
  just above 0.9 misses on some fresh halves. The percentile bootstrap's known optimism for a
  statistic bounded at 1 takes the last point or two. The shares also move by a few points
  with the study seed (`--seed`), so read them as about nine in ten against about one in two.
- **Temperature scaling makes the confidences honest but buys no coverage.** Both students
  are slightly overconfident (T 1.24 and 1.22); on the unseen halves scaling cut the median ECE
  from 0.070 to 0.054 (Banking77) and 0.051 to 0.047 (AG News). Coverage and hit rate with
  uncalibrated confidences are the same within noise (Banking77 at 0.9: 87.9% and 53.0%
  uncalibrated, 87.1% and 55.5% calibrated), because the threshold only uses the order of
  the confidences, which one temperature barely changes. Calibration matters for reading a
  confidence as a probability, not for the cascade's coverage.
- **The product number.** At the served setting (target 0.9, lower bound) the Banking77
  student answers 82.3% of held-out inputs at kappa 0.923 (lower bound 0.900) and escalates
  the other 17.7%; for
  0.95 it answers 70.8%. AG News, whose student has almost the same overall kappa (0.844
  against 0.842), keeps less
  (75.7% and 41.9%): its errors are spread further up the confidence range (agreement on its
  most confident half is 0.971, against 0.992 for Banking77), so a strict target has to give
  up more inputs to reach it. The same overall kappa can mean very different cascades.

<!-- BEGIN generated:calibration -->
| Dataset | Teacher | Student | Held-out | Kappa (all inputs) | Temperature | ECE before -> after (all) | ECE before -> after (unseen half, median) | Coverage at kappa 0.9 | Coverage at kappa 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | gold | ModernBERT-base | 477 | 0.844 | 1.22 | 0.035 -> 0.031 | 0.051 -> 0.047 | 75.7% | 41.9% |
| ag_news | qwen2.5:3b | ModernBERT-base | 477 | 0.847 | 1.01 | 0.038 -> 0.040 | 0.046 -> 0.048 | 84.5% | 74.6% |
| banking77 | gold | ModernBERT-base | 496 | 0.842 | 1.24 | 0.062 -> 0.043 | 0.070 -> 0.054 | 82.3% | 70.8% |
| banking77 | qwen2.5:3b | ModernBERT-base | 496 | 0.687 | 1.60 | 0.160 -> 0.079 | 0.165 -> 0.091 | 31.9% | 13.3% |

#### ag_news, teacher `gold`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-ag_news-gold/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration --teacher gold` at `b127db2`; student run `20261020-012041-n2000`; 200 random splits into 238 inputs to fit on and 239 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 122 s; spent $0.00
- Fitted temperature over the splits: 1.22 [1.10, 1.34]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 46.0% | 88.3% [75.7%, 95.4%] | 0.897 [0.847, 0.948] | 0.911 [0.855, 0.961] |
| 0.9 | 95% lower bound, calibrated | 100.0% | 87.0% | 65.1% [46.4%, 85.5%] | 0.932 [0.873, 0.988] | 0.961 [0.889, 0.994] |
| 0.9 | point estimate, uncalibrated | 100.0% | 50.0% | 88.3% [76.6%, 95.4%] | 0.900 [0.847, 0.948] | 0.913 [0.855, 0.961] |
| 0.95 | point estimate, calibrated | 100.0% | 50.5% | 54.0% [43.9%, 77.0%] | 0.951 [0.901, 0.987] | 0.977 [0.927, 0.994] |
| 0.95 | 95% lower bound, calibrated | 100.0% | 82.0% | 36.6% [17.2%, 55.6%] | 0.970 [0.926, 1.000] | 0.992 [0.966, 1.000] |
| 0.95 | point estimate, uncalibrated | 100.0% | 50.5% | 54.8% [44.3%, 77.1%] | 0.952 [0.901, 0.987] | 0.977 [0.927, 0.994] |

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
- Command: `python -m eval.calibration --teacher ollama` at `ef873d0`; student run `20261022-000137-n2000`; 200 random splits into 238 inputs to fit on and 239 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 105 s; spent $0.00
- Fitted temperature over the splits: 0.99 [0.87, 1.10]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 42.0% | 91.6% [84.9%, 96.7%] | 0.893 [0.846, 0.945] | 0.903 [0.854, 0.954] |
| 0.9 | 95% lower bound, calibrated | 100.0% | 81.0% | 81.2% [74.5%, 90.0%] | 0.933 [0.880, 0.976] | 0.948 [0.896, 0.983] |
| 0.9 | point estimate, uncalibrated | 100.0% | 42.0% | 91.6% [84.5%, 96.7%] | 0.893 [0.851, 0.945] | 0.903 [0.857, 0.954] |
| 0.95 | point estimate, calibrated | 100.0% | 49.0% | 77.8% [72.8%, 85.8%] | 0.946 [0.895, 0.984] | 0.960 [0.914, 0.988] |
| 0.95 | 95% lower bound, calibrated | 100.0% | 80.0% | 69.5% [60.2%, 78.2%] | 0.973 [0.929, 1.000] | 0.983 [0.948, 1.000] |
| 0.95 | point estimate, uncalibrated | 100.0% | 47.0% | 77.8% [73.2%, 85.4%] | 0.946 [0.896, 0.984] | 0.960 [0.914, 0.988] |

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
| served at kappa 0.9 | 84.5% | 168 | 0.966 [0.930, 0.992] | 0.940 [0.891, 0.982] |
| served at kappa 0.95 | 74.6% | 154 | 0.972 [0.935, 1.000] | 0.962 [0.924, 0.991] |

#### banking77, teacher `gold`: does the threshold hold on inputs it was not chosen on?

- Raw output: `eval/results/calibration-banking77-gold/` (`report.json`, `meta.json`, `logits.npz`)
- Command: `python -m eval.calibration --teacher gold` at `b127db2`; student run `20261020-001737-n2000`; 200 random splits into 248 inputs to fit on and 248 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 58 s; spent $0.00
- Fitted temperature over the splits: 1.24 [1.15, 1.31]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 100.0% | 55.5% | 87.1% [81.0%, 92.8%] | 0.905 [0.857, 0.942] | 0.918 [0.869, 0.955] |
| 0.9 | 95% lower bound, calibrated | 100.0% | 90.5% | 80.2% [75.8%, 85.9%] | 0.933 [0.887, 0.968] | 0.947 [0.905, 0.976] |
| 0.9 | point estimate, uncalibrated | 100.0% | 53.0% | 87.9% [81.0%, 92.7%] | 0.901 [0.856, 0.943] | 0.914 [0.868, 0.955] |
| 0.95 | point estimate, calibrated | 100.0% | 56.5% | 76.6% [69.4%, 81.5%] | 0.952 [0.918, 0.982] | 0.963 [0.934, 0.988] |
| 0.95 | 95% lower bound, calibrated | 100.0% | 89.5% | 69.0% [64.1%, 75.8%] | 0.975 [0.940, 0.994] | 0.984 [0.955, 0.996] |
| 0.95 | point estimate, uncalibrated | 100.0% | 50.0% | 75.4% [71.0%, 81.0%] | 0.950 [0.917, 0.978] | 0.963 [0.934, 0.984] |

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
- Command: `python -m eval.calibration --teacher ollama` at `ef873d0`; student run `20261022-004245-n2000`; 200 random splits into 248 inputs to fit on and 248 to test on; at least 30 answered inputs; 1000 bootstrap resamples; wall 58 s; spent $0.00
- Fitted temperature over the splits: 1.61 [1.53, 1.67]

| Target kappa | Threshold rule | Threshold found | Unseen half: kappa reaches target | Unseen half: coverage | Unseen half: kappa on answered | Unseen half: cascade kappa |
|---|---|---|---|---|---|---|
| 0.9 | point estimate, calibrated | 99.5% | 39.7% | 39.5% [30.2%, 56.6%] | 0.894 [0.834, 0.960] | 0.959 [0.905, 0.988] |
| 0.9 | 95% lower bound, calibrated | 75.0% | 82.7% | 29.8% [11.1%, 36.7%] | 0.926 [0.855, 1.000] | 0.979 [0.950, 1.000] |
| 0.9 | point estimate, uncalibrated | 100.0% | 42.0% | 40.1% [29.8%, 57.7%] | 0.893 [0.824, 0.960] | 0.959 [0.905, 0.988] |
| 0.95 | point estimate, calibrated | 91.5% | 38.8% | 18.5% [11.3%, 34.6%] | 0.932 [0.893, 1.000] | 0.988 [0.963, 1.000] |
| 0.95 | 95% lower bound, calibrated | 45.5% | 53.8% | 12.9% [8.1%, 21.0%] | 0.959 [0.891, 1.000] | 0.996 [0.979, 1.000] |
| 0.95 | point estimate, uncalibrated | 98.0% | 40.3% | 22.6% [13.7%, 34.3%] | 0.942 [0.879, 1.000] | 0.988 [0.962, 1.000] |

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
| served at kappa 0.9 | 31.9% | 66 | 0.936 [0.872, 0.984] | 0.952 [0.889, 1.000] |
| served at kappa 0.95 | 13.3% | 32 | 0.966 [0.894, 1.000] | 0.966 [0.894, 1.000] |
<!-- END generated:calibration -->

With the local teacher (M8, `make eval-calibration TEACHER=ollama`, on the students of the
fallible-teacher table): "correct" now means "agrees with `qwen2.5:3b`", which is wrong
against gold on 21% of AG News and 45% of Banking77 held-out inputs.

- **The plan's prediction failed: a kappa-0.9 threshold exists on Banking77.** The plan
  expected none, since the teacher's self-agreement over all 200 ceiling inputs is 0.831. The
  lower-bound rule finds one at 31.9% coverage, because the teacher is more consistent on the
  inputs the student is sure of. On the 66 ceiling inputs above the served threshold, its
  self-agreement is 0.936 [0.872, 0.984], and the student's agreement with it is 0.952. The
  ceiling has to be read on the inputs the cascade answers, not on the whole split.
- **The rule holds up less well against a noisier teacher.** At kappa 0.9, the lower-bound
  threshold kept the target on the unseen half in 82.7% of the splits where it found one on
  Banking77 (it found one in 75.0% of the 200, so 62% of all splits) and 81.0% on AG News, against 90.5% and 87.0% with gold. The
  point-estimate threshold kept it in 39.7% and 42.0%.
- **This is consistency with the teacher, not correctness.** Inputs that a confident student
  and a self-consistent teacher agree on can still be wrong in the same way, like AG News's
  `sci_tech` stories labelled `business`. A cascade calibrated against a fallible teacher
  serves the teacher's systematic errors at full confidence. Telling them apart needs gold
  labels or a better teacher.

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
  `python -m eval.render`).

<!-- BEGIN generated:shift -->
No shift run yet: `make eval-shift`.
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
| A third training seed on both datasets (seeds 0 and 1 are in the table above) | `uv run --group eval python -m eval.classify ag_news banking77 --teacher gold --seed 2`, then `make results` | about 1.5 h CPU (2 ModernBERT-base runs) | $0 |

The toy task is synthetic and only exercises the pipeline; real parity numbers come from the
M2 datasets.
