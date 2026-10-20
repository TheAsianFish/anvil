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
| banking77 | gold | ModernBERT-base | 496 | 0.842 | 1.24 | 0.062 -> 0.043 | 0.070 -> 0.054 | 82.3% | 70.8% |

#### ag_news: does the threshold hold on inputs it was not chosen on?

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

#### banking77: does the threshold hold on inputs it was not chosen on?

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
<!-- END generated:calibration -->

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
| A third training seed on both datasets (seeds 0 and 1 are in the table above) | `uv run --group eval python -m eval.classify ag_news banking77 --teacher gold --seed 2`, then `make results` | about 1.5 h CPU (2 ModernBERT-base runs) | $0 |

The toy task is synthetic and only exercises the pipeline; real parity numbers come from the
M2 datasets.
