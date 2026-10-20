# Anvil

**Distil LLM calls into small, fast task models, with measured parity.**

Teams prototype a narrow task (ticket triage, intent routing, entity
extraction) as a prompt against a frontier model, then keep paying frontier
prices and latency for it. Replacing the prompt with a fine-tuned small model
is the standard fix, but every team hand-rolls it, and almost nobody measures
the question that decides whether it worked: **how many teacher labels does
the student need to agree with the teacher, and where does it still disagree?**

Anvil takes a task spec and a file of unlabeled inputs. It labels them with
Claude (structured outputs, cached, every call costed), fine-tunes a small
model in PyTorch, and measures the student's agreement with the teacher on a
held-out, teacher-labelled split as Cohen's kappa. For extraction the teacher
returns each entity's text and character offset; Anvil verifies every span
against the input, repairs the offsets a language model gets wrong, refuses
to train when too many spans cannot be aligned, and scores a token tagger by
entity F1 and token kappa. `anvil loop` grows the
training set round by round (pick, label, retrain from scratch, score) under
a label and a dollar budget and records the labels-vs-kappa-vs-cost frontier.
`anvil calibrate` fits a temperature and picks the confidence threshold that
keeps a target kappa on the inputs the student answers; `anvil export` writes
an ONNX bundle that must match PyTorch before it can be served; `anvil serve`
answers over HTTP from ONNX Runtime (no torch in the image) and sends the
inputs under the threshold back to the teacher.

Status: **M5 of 7** (Claude teacher, label store, a full PyTorch training
loop with exact resume, kappa with bootstrap intervals and error slices, the
active-learning loop with resumable rounds, span extraction with verified
character offsets and a token tagger, calibration and the escalating cascade,
ONNX export with parity gates, the server and its image, and the
`make eval-classify`, `make eval-extract`, `make eval-frontier`,
`make eval-select`, `make eval-calibration`, `make eval-export` and
`make bench` harnesses).
See [PLAN.md](PLAN.md) for the milestones.

## Demo

```console
$ make demo
== anvil label (fake teacher, offline)
labelled 91 held-out + 200 pool inputs with fake-rules-c7ee422d (0 from cache, $0.0000 spent); 291 labels in store worth $0.0000
== anvil train (tiny random-init ModernBERT)
trained 20261019-161530-n200 on 200 fake-rules-c7ee422d labels (16 held back for early stopping, best epoch 4, kappa 0.916): 6 epochs, loss 1.899 -> 0.196, 0.5s
== anvil eval (held-out agreement with the teacher)
student 20261019-161530-n200 vs fake-rules-c7ee422d on 91 held-out inputs: kappa 1.000 [1.000, 1.000], accuracy 1.000 (majority 'bug' accuracy 0.209); report .anvil/demo/ticket-triage/reports/eval-20261019-161530-n200.json
== anvil calibrate (temperature and the threshold that keeps kappa 0.9 with the teacher)
calibrated 20261019-161530-n200 on 91 held-out fake-rules-c7ee422d labels: temperature 1.000, ECE 0.020 -> 0.020
threshold 0.8821: the student answers 100.0% of held-out inputs at kappa 1.000, lower bound 1.000 (target 0.9); written to .anvil/demo/ticket-triage/students/20261019-161530-n200/calibration.json
== anvil export (ONNX fp32, checked against PyTorch before it can be served)
fp32: passed parity vs pytorch on 91 inputs: max |logit diff| 1.91e-06, agreement 1.000, kappa vs teacher 1.000 (drop +0.000); 0.7 MB (max |logit diff| <= 1e-04 and agreement == 1.0)
bundle .anvil/demo/bundle; serve it with `anvil serve .anvil/demo/bundle`
== anvil serve (ONNX Runtime; inputs under the threshold escalate to the fake teacher)
  billing          from the student  (confidence 0.988)
  feature_request  from the student  (confidence 0.995)
  account          from the teacher  (confidence 0.870)
== anvil loop (random selection, three rounds, fresh student each round)
loop demo (random, seed 0) vs fake-rules-c7ee422d:
  round  labels  kappa [95% CI]          label cost  spent
      0      50  0.582 [0.452, 0.697]  $   0.0000  $0.0000
      1     100  0.881 [0.793, 0.955]  $   0.0000  $0.0000
      2     200  0.836 [0.738, 0.923]  $   0.0000  $0.0000
== anvil label (extraction, fake teacher with off-by-one offsets)
labelled 72 held-out + 240 pool inputs with fake-rules-a25234ff (0 from cache, $0.0000 spent); 312 labels in store worth $0.0000
== anvil train (tiny random-init ModernBERT tagger; aligns every answer first)
aligned 548 teacher mentions: 430 at their offset, 118 repaired (shifted 118), 0 dropped (0.0%), 0 snapped to token boundaries
trained 20261019-161612-n240 on 240 fake-rules-a25234ff labels (21 held back for early stopping, best epoch 3, entity F1 1.000): 5 epochs, loss 1.752 -> 0.015, 0.5s
== anvil eval (entity F1 and token kappa against the teacher)
student 20261019-161612-n240 vs fake-rules-a25234ff on 72 held-out inputs: entity F1 0.929 [0.885, 0.968] (precision 0.904, recall 0.955), token kappa 0.968 [0.949, 0.985] (held-out teacher mentions: 37 repaired, 0 dropped, 0.0%); report .anvil/demo/restaurant-slots/reports/eval-20261019-161612-n240.json
== demo finished in 59s
```

The demo runs offline (loop warnings and its last line trimmed above): the
teacher is a keyword-rule `FakeTeacher` with 5% injected label noise for
classification and, for extraction, a fifth of its offsets moved by one
character, which alignment repairs (118 of 548 here). The first toy student
is calibrated (it agrees with every held-out label, so its temperature stays
1 and the threshold is its lowest held-out confidence), exported to ONNX and
served; of the three requests, the one under the threshold is answered by the
teacher. The students are two-layer random-init ModernBERTs with a
vocabulary built from the pool, so the demo finishes in about a minute on a
laptop CPU, most of it Python start-up across ten commands. Its numbers are a
smoke check of the pipeline, not results.

<!-- GIF placeholder: record `anvil label/train/eval` against Claude with
`vhs docs/demo.tape` (or asciinema + agg) once real labels exist, save it as
docs/screenshots/demo.gif and link it here. -->

## Quickstart

```bash
git clone https://github.com/TheAsianFish/anvil && cd anvil
uv sync --all-extras            # Python 3.12, CPU torch from the PyTorch index
make demo                       # offline slice on the toy task

# With Claude as the teacher (spends money; every call is cached and costed):
export ANTHROPIC_API_KEY=...
cd examples/toy
uv run anvil label --inputs inputs.jsonl --n 200     # held-out split + 200 pool inputs
uv run anvil train                                   # student.model: answerdotai/ModernBERT-base for real runs
uv run anvil eval
uv run anvil train --resume <run-id>                 # continue an interrupted run from its checkpoint

# Active learning: 100, 200, 400 labels, a fresh student each round, stop at $1 of labels
uv run anvil loop --budgets 100,200,400 --max-usd 1.00 --strategy random --id first
uv run anvil loop --budgets 100,200,400 --max-usd 1.00 --strategy random --id first  # resumes

# The parity eval on AG News and Banking77 (downloads the datasets and ModernBERT-base):
make eval-classify                    # with a key: labels with Claude Haiku once, replays after
make eval-classify TEACHER=gold       # no key: dataset gold labels as the teacher
make eval-frontier TEACHER=gold       # kappa vs labels, 3 seeds x 5 budgets per strategy (CPU, ~2.5 h)
make eval-select                      # the strategies on synthetic blobs (seconds)
make eval-extract TEACHER=gold        # span extraction on MIT Restaurant (ModernBERT-base tagger, CPU)

# Calibrate, export and serve (classify tasks)
uv run anvil calibrate --target-kappa 0.9            # temperature + threshold, on held-out labels
uv run anvil export --int8                           # ONNX bundle, parity-gated, under .anvil/<task>/exports/
uv run anvil serve .anvil/ticket-triage/exports/<run> --teacher claude   # POST /predict {"texts": [...]}
make eval-calibration eval-export TEACHER=gold       # the M2 students: threshold study, int8 layer probe
make bench TEACHER=gold                              # PyTorch vs ONNX Runtime vs HTTP, CPU, 4 threads

# Extraction: the same three commands on a `kind: extract` spec
cd examples/toy-extract
uv run anvil label --inputs inputs.jsonl --n 200 --teacher fake
uv run anvil train --teacher fake     # aligns every answer first; refuses above 5% unaligned
uv run anvil eval                     # entity F1 and token kappa with bootstrap intervals
```

A task is a YAML file:

```yaml
name: ticket-triage
kind: classify
instructions: |
  You route customer support tickets to the team that should handle them.
labels:
  - name: billing
    description: charges, invoices, refunds, payment methods, plans and pricing
  - name: bug
    description: something in the product is broken, crashes or returns an error
teacher: {provider: claude, model: claude-opus-5-5, effort: low}
student: {model: answerdotai/ModernBERT-base, max_length: 64, epochs: 3, learning_rate: 5.0e-5}
splits: {heldout_fraction: 0.25, seed: 0}
```

An extraction task has `kind: extract` and lists entity types under
`labels`; see [examples/toy-extract/anvil.yaml](examples/toy-extract/anvil.yaml).

Inputs are JSONL, one `{"text": ..., "id": optional}` per line. State lives
under `ANVIL_HOME` (default `.anvil/`): the SQLite label store, the teacher
response cache and trained students.

`POST /predict` answers each input with its label, the student's confidence,
`source` (`student` or `teacher`) and whether it was escalated; a teacher
failure falls back to the student's answer with the reason in `error`.
`/healthz` and `/metrics` (Prometheus text) come with it, and every request is
one JSON log line with counts and latency, never the input text.

Docker (the container runs as uid 1000, so pass your uid to write into the
bind mount):

```bash
make docker
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" -e ANTHROPIC_API_KEY anvil:dev label --spec anvil.yaml --inputs inputs.jsonl --n 200
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" anvil:dev label --spec anvil.yaml --inputs inputs.jsonl --teacher fake   # offline

# The serving image: ONNX Runtime and the Rust tokenizer, no torch; mount a bundle at /model
docker run --rm -p 8000:8000 -v "$PWD/.anvil/ticket-triage/exports/<run>:/model:ro" anvil-serve:dev --teacher none
```

## Architecture

```mermaid
flowchart LR
    I[inputs.jsonl] --> ST[(store.sqlite<br/>inputs, append-only labels,<br/>hash splits)]
    SPEC[anvil.yaml] --> T
    ST -->|unlabelled batch| T[Teacher<br/>Claude structured outputs<br/>disk cache + cost]
    T -->|labels + tokens + $| ST
    ST -->|labelled pool, classify| TR[train/loop.py<br/>AMP, accumulation, warmup,<br/>early stopping, exact resume]
    ST -->|labelled pool, extract| AL[align.py<br/>spans verified vs text,<br/>snapped to tokens, 5% gate]
    AL --> TR
    TR --> S[Student<br/>ModernBERT classifier<br/>or BIO tagger]
    ST -->|held-out, teacher-labelled| EV[evaluate/<br/>kappa + bootstrap CI,<br/>per-class, slices]
    S --> EV
    EV --> R[reports/eval-*.json]
    S -->|probabilities + embeddings<br/>on the unlabelled pool| SEL[select.py<br/>next batch]
    SEL -->|picks, recorded first| ST
    EV --> FR[loops/*/frontier.jsonl<br/>labels vs kappa vs $]
    LF[eval/labels/<br/>recorded labels] -.->|replay| T
    EV --> RES[eval/results/ -> RESULTS.md]
    S -->|held-out logits| CAL[calibrate.py<br/>temperature, ECE,<br/>threshold at target kappa]
    CAL --> EX[export.py<br/>ONNX fp32 / int8,<br/>parity gates]
    EX -->|bundle| SV[serve.py<br/>FastAPI + ONNX Runtime]
    SV -->|confidence under threshold| T
```

| Module | Role |
|---|---|
| `anvil/spec.py` | Pydantic task spec; the prompt hash keys every label |
| `anvil/teacher/` | `Teacher` protocol; `ClaudeTeacher` (one structured-output request per input, enum-constrained label, content-addressed disk cache, token and dollar accounting); transports for the live API, recording and replay; `FakeTeacher` for offline runs |
| `anvil/store.py` | SQLite store: inputs, append-only labels (triggers reject UPDATE and DELETE), held-out membership from a seeded hash of the input text; failed teacher calls kept with their cost |
| `anvil/students/encoder.py` | Any Hugging Face encoder with a classification head; `tiny` builds a random-init ModernBERT for tests and the demo |
| `anvil/train/loop.py` | One generic PyTorch loop: seeded batches, gradient accumulation, AMP, warmup + linear decay, clipping, early stopping, per-epoch checkpoints that resume bit for bit |
| `anvil/train/classifier.py` | Cross-entropy for the encoder classifier; early stopping on the kappa of a validation slice of the training labels |
| `anvil/evaluate/` | Kappa and accuracy with vectorised percentile bootstrap CIs, macro F1, per-class scores, top confusions, teacher self-agreement, slices by confidence and input length; for extraction, exact-match entity F1, word-level tag kappa, an error breakdown (boundary, type, missed, spurious) and intervals that resample inputs |
| `anvil/spans.py`, `anvil/align.py` | Extraction answers as mentions (text, type, claimed offset); alignment verifies each against the input with a ladder of repair rules, resolves overlaps, snaps spans to student tokens as BIO tags and reports every repair and drop |
| `anvil/students/tagger.py`, `anvil/train/tagger.py` | `EncoderTagger`: any encoder with a fast tokenizer and a token-classification head, trained by the same loop with early stopping on validation entity F1 |
| `eval/extract.py` | `make eval-extract` on MIT Restaurant: gazetteer baseline, ceiling, gold comparison and the alignment stress study |
| `anvil/teacher/labelfile.py` | Replays recorded labels (or a dataset's gold labels, as model `gold`) offline |
| `eval/classify.py`, `eval/render.py` | `make eval-classify` and the RESULTS.md renderer |
| `anvil/select.py` | `select_batch(pool, k, strategy, seed)`: `random` (the reference curve); `uncertainty`, `margin` and `diverse` are declared with a contract test suite and not written yet |
| `anvil/loop.py` | `anvil loop`: rounds of select, label, retrain from scratch, score; label, dollar and target-kappa stops; picks stored before labelling so an interrupted loop resumes to the same rows |
| `eval/frontier.py`, `eval/select_study.py` | `make eval-frontier` (3 seeds x 5 budgets per strategy) and `make eval-select` (synthetic blobs, logistic student) |
| `anvil/calibrate.py` | Temperature scaling (bisection on the NLL gradient), ECE and reliability bins, the coverage-vs-kappa curve, and the threshold search on the point estimate or a one-sided bootstrap lower bound; numpy only |
| `anvil/export.py`, `anvil/runtime.py` | `torch.onnx` (dynamo) export with dynamic batch and sequence axes, fp32 parity gate through the serving tokenizer, opt-in int8 with the MLP down-projections in fp32; `OnnxStudent` loads a bundle with ONNX Runtime and `tokenizers`, never torch |
| `anvil/serve.py` | `anvil serve`: FastAPI `/predict`, `/healthz`, `/metrics`; inputs under the threshold go to the teacher in one batch per request; every teacher failure degrades to the student's answer |
| `eval/calibration.py`, `eval/export.py`, `eval/bench.py` | `make eval-calibration` (temperature, coverage, and the split study of whether a threshold holds on unseen inputs), `make eval-export` (parity and the int8 layer probe), `make bench` |
| `anvil/pipeline.py`, `anvil/cli.py` | `label`, `train`, `eval`, `loop`, `calibrate`, `export`, `serve` as a library API and a Typer CLI |

**The hard part** (the core of the project, built over M2 and M3) is the
cost-versus-quality frontier: how few teacher labels reach a target kappa
with the teacher, how active selection moves that curve against random
sampling, and where the student still fails. M2 built the measuring
instrument: kappa with a bootstrap interval verified against scikit-learn and
scipy (and a coverage simulation), a ceiling from the teacher's own
self-agreement, slices that show where the student disagrees, and a training
loop whose runs are reproducible to the bit. M1 laid the ground it needs:
labels are keyed by (teacher model, prompt hash) so curves never mix two
prompts, splits are a pure function of the input text so re-runs never move an
item, and every label carries its cost so a curve can be drawn in dollars. M4
extends the instrument to extraction, where the hard part is upstream of
training: a teacher's character offsets are unreliable, so every span is
verified against the input and repaired by a ladder of matching rules, snapped
to the student's tokens, and counted; a teacher whose spans cannot be aligned
stops training instead of silently teaching the student the wrong targets.
M5 turns the measurement into the product number, coverage at a target
parity: a temperature-scaled student answers what it is sure of and
escalates the rest, with the threshold checked on inputs it was not chosen
on, because a threshold that reaches the target in-sample misses it on new
data about half the time (RESULTS.md).

## Design decisions

- **Agreement with the teacher, as Cohen's kappa.** Accuracy against the
  teacher is inflated by class imbalance; kappa corrects for chance. Each
  report shows the majority-class baseline next to it.
- **One request per input, with an enum-constrained JSON schema.** Batching
  several inputs into one prompt is cheaper per token, but per-input requests
  make the cache per input (adding inputs only pays for new ones) and let the
  schema guarantee a valid label. Invalid outputs, truncation, refusals and
  an input too long for the teacher are recorded as failures, never guessed.
- **No server-side model fallback on refusal.** A fallback would answer with
  a different model under the same key and silently mix two teachers in one
  agreement measurement. A refused input is reported and retried next run.
- **Hash-based splits and an append-only store.** Held-out membership is a
  seeded hash of the input *text*, so adding inputs never moves an item and
  two copies of the same text can never sit on both sides. Labels and failed
  calls are never overwritten; a change to the instructions, labels or
  teacher effort creates a new key, and a student trained under other split
  settings is refused at eval time.
- **Every billed call is accounted, including failures.** A refused or
  truncated response still costs its input and thinking tokens, so failures
  are stored with their cost and count toward the dollars spent. Two inputs
  with the same text are one request, and when a run stops early (a
  configuration error, Ctrl-C) the answers that were already paid for are
  stored before the error surfaces.
- **Intervals, not point estimates.** Kappa comes with a 95% percentile
  bootstrap interval over held-out items (1,000 paired resamples, computed
  for all resamples at once from per-resample marginals). Two students
  scored on the same items are compared with a paired bootstrap of the
  difference, not by whether their separate intervals overlap: overlapping
  intervals do not make two students tied.
- **Early stopping never looks at the held-out split.** The epoch is chosen
  on a validation slice of the training labels (10%, picked by a hash of the
  text); choosing it on the reported split would bias the headline upward.
- **Exact resume.** A checkpoint holds the model, optimizer, scheduler,
  gradient scaler, both RNGs, the epoch's data order and position, and the
  early-stopping state, fingerprinted with the config and the training data;
  an interrupted run resumed from it ends with bit-identical weights (tested
  at five crash points on CPU; on CUDA that holds only with deterministic
  kernels).
- **No key, no fake parity.** Without a key, `make eval-classify
  TEACHER=gold` uses the dataset's own labels as the teacher under the model
  id `gold`, so the harness and the student are measured on real data while
  Claude parity stays under "To measure".
- **For extraction the copied text is evidence, the offset is a hint.** A
  language model reads tokens, so its character offsets drift (off by one,
  counted in UTF-16 units past an emoji, counted in words) while the text it
  copies is usually exact. Alignment matches each mention to an occurrence of
  its text, from the strictest rule (the exact offset) to the loosest (case,
  whitespace, Unicode normalisation, trimmed punctuation), takes the occurrence
  nearest the claimed offset, and records which rule fired. *Rejected:*
  asking the teacher for BIO tags (it tokenises differently from the student)
  and trusting offsets (a plain check loses almost every shifted span; see the
  stress table in RESULTS.md).
- **Overlaps keep the longer span; training stops above 5% unaligned.** BIO
  tags cannot hold nested entities, so the longer of two overlapping spans
  wins and the other is counted as dropped. When more than 5% of a teacher's
  mentions cannot be aligned to the text or the student's tokens, the prompt
  or the labels are broken and `anvil train` refuses (exit code 2) before the
  first step, with examples in the report.
- **Extraction is scored two ways, with intervals over sentences.** Exact-match
  entity F1 is what a consumer of the extractions sees; kappa over each word's
  BIO tag gives partial credit and is chance-corrected like the classification
  number. The bootstrap resamples sentences, not words: words of one sentence
  are not independent draws.
- **Every round retrains from scratch.** A warm-started student at 400
  labels has had more updates than a fresh one, so its kappa would mix "more
  labels" with "more training". Each round trains a new student with the same
  seed and settings on every label so far; CPU time is the price.
- **Round 0 is shared.** The first batch is a random draw that depends only
  on the seed, so every strategy of a seed starts from the same student and
  the curves differ only from the first selected batch on (a paired
  comparison, not two unrelated runs).
- **Picks are written before labels.** A loop records each round's picks in
  the store, then labels them. An interrupted loop resumes with the same
  picks and, training being seeded, ends with the same rows as one that never
  stopped (tested by crashing a round). A refused pick keeps its slot in the
  budget and is never paid for twice.
- **Dollar budgets count label cost, not spend.** A label replayed from the
  cache or a recording counts what it cost to make, so a replay stops at the
  same round as the live run it reproduces.
- **Fixed-length training on the frontier.** At 100 labels a 10% validation
  slice is ten inputs for up to 77 classes; in a pilot (output not kept),
  early stopping on it quit at epoch 4 and kept an untrained epoch 1. Frontier
  rounds train for at least 600 optimizer steps instead (`min_steps`).
- **A deterministic tiny student.** `tiny` uses a word-level vocabulary built
  by hand: the `tokenizers` WordPiece trainer breaks frequency ties in
  hash-map order, which gave a different vocabulary, and a different trained
  model, in every process.
- **Offline by default.** Unit tests use `FakeTeacher`, a scripted transport
  and tiny models, with the network blocked; live tests run only with a key;
  recorded Claude responses replay through the same parsing code (none are
  committed yet; `make record` creates them once a key exists).
- **The cascade threshold is set on a lower bound, not the estimate.** The
  threshold is the largest coverage whose kappa on the answered inputs
  reaches the target; picking it on the point estimate fits the sample's
  noise, so fresh inputs land below the target about as often as above.
  `anvil calibrate` uses the one-sided 95% bootstrap lower bound by default
  and `make eval-calibration` measures both rules on held-out halves the
  threshold never saw. *Rejected:* a learned deferral model (more to
  maintain for a gain the plain threshold already measures).
- **Temperature scaling, and when not to.** One scalar T fitted by NLL on
  the held-out teacher labels makes confidence readable as agreement; it
  does not change any label. When the fit runs to a bound (the student
  agrees with every held-out label, or its few errors have nearly flat
  logits) the NLL has no minimum, and at the lower bound every input,
  nonsense included, would score 1.0, so T stays 1. Because T reorders
  inputs by confidence, the threshold is chosen and served on the same T.
- **No threshold, no trust.** When no threshold reaches the target, the
  server escalates every input, and refuses to start without a teacher to
  send them to; it never quietly answers everything. An explicit
  `--threshold` is the operator's override.
- **A bundle is served only if it matches PyTorch.** fp32 must agree to
  1e-4 in logits and exactly in argmax on held-out inputs, measured through
  the serving tokenizer, or nothing is written; int8 must keep argmax
  agreement on the inputs the cascade answers, batched and one at a time
  (dynamic quantization scales activations per batch), and the kappa with
  the teacher on every input within 0.01, or its file is deleted and the
  server refuses it.
- **No torch on the serving path.** `anvil serve` uses ONNX Runtime and the
  Rust tokenizer; a test imports it with torch blocked, and the serving
  image is built from the lock file without torch or transformers.
- **ONNX export goes through `torch.onnx`.** `optimum-onnx` 0.1 requires
  `transformers<4.58` and cannot be installed next to transformers 5; the M1
  spike in [docs/notes/onnx-export-spike.md](docs/notes/onnx-export-spike.md)
  measures the `torch.onnx` path for ModernBERT-base.

## Results

Every number comes from a command in [CONTRIBUTING.md](CONTRIBUTING.md) with
its raw output under `eval/results/`; [RESULTS.md](RESULTS.md) tables are
rendered from those files. ModernBERT-base on 2,000 teacher labels (10% of
them held back to pick the epoch), CPU, with the dataset's gold labels as the
teacher (no key yet, so not a Claude parity number):

| Dataset | Student kappa [95% CI] | TF-IDF + LR kappa [95% CI] | Student minus TF-IDF [95% paired CI] | Majority accuracy |
|---|---|---|---|---|
| AG News (4 topics) | 0.844 [0.805, 0.882] | 0.835 [0.795, 0.873] | +0.009 [-0.033, 0.054] | 0.260 |
| Banking77 (77 intents) | 0.842 [0.807, 0.873] | 0.662 [0.621, 0.702] | +0.180 [0.141, 0.223] | 0.024 |

The encoder's gain over a bag of words is large on short, fine-grained
intents and nil on long news articles (the paired interval includes zero).
Training randomness matters as much as the held-out sample: re-training on
AG News with another student seed moved kappa from 0.844 to 0.896, about 70%
of one run's interval width; on Banking77 two seeds tie at 0.842 but disagree
on 15% of items. That is why the frontier milestone runs three seeds per
point. On Banking77 the least confident tenth of predictions (50 items) agrees
40% of the time against 98-100% for the top six tenths, which is what the
active-selection and cascade milestones build on. Parity with Claude and the
teacher's self-agreement ceiling are listed under "To measure" with their
commands and cost (about $0.35 with Haiku 5.5).

Span extraction on MIT Restaurant (8 slot types, ~9-word queries), the same
setup with a ModernBERT-base BIO tagger on 1,442 gold-teacher labels, scored
on 500 held-out sentences with intervals that resample sentences:

| Entity F1 [95% CI] | Token kappa [95% CI] | Gazetteer F1 [95% CI] | Student minus gazetteer [95% paired CI] |
|---|---|---|---|
| 0.727 [0.696, 0.757] | 0.831 [0.810, 0.850] | 0.472 [0.445, 0.499] | +0.254 [0.222, 0.288] |

Most of the student's misses are boundaries (127 of 237), not types (43). The
alignment stress study shows why offsets cannot be trusted as given: with
offsets off by one, a few characters, a word index or zero, plain
verification keeps none of the ~960 changed spans and the repair ladder
recovers all of them. Claude parity on this task is under "To measure"
(about $0.30 with Haiku 5.5).

The cascade and the served student (M5), on the same Banking77 and AG News
students, gold labels as the teacher, CPU:

| | Banking77 | AG News |
|---|---|---|
| Temperature; ECE before -> after on unseen halves (median of 200 splits) | 1.24; 0.070 -> 0.054 | 1.22; 0.051 -> 0.047 |
| Student coverage at kappa 0.9 with the teacher (95% lower-bound rule) | 82.3% | 75.7% |
| Splits where the unseen half kept kappa 0.9: estimate rule / lower-bound rule | 55.5% / 90.5% | 46.0% / 87.0% |
| ONNX fp32 vs PyTorch, max abs logit diff over held-out | 2.3e-05 | 2.9e-05 |
| int8 (211 MB vs 601 MB): agreement with fp32 on all / on answered inputs; kappa change | 0.964 / 1.000; -0.002 | 0.992 / 1.000; -0.006 |

Choosing the threshold where the kappa estimate reaches the target holds that
target on new inputs about half the time; requiring the bootstrap lower bound
to reach it holds it in roughly nine of ten splits, for 7 (Banking77) to 23
(AG News) points of coverage. Temperature scaling makes the confidences
readable as agreement rates but leaves coverage unchanged. On Banking77 the
served student answers in 27.6 ms p50 over HTTP in fp32 and 18.1 ms in int8
(PyTorch in process: 55.3 ms), at 4 threads on one CPU; on a CPU, batch 32
costs more per input than batch 1 because of padding. The int8 layer probe
pins its accuracy loss on the MLP down-projections, which stay fp32. Teacher
latency and dollars per 1,000 inputs need a key and are under "To measure".

## Roadmap

- [x] M1: scaffold, CI, Claude teacher with cache and cost accounting, label store, offline slice, ONNX export spike
- [x] M2: full training loop and evaluation core (bootstrap CI, teacher self-agreement ceiling, error slices); AG News and Banking77 harness (Claude parity to measure)
- [x] M3: active-learning loop, selection interface with the random baseline and contract tests, frontier and synthetic selection evals (Claude frontier and the non-random strategies to measure)
- [x] M4: span extraction with character-offset alignment, a BIO tagger, entity F1 and token kappa, MIT Restaurant harness (Claude parity to measure)
- [x] M5: calibration and the cascade threshold, ONNX export with parity gates, FastAPI server with teacher escalation and a torch-free image, benchmarks (teacher latency and cost to measure)
- [ ] M6: Qwen2.5-0.5B LoRA generation student
- [ ] M7: release (PyPI, GHCR, Hugging Face model cards)

## License

MIT
