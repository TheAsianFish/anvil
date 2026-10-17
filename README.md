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
held-out, teacher-labelled split as Cohen's kappa. Later milestones add
uncertainty-driven selection of the next batch to label, calibration, ONNX
export with a parity gate, and a server that escalates low-confidence inputs
back to the teacher.

Status: **M2 of 7** (Claude teacher, label store, a full PyTorch training
loop with exact resume, kappa with bootstrap intervals and error slices, and
the `make eval-classify` harness on AG News and Banking77). See
[PLAN.md](PLAN.md) for the milestones.

## Demo

```console
$ make demo
== anvil label (fake teacher, offline)
labelled 91 held-out + 200 pool inputs with fake-rules-c7ee422d (0 from cache, $0.0000 spent); 291 labels in store worth $0.0000
== anvil train (tiny random-init ModernBERT)
trained 20261017-072429-n200 on 200 fake-rules-c7ee422d labels (16 held back for early stopping, best epoch 4, kappa 0.916): 6 epochs, loss 1.899 -> 0.196, 0.5s
== anvil eval (held-out agreement with the teacher)
student 20261017-072429-n200 vs fake-rules-c7ee422d on 91 held-out inputs: kappa 1.000 [1.000, 1.000], accuracy 1.000 (majority 'bug' accuracy 0.209); report .anvil/demo/ticket-triage/reports/eval-20261017-072429-n200.json
== demo finished in 17s
```

The demo runs offline: the teacher is a keyword-rule `FakeTeacher` with 5%
injected label noise and the student is a two-layer random-init ModernBERT
with a vocabulary built from the pool, so it finishes quickly on
a laptop CPU (17 s here, most of it Python start-up across three commands). Its kappa is a smoke check of the pipeline, not a result.

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

# The parity eval on AG News and Banking77 (downloads the datasets and ModernBERT-base):
make eval-classify                    # with a key: labels with Claude Haiku once, replays after
make eval-classify TEACHER=gold       # no key: dataset gold labels as the teacher
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

Inputs are JSONL, one `{"text": ..., "id": optional}` per line. State lives
under `ANVIL_HOME` (default `.anvil/`): the SQLite label store, the teacher
response cache and trained students.

Docker (the container runs as uid 1000, so pass your uid to write into the
bind mount):

```bash
make docker
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" -e ANTHROPIC_API_KEY anvil:dev label --spec anvil.yaml --inputs inputs.jsonl --n 200
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" anvil:dev label --spec anvil.yaml --inputs inputs.jsonl --teacher fake   # offline
```

## Architecture

```mermaid
flowchart LR
    I[inputs.jsonl] --> ST[(store.sqlite<br/>inputs, append-only labels,<br/>hash splits)]
    SPEC[anvil.yaml] --> T
    ST -->|unlabelled batch| T[Teacher<br/>Claude structured outputs<br/>disk cache + cost]
    T -->|labels + tokens + $| ST
    ST -->|labelled pool| TR[train/loop.py<br/>AMP, accumulation, warmup,<br/>early stopping, exact resume]
    TR --> S[Student<br/>ModernBERT classifier]
    ST -->|held-out, teacher-labelled| EV[evaluate/<br/>kappa + bootstrap CI,<br/>per-class, slices]
    S --> EV
    EV --> R[reports/eval-*.json]
    LF[eval/labels/<br/>recorded labels] -.->|replay| T
    EV --> RES[eval/results/ -> RESULTS.md]
```

| Module | Role |
|---|---|
| `anvil/spec.py` | Pydantic task spec; the prompt hash keys every label |
| `anvil/teacher/` | `Teacher` protocol; `ClaudeTeacher` (one structured-output request per input, enum-constrained label, content-addressed disk cache, token and dollar accounting); transports for the live API, recording and replay; `FakeTeacher` for offline runs |
| `anvil/store.py` | SQLite store: inputs, append-only labels (triggers reject UPDATE and DELETE), held-out membership from a seeded hash of the input text; failed teacher calls kept with their cost |
| `anvil/students/encoder.py` | Any Hugging Face encoder with a classification head; `tiny` builds a random-init ModernBERT for tests and the demo |
| `anvil/train/loop.py` | One generic PyTorch loop: seeded batches, gradient accumulation, AMP, warmup + linear decay, clipping, early stopping, per-epoch checkpoints that resume bit for bit |
| `anvil/train/classifier.py` | Cross-entropy for the encoder classifier; early stopping on the kappa of a validation slice of the training labels |
| `anvil/evaluate/` | Kappa and accuracy with vectorised percentile bootstrap CIs, macro F1, per-class scores, top confusions, teacher self-agreement, slices by confidence and input length |
| `anvil/teacher/labelfile.py` | Replays recorded labels (or a dataset's gold labels, as model `gold`) offline |
| `eval/classify.py`, `eval/render.py` | `make eval-classify` and the RESULTS.md renderer |
| `anvil/pipeline.py`, `anvil/cli.py` | `label`, `train`, `eval` as a library API and a Typer CLI |

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
item, and every label carries its cost so a curve can be drawn in dollars.

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
- **A deterministic tiny student.** `tiny` uses a word-level vocabulary built
  by hand: the `tokenizers` WordPiece trainer breaks frequency ties in
  hash-map order, which gave a different vocabulary, and a different trained
  model, in every process.
- **Offline by default.** Unit tests use `FakeTeacher`, a scripted transport
  and tiny models, with the network blocked; live tests run only with a key;
  recorded Claude responses replay through the same parsing code (none are
  committed yet; `make record` creates them once a key exists).
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

## Roadmap

- [x] M1: scaffold, CI, Claude teacher with cache and cost accounting, label store, offline slice, ONNX export spike
- [x] M2: full training loop and evaluation core (bootstrap CI, teacher self-agreement ceiling, error slices); AG News and Banking77 harness (Claude parity to measure)
- [ ] M3: active learning and the labels-vs-kappa-vs-cost frontier
- [ ] M4: span extraction with character-offset alignment
- [ ] M5: calibration, ONNX export with a parity gate, FastAPI server with teacher escalation, benchmarks
- [ ] M6: Qwen2.5-0.5B LoRA generation student
- [ ] M7: release (PyPI, GHCR, Hugging Face model cards)

## License

MIT
