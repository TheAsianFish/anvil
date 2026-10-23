# Contributing

This is the working guide for anyone touching the repo. Read it before the
first commit of a session.

## Setup

```bash
uv sync --all-extras          # Python 3.12, creates .venv and installs dev tools
uv run anvil --help
```

CPU-only torch is enough for everything except the GPU frontier run; the lock
file pins it to the PyTorch CPU wheel index (`[tool.uv.sources]` in
`pyproject.toml`), so local installs, CI and the Docker image all get the same
CPU build. Set `ANTHROPIC_API_KEY` only when you intend to spend money:
live tests and evals are skipped without it and nothing else needs it.

## Commands

| Command | What it does |
|---|---|
| `make check` | ruff format check, ruff lint, mypy strict, pytest (offline, fast). The quality gate. |
| `make fmt` | ruff format + autofix |
| `make test` | offline unit and integration tests (fake teacher, tiny models, replayed recordings) |
| `make test-live` | tests that call Claude; skipped without `ANTHROPIC_API_KEY` |
| `make record` | re-record the Claude fixtures under `tests/recordings/` (a few cents with Haiku) |
| `make demo` | the offline vertical slice on the toy task (under two minutes) |
| `make wheel-check` | build the wheel, install it into a fresh venv from PyPI (CPU torch), run `anvil --help` and the demo against it; the release gate |
| `make eval-classify` / `eval-extract` / `eval-generate` | parity evals (M2, M4, M6); write `eval/results/` and refresh RESULTS.md. All three take `TEACHER=auto\|claude\|replay\|gold` (`auto` replays committed labels, else calls Claude when a key is set); `eval-classify` also takes `TEACHER=ollama` (M8: the local model pinned in `eval/classify/*.yaml`, replayed from `eval/labels/` when recorded, else labelled through `ollama serve`; `python -m eval.classify <dataset> --teacher ollama --label-only` records the labels without training). `eval-generate` fine-tunes Qwen2.5-0.5B with LoRA on CPU (about 2 h) or on a Colab GPU (`notebooks/anvil_gpu.ipynb`); after a scoring change, `python -m eval.generate wikisql --teacher <t> --rescore` scores the stored queries again without retraining |
| `make results` | re-render the RESULTS.md tables from `eval/results/` (a test fails when they are stale) |
| `make eval-frontier` / `eval-select` / `eval-calibration` / `eval-export` | the frontier, selection (M3), calibration and export (M5) studies; `eval-calibration TEACHER=ollama` calibrates the M8 student and reports the teacher's ceiling on the inputs the cascade accepts |
| `make eval-noise` | the M8 control: the frontier's MiniLM student on gold, on the local teacher's labels, and on gold corrupted at the teacher's error rate (uniform and the teacher's own confusions), 3 seeds; replays `eval/labels/`, about 2 h of CPU |
| `make eval-shift` | M9: the served Banking77 bundle off-distribution (its test split, CLINC150's out-of-scope utterances, AG News; max softmax against max logit, energy, k-NN and Mahalanobis scores). Datasets pinned in `eval/shift.yaml`; exports the bundle when `make eval-export` has not; about 20 min of CPU once the student exists. `python -m eval.shift --scores eval/results/shift-banking77-gold/scores.npz` re-runs the analysis from the committed scores without a model |
| `make eval-coverage` | M10, M11: a seeded simulation of how often the one-sided percentile and BCa bootstrap bounds and the exact (Clopper-Pearson) and Wilson bounds on the agreement count cover the population kappa (kappa {0.85, 0.9, 0.95} x items {100, 250, 500} x labels {4, 77}) and how often the threshold search's chosen threshold truly keeps its target, plus the count bounds' exact pointwise coverage of the agreement rate; offline, about 30 s on two cores. `python -m eval.coverage --check` re-runs it and compares the counts with the committed report (CI does, under a 60 s timeout) |
| `make eval-extract-cascade` | M12: the tagger in the cascade from the committed extraction predictions (`python -m eval.extract --predictions eval/results/extract-mit_restaurant-gold/report.json`; no model, about 20 s): F1-vs-coverage curve, thresholds at F1 0.9 and 0.95 under the point, percentile and BCa rules (`calibrate.choose_statistic_threshold` on `extraction.entity_f1_statistic`), 200 unseen-half splits, M4's error kinds either side of the threshold and the per-kind AUROC of the confidence; writes `eval/results/cascade-extract-<dataset>-<teacher>/`. A full `make eval-extract` runs it on its own report |
| `make bench` | latency, throughput and cost table (M5) |
| `make docker` | build the CLI image and the torch-free serving image (`--target serve`) |
| `anvil push <export> --repo ns/name [--dry-run]` | write the model card (`README.md`) into an export and upload it to the Hugging Face Hub; needs `HF_TOKEN` unless `--dry-run` |
| `make spike-onnx` | the M1 ModernBERT ONNX export spike (downloads ModernBERT-base) |

Every eval and bench writes raw output under `eval/results/<name>/` with the
command, git SHA, models, seed, wall time and dollars spent. RESULTS.md is
regenerated from those files by `make results` (`python -m eval.render`),
never edited by hand for numbers: generated tables sit between
`<!-- BEGIN generated:... -->` markers, and the M1 spike table, which predates
the renderer, names its source file. Eval dependencies (`datasets`) are in the
`eval` dependency group, which `make eval-*` installs on demand; CI does not.
`make eval-extract` reads its dataset's raw files with `huggingface_hub` and
needs no extra group; `make eval-generate` reads WikiSQL's parquet with pyarrow (the `eval`
group). Labels and judge verdicts a live eval pays for are committed under
`eval/labels/` so every later run replays them offline and for free. The
`## Spend` ledger at the top of RESULTS.md lists every committed run with its
command, commit, wall time and dollars.

Environment variables (all optional, read in `anvil/settings.py`):
`ANVIL_HOME` (state directory, default `.anvil`), `ANVIL_TEACHER_WORKERS`
(concurrent teacher requests, default 4), `ANVIL_RECORD_DIR` (also write
every live teacher response there as a replayable recording),
`ANVIL_LOG_LEVEL` (default `INFO`; logs are JSON lines on stderr),
`ANVIL_SERVE_THREADS` (ONNX Runtime intra-op threads for `anvil serve`,
default: ONNX Runtime's own choice), `ANVIL_OLLAMA_URL` (the Ollama server a
`provider: ollama` teacher uses, default `http://127.0.0.1:11434`), `HF_TOKEN` (write token for `anvil push`;
never logged).

## Layout

```
anvil/            package: spec, teacher/, store, students/, train/, evaluate/,
                  spans + align + extraction (M4), select + loop (M3), calibrate (any
                  pooled statistic since M12) +
                  export + runtime + serve (M5), outputs + chat + generation + merge,
                  students/generator.py, train/generator.py, evaluate/sql.py,
                  generation.py, judge.py (M6), hub (model card + push, M7),
                  pipeline, cli
examples/toy/     the toy ticket-triage task used by the demo and the tests
examples/toy-extract/  the toy restaurant-slot extraction task (demo and tests)
examples/toy-generate/ the toy text-to-SQL generation task (demo and tests)
scripts/          demo.sh, wheel_check.sh (clean-venv install of the built wheel)
tests/            unit + integration; tests/recordings/ holds real Claude
                  responses replayed offline (empty until `make record`)
eval/             classify.py, extract.py and generate.py (the parity harnesses; wikisql.py
                  loads WikiSQL), extract_cascade.py (M12, the tagger's threshold),
                  calibration.py,
                  export.py and bench.py (M5, on the student student.py finds), coverage.py
                  (M10, the bound and threshold-search simulation), shift.py
                  and shift.yaml (M9, the bundle off-distribution), data.py
                  (pinned dataset loaders), render.py (RESULTS.md tables),
                  classify/*.yaml, extract/*.yaml and generate/*.yaml (dataset, sample, ceiling
                  prompt, task spec), labels/ (recorded teacher labels),
                  results/, spikes/
docs/             notes/ (spikes)
notebooks/        anvil_gpu.ipynb, the Colab GPU run (M6)
```

## Conventions

- **Typed.** mypy strict on `anvil/` and `tests/`. No `Any` without a comment.
- **Formatted and linted** with ruff (line length 99). Imports sorted.
- **Tests are offline, fast and deterministic.** Network is mocked or
  replayed; models in tests are tiny random-init configs, never downloads;
  seeds are explicit. A test that needs a key is marked `live` and skipped
  without one.
- **Config via environment variables** with the `ANVIL_` prefix, read in one
  place (`anvil/settings.py`). No secrets in the repo, ever. The disk cache
  under `.anvil/` is gitignored.
- **Structured logging** (JSON lines) through `anvil/log.py`; no bare
  `print` outside the CLI's final summary.
- **Errors are specific.** Raise `AnvilError` subclasses with a message that
  says what to do; the CLI maps them to exit code 2 for configuration and
  1 for runtime failures.
- **Numbers are measured.** A number in README or RESULTS.md must come from
  a command in this file, with its raw output committed. If it was not run,
  it goes under "To measure" with the command and a cost estimate.
- **Teacher calls are cached and accounted.** Every call records tokens and
  dollars; re-running an eval with the same prompt, model and inputs costs
  nothing.

## Commits and PRs

- One logical step per commit, imperative subject under 72 characters, a
  short body only when the why is not obvious. Fold fix-ups into the commit
  they fix before pushing.
- Each milestone is one PR from `dev/<milestone>` into `main`. The PR body
  states what was built, what was measured (with the commands and output),
  and what was left out and why.
- Tick a milestone in PLAN.md only when its definition of done is met and
  `make check` is green in CI.
- Releases follow [RELEASE.md](RELEASE.md): a published GitHub release runs
  `.github/workflows/release.yml` (wheel check, PyPI by trusted publishing,
  both images to GHCR). Nothing publishes from a push or a pull request.

## Quality gates

A PR is mergeable when:

1. `make check` passes locally and in CI (lint, types, tests);
2. `make demo` runs offline;
3. every new number in RESULTS.md has its raw output under `eval/results/`;
4. no file under `.anvil/`, no credentials, no `CLAUDE.md`/`AGENTS.md`/`.claude/`/`.agent/` is tracked;
5. PLAN.md reflects the state of the milestone.
