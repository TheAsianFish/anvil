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
| `make eval-classify` / `eval-extract` / `eval-generate` | parity evals (M2, M4, M6); write `eval/results/` and refresh RESULTS.md. `eval-classify` and `eval-extract` take `TEACHER=auto\|claude\|replay\|gold` (`auto` replays committed labels, else calls Claude when a key is set) |
| `make results` | re-render the RESULTS.md tables from `eval/results/` (a test fails when they are stale) |
| `make eval-frontier` / `eval-select` / `eval-calibration` / `eval-export` | the frontier, selection (M3), calibration and export (M5) studies |
| `make bench` | latency, throughput and cost table (M5) |
| `make docker` | build the CLI image (M5 adds the slim serving image) |
| `make spike-onnx` | the M1 ModernBERT ONNX export spike (downloads ModernBERT-base) |

Every eval and bench writes raw output under `eval/results/<name>/` with the
command, git SHA, models, seed, wall time and dollars spent. RESULTS.md is
regenerated from those files by `make results` (`python -m eval.render`),
never edited by hand for numbers: generated tables sit between
`<!-- BEGIN generated:... -->` markers, and the M1 spike table, which predates
the renderer, names its source file. Eval dependencies (`datasets`) are in the
`eval` dependency group, which `make eval-*` installs on demand; CI does not.
`make eval-extract` reads its dataset's raw files with `huggingface_hub` and
needs no extra group.
Labels a live eval pays for are committed under `eval/labels/` so every later
run replays them offline and for free.

Environment variables (all optional, read in `anvil/settings.py`):
`ANVIL_HOME` (state directory, default `.anvil`), `ANVIL_TEACHER_WORKERS`
(concurrent teacher requests, default 4), `ANVIL_RECORD_DIR` (also write
every live teacher response there as a replayable recording),
`ANVIL_LOG_LEVEL` (default `INFO`; logs are JSON lines on stderr).

## Layout

```
anvil/            package: spec, teacher/, store, students/, train/, evaluate/,
                  pipeline, cli (M2+ add align, select, loop, calibrate, export,
                  serve, hub)
examples/toy/     the toy ticket-triage task used by the demo and the tests
examples/toy-extract/  the toy restaurant-slot extraction task (demo and tests)
scripts/          demo.sh
tests/            unit + integration; tests/recordings/ holds real Claude
                  responses replayed offline (empty until `make record`)
eval/             classify.py and extract.py (the parity harnesses), data.py
                  (pinned dataset loaders), render.py (RESULTS.md tables),
                  classify/*.yaml and extract/*.yaml (dataset, sample, ceiling
                  prompt, task spec), labels/ (recorded teacher labels),
                  results/, spikes/
docs/             notes/ (spikes); M7 adds screenshots
notebooks/        the Colab GPU run (M6)
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

## Quality gates

A PR is mergeable when:

1. `make check` passes locally and in CI (lint, types, tests);
2. `make demo` runs offline;
3. every new number in RESULTS.md has its raw output under `eval/results/`;
4. no file under `.anvil/`, no credentials, no `CLAUDE.md`/`AGENTS.md`/`.claude/`/`.agent/` is tracked;
5. PLAN.md reflects the state of the milestone.
